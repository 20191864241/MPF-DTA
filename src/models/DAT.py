# -*- coding: utf-8 -*-
"""
MPF-DTA: Multi-scale Protein Information Fusion Network for Drug-Target Affinity prediction

Modules:
    1. Drug representation: Transformer Encoder (from transformer.py)
    2. Protein-1D: ESM1b embeddings + Bi-LSTM
    3. Protein-3D: Distance-aware GNN with dense connections
    4. Multi-scale fusion: Cross-Attention (3D queries 1D) + Contrastive Learning alignment
    5. DTA predictor: Joint Attention + FC layers
"""
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from src.models.transformer import Transformer


# ==============================================================================
# Module 1: Distance-aware GNN (论文 Section 2.1, 公式5-7)
# ==============================================================================

class DistanceAwareGNNLayer(nn.Module):
    """单层距离感知消息传递，对应公式(5)(6)
    m_ij = φ_m(h_i, h_j, d_ij)
    h_i^(l+1) = φ_u(h_i, Σ α_ij * m_ij)
    """
    def __init__(self, in_dim, out_dim, edge_dim=1):
        super(DistanceAwareGNNLayer, self).__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + edge_dim, 1),
            nn.LeakyReLU(0.2)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, h, adj, dist_matrix):
        """
        h:           (B, N, in_dim)  节点特征
        adj:         (B, N, N)       邻接矩阵 (0/1)
        dist_matrix: (B, N, N)       距离矩阵 (实际距离值)
        """
        B, N, D = h.shape

        h_i = h.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)
        d_ij = dist_matrix.unsqueeze(-1)          # (B, N, N, 1)

        msg_input = torch.cat([h_i, h_j, d_ij], dim=-1)  # (B, N, N, 2D+1)

        msg = self.msg_mlp(msg_input)     # (B, N, N, out_dim)
        attn = self.attn_mlp(msg_input)   # (B, N, N, 1)

        mask = adj.unsqueeze(-1)  # (B, N, N, 1)
        attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=2)  # 沿邻居维度归一化

        agg = (attn * msg * mask).sum(dim=2)  # (B, N, out_dim)

        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))  # (B, N, out_dim)
        return h_new


class DistanceAwareGNN(nn.Module):
    """多层距离感知GNN + 密集连接，对应公式(7)
    h_i^(l) = φ_u^(l)([h_i^(0) || h_i^(1) || ... || h_i^(l-1)])
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, edge_dim=1):
        super(DistanceAwareGNN, self).__init__()
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_d = hidden_dim * (l + 1)
            self.layers.append(DistanceAwareGNNLayer(in_d, hidden_dim, edge_dim))
        self.out_proj = nn.Linear(hidden_dim * (num_layers + 1), output_dim)

    def forward(self, x, adj, dist_matrix):
        """
        x:           (B, N, input_dim)  节点特征 (Cα坐标+生化特征)
        adj:         (B, N, N)          邻接矩阵
        dist_matrix: (B, N, N)          距离矩阵
        return:      (B, N, output_dim) 残基级特征
        """
        h = self.input_proj(x)
        all_h = [h]
        for layer in self.layers:
            h_cat = torch.cat(all_h, dim=-1)
            h_new = layer(h_cat, adj, dist_matrix)
            all_h.append(h_new)
        h_final = torch.cat(all_h, dim=-1)
        return self.out_proj(h_final)


# ==============================================================================
# Module 2: Cross-Attention (论文 Section 2.2, 公式10-11)
# ==============================================================================

class CrossAttention(nn.Module):
    """3D残基级特征查询1D序列特征的交叉注意力
    Attention(Q_3D, K_1D, V_1D) = softmax(Q_3D K_1D^T / sqrt(d_k)) V_1D
    O = Attention(...) ⊕ H_3D  (残差连接)
    """
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_3d, h_1d):
        """
        h_3d: (B, n_residues, dim) 3D残基级特征 → Query
        h_1d: (B, seq_len, dim)    1D序列级特征 → Key, Value
        return: (B, n_residues, dim)
        """
        attn_out, _ = self.mha(h_3d, h_1d, h_1d)
        out = self.norm(attn_out + h_3d)
        return out


# ==============================================================================
# Module 3: Contrastive Learning (论文 Section 2.2, 公式12)
# ==============================================================================

class ContrastiveLoss(nn.Module):
    """InfoNCE对比学习损失，对齐1D和3D蛋白质表示
    L = -log( exp(sim(v_i, v_j)/τ) / Σ exp(sim(v_i, v_k)/τ) )
    """
    def __init__(self, dim, proj_dim=128, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.proj_1d = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        self.proj_3d = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, h_1d, h_3d):
        """
        h_1d: (B, dim) 1D蛋白质全局表示
        h_3d: (B, dim) 3D蛋白质全局表示
        return: 标量对比损失
        """
        z_1d = F.normalize(self.proj_1d(h_1d), dim=-1)
        z_3d = F.normalize(self.proj_3d(h_3d), dim=-1)

        B = z_1d.size(0)
        z = torch.cat([z_1d, z_3d], dim=0)  # (2B, proj_dim)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
        mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
        sim.masked_fill_(mask, -9e15)

        loss = F.cross_entropy(sim, labels)
        return loss


# ==============================================================================
# Module 4: Joint Attention (论文 Section 2.3, 公式13-16)
# ==============================================================================

class JointAttention(nn.Module):
    """药物-蛋白质联合注意力
    N_ij = Tanh(W_d h_i^d + W_p h_j^p)
    A_ij = softmax(N_ij) along protein dim
    f_ij = Tanh(W_a h_i^d + W_b h_j^p)
    X_int = ΣΣ A_ij · f_ij
    """
    def __init__(self, drug_dim, prot_dim, hidden_dim, output_dim):
        super(JointAttention, self).__init__()
        self.W_d = nn.Linear(drug_dim, hidden_dim, bias=False)
        self.W_p = nn.Linear(prot_dim, hidden_dim, bias=False)
        self.W_a = nn.Linear(drug_dim, output_dim, bias=False)
        self.W_b = nn.Linear(prot_dim, output_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, X_d, X_p):
        """
        X_d: (B, n_d, d_d) 药物token级表示
        X_p: (B, n_p, d_p) 蛋白质残基级表示
        return: X_int (B, output_dim)
        """
        # (B, n_d, 1, h) + (B, 1, n_p, h) → (B, n_d, n_p, h)
        N = torch.tanh(self.W_d(X_d).unsqueeze(2) + self.W_p(X_p).unsqueeze(1))
        A = torch.softmax(self.score_proj(N).squeeze(-1), dim=-1)  # (B, n_d, n_p)

        f = torch.tanh(self.W_a(X_d).unsqueeze(2) + self.W_b(X_p).unsqueeze(1))  # (B, n_d, n_p, out)
        X_int = (A.unsqueeze(-1) * f).sum(dim=1).sum(dim=1)  # (B, output_dim)
        return X_int


# ==============================================================================
# Main Model: MPF-DTA (DAT3)
# ==============================================================================

class DAT3(nn.Module):
    def __init__(self, embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout_rate,
                 alpha, n_heads, graph_input_dim=78, rnn_layers=2,
                 attn_type='dotproduct', vocab=26, smile_vocab=63, is_pretrain=True,
                 is_drug_pretrain=False, n_extend=1, num_feature_xd=156, output_dim=128,
                 dropout=0.2, feature_dim=1024, DENSE_DIM=16, ATTENTION_HEADS=4,
                 prot_node_dim=3, gnn_hidden_dim=128, gnn_layers=3,
                 cross_attn_heads=8, cl_proj_dim=128, cl_temperature=0.07,
                 joint_hidden_dim=128):
        super(DAT3, self).__init__()

        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # ===== Drug: Transformer Encoder =====
        self.smiles_vocab = smile_vocab
        self.encoder = Transformer(256, 256)
        self.drug_seq_fc = nn.Linear(256, output_dim)

        # ===== Protein-1D: ESM1b + Bi-LSTM =====
        self.is_pretrain = is_pretrain
        if not is_pretrain:
            self.vocab = vocab
            self.embed = nn.Embedding(vocab + 1, embedding_dim, padding_idx=vocab)
        self.sentence_input_fc = nn.Linear(embedding_dim, rnn_dim)
        self.encode_rnn = nn.LSTM(rnn_dim, rnn_dim, rnn_layers, batch_first=True,
                                  bidirectional=True, dropout=dropout_rate)
        self.rnn_out_fc = nn.Linear(rnn_dim * 2, output_dim)

        # ===== Protein-3D: Distance-aware GNN =====
        self.dist_gnn = DistanceAwareGNN(
            input_dim=prot_node_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=output_dim,
            num_layers=gnn_layers
        )

        # ===== Multi-scale fusion: Cross-Attention =====
        self.cross_attn = CrossAttention(dim=output_dim, n_heads=cross_attn_heads, dropout=dropout)

        # ===== Multi-scale fusion: Contrastive Learning =====
        self.contrastive = ContrastiveLoss(dim=output_dim, proj_dim=cl_proj_dim,
                                           temperature=cl_temperature)

        # ===== DTA Predictor: Joint Attention =====
        self.joint_attn = JointAttention(
            drug_dim=output_dim,
            prot_dim=output_dim,
            hidden_dim=joint_hidden_dim,
            output_dim=output_dim
        )

        # ===== Affinity prediction: FC [256, 128, 64] =====
        self.predictor = nn.Sequential(
            nn.Linear(output_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, protein, smiles, prot_node_feat, prot_adj, prot_dist_matrix):
        """
        Args:
            protein:          list of (seq_len, embedding_dim) tensors, ESM1b embeddings
            smiles:           list of token id tensors, SMILES sequences
            prot_node_feat:   list of (N, prot_node_dim) tensors, 3D structure node features (Cα coords + biochemical)
            prot_adj:         list of (N, N) tensors, adjacency matrix (d_ij < 8Å → 1)
            prot_dist_matrix: list of (N, N) tensors, distance matrix (actual distances)

        Returns:
            y_pred:  (B,) predicted affinity
            cl_loss: scalar, contrastive learning loss
        """
        batchsize = len(protein)

        # ===== 1. Drug: Transformer Encoder =====
        smiles_lengths = np.array([len(x) for x in smiles])
        max_smi_len = max(smiles_lengths)
        temp = (torch.zeros(batchsize, max_smi_len) * self.smiles_vocab).long()
        for i in range(batchsize):
            temp[i, :len(smiles[i])] = smiles[i]
        smiles_padded = temp.cuda()
        drug_seq = self.encoder(smiles_padded)               # (B, max_smi_len, 256)
        drug_seq_proj = self.drug_seq_fc(drug_seq)            # (B, max_smi_len, output_dim)
        drug_pool = drug_seq_proj.mean(dim=1)                 # (B, output_dim)

        # ===== 2. Protein-1D: ESM1b + Bi-LSTM =====
        protein_stacked = torch.stack(protein)                # (B, seq_len, embedding_dim)
        h_1d = self.sentence_input_fc(protein_stacked)        # (B, seq_len, rnn_dim)
        h_1d_seq, _ = self.encode_rnn(h_1d)                   # (B, seq_len, rnn_dim*2)
        h_1d_seq = self.rnn_out_fc(h_1d_seq)                  # (B, seq_len, output_dim)
        h_1d_pool = h_1d_seq.mean(dim=1)                      # (B, output_dim)

        # ===== 3. Protein-3D: Distance-aware GNN =====
        prot_node_feat = torch.stack(prot_node_feat)          # (B, N, prot_node_dim)
        prot_adj = torch.stack(prot_adj)                      # (B, N, N)
        prot_dist_matrix = torch.stack(prot_dist_matrix)      # (B, N, N)
        h_3d_nodes = self.dist_gnn(prot_node_feat, prot_adj, prot_dist_matrix)
                                                               # (B, N, output_dim)
        h_3d_pool = h_3d_nodes.mean(dim=1)                    # (B, output_dim)

        # ===== 4. Contrastive Learning: 1D↔3D alignment =====
        cl_loss = self.contrastive(h_1d_pool, h_3d_pool)

        # ===== 5. Cross-Attention: 3D queries 1D (residue-level) =====
        h_protein_seq = self.cross_attn(h_3d_nodes, h_1d_seq) # (B, N, output_dim)
        h_protein_pool = h_protein_seq.mean(dim=1)             # (B, output_dim)

        # ===== 6. Joint Attention: Drug × Protein interaction =====
        X_int = self.joint_attn(drug_seq_proj, h_protein_seq)  # (B, output_dim)

        # ===== 7. Prediction =====
        combined = torch.cat([X_int, drug_pool, h_protein_pool], dim=-1)  # (B, output_dim*3)
        y_pred = self.predictor(combined).squeeze(-1)          # (B,)

        return y_pred, cl_loss
