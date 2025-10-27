import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from KnowledgeTracing.BipartiteGCN import BipartiteGNN
from KnowledgeTracing.Constant import Constants as C

class GNNBias(nn.Module):
    def __init__(self, emb_dim, device):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.Q = C.QUES
        self.S = C.SKILL
        self.gnn_model = BipartiteGNN.BiGNN(emb_dim).to(device)

        # 构建图结构 edge_index（固定）
        adj_matrix = pd.read_csv("../../KTDataset/ASSIST/adj_matrix.csv", index_col=0).values
        edge_index = np.array(np.nonzero(adj_matrix))
        edge_index[1] += self.Q  # S 节点偏移
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

        # 构建节点初始特征
        self.nodes_features = torch.cat([
            torch.eye(self.Q, emb_dim),
            torch.eye(self.S, emb_dim)
        ], dim=0).to(device)

        # learnable bias（可训练）
        self.correct_bias = nn.Parameter(torch.randn(1, emb_dim) * 0.01)
        self.incorrect_bias = nn.Parameter(torch.randn(1, emb_dim) * 0.01)

    def forward(self):
        knowledge_emb = self.gnn_model(self.nodes_features, self.edge_index)
        ques_base = knowledge_emb[:self.Q]  # [Q, D]

        # 构造结构感知的 interaction embedding
        wrong_emb = ques_base + self.incorrect_bias     # [Q, D]
        right_emb = ques_base + self.correct_bias       # [Q, D]
        padding = torch.zeros((1, self.emb_dim), device=self.device)
        full_emb = torch.cat([wrong_emb, right_emb, padding], dim=0)  # [2Q+1, D]
        return full_emb