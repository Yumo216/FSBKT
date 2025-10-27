import torch
import torch.nn as nn
from KnowledgeTracing.BipartiteGCN import BipartiteGNN
from KnowledgeTracing.Constant import Constants as C

class GrapheEmbedding(nn.Module):
    def __init__(self, emb_dim, edge_index, node_features, device):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.Q = C.QUES
        self.S = C.SKILL
        self.edge_index = edge_index
        self.node_features = node_features
        self.gnn_model = BipartiteGNN.BiGNN(emb_dim).to(device)

        # learnable bias（可训练）
        self.correct_bias = nn.Parameter(torch.randn(1, emb_dim, device=device) * 0.01)
        self.incorrect_bias = nn.Parameter(torch.randn(1, emb_dim, device=device) * 0.01)

    def forward(self):
        knowledge_emb = self.gnn_model(self.node_features, self.edge_index)
        ques_base = knowledge_emb[:self.Q]  # [Q, D]

        # 构造结构感知的 interaction embedding
        wrong_emb = ques_base + self.incorrect_bias     # [Q, D]
        right_emb = ques_base + self.correct_bias       # [Q, D]
        padding = torch.zeros((1, self.emb_dim), device=self.device)
        full_emb = torch.cat([wrong_emb, right_emb, padding], dim=0)  # [2Q+1, D]
        return full_emb
