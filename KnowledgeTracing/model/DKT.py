import torch
import torch.nn as nn
from KnowledgeTracing.BipartiteGCN.GEmbedding import GrapheEmbedding
from KnowledgeTracing.Constant import Constants as C


class DKT(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim, device, edge, node, use_gnn_emb=False):
        super(DKT, self).__init__()

        self.interaction_emb = nn.Embedding(2 * C.QUES + 1, emb_dim)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.use_gnn_emb = use_gnn_emb
        self.device = device

        self.gnn_embedder = GrapheEmbedding(emb_dim, edge, node, device)

        ques_emb = self.gnn_embedder()  # [2Q+1, D]

        self.gru = nn.RNN(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=2,
                                          batch_first=True)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim)
        self.predict_linear = nn.Linear(hidden_dim, C.QUES, bias=True)

    def forward(self, x, _):  # shape of input: [batch_size, length, 2q ]   [batch_size, length]

        if self.use_gnn_emb:
            ques_emb = self.gnn_embedder()
            x_idx = x.long() - 1
            emb = ques_emb[x_idx]
        else:
            emb = self.interaction_emb(x)

        # x = self.interaction_emb(x)  # random initial
        x_e = self.fc1(emb)
        out, _ = self.gru(x_e)  # [bs,l,d]
        logit = self.fc2(out)  # [bs,l,1]
        # summary = torch.mean(logit, dim=1, keepdim=True)  # 从  AUC: 79.25%  ACC: 71.53%
        # logit = logit + summary   # 提升到  AUC: 81.65%  ACC: 73.90%
        return logit



