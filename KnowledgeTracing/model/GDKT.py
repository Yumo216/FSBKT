import torch
import torch.nn as nn
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.GCN.Bias import BiasEmbedder

class GDKT(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim, device,
                 graph_encoder, x_init):
        super(GDKT, self).__init__()

        self.device = device
        self.graph_encoder = graph_encoder
        self.x_init = nn.Parameter(x_init.to(device), requires_grad=True)

        self.bias_embedder = BiasEmbedder(emb_dim, device)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.gru = nn.RNN(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # ramdom emd
        self.emb_layer = nn.Embedding(2 * C.QUES + 1, emb_dim)

    def forward(self, x, _):  # 现在 q_seq 是 [B, L]，用于提取题目嵌入

        base_emb = self.graph_encoder(self.x_init).float()  # [Q, D]
        full_emb_2Q = self.bias_embedder(base_emb)          # [2Q, D]
        # full_emb_2Q = self.emb_layer.weight  # shape: [2Q, D]

        x = x.long()
        struct_emb = full_emb_2Q[x - 1]                            # [B, L, D]
        rand_emb = self.emb_layer(x)
        emb = struct_emb + rand_emb                                 # [B, L, D] 题目嵌入 + 随机嵌入
        x_e = self.fc1(emb)                                 # [B, L, H]
        out, _ = self.gru(x_e)                              # [B, L, H]
        logit = self.fc2(out)                               # [B, L, output_dim]

        return logit

