import torch
import torch.nn as nn
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.GCN.Bias import BiasEmbedder


class DKT_GAT(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, output_dim, device,
                 graph_encoder, x_init):
        super(DKT_GAT, self).__init__()

        self.device = device
        self.graph_encoder = graph_encoder
        self.x_init = x_init.to(device)

        self.bias_embedder = BiasEmbedder(emb_dim, device)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.gru = nn.RNN(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.emb_layer = nn.Embedding(2 * C.QUES, emb_dim)

    def forward(self, x, q_seq):  # 现在 q_seq 是 [B, L]，用于提取题目嵌入
        base_emb = self.graph_encoder(self.x_init).float()  # [Q, D]
        # full_emb_2Q = self.bias_embedder(base_emb)          # [2Q+1, D]

        full_emb_2Q = self.emb_layer.weight  # shape: [2Q, D]

        x = x.long()
        emb = full_emb_2Q[x - 1]                            # [B, L, D]
        x_e = self.fc1(emb)                                 # [B, L, H]
        out, _ = self.gru(x_e)                              # [B, L, H]
        logit = self.fc2(out)                               # [B, L, output_dim]

        # ===== 新增部分：提取行为原型和子图嵌入 =====
        student_proto = out[:, -1, :]                       # [B, H]

        q_seq = q_seq.long()
        q_emb = base_emb[q_seq]                             # [B, L, D] 从嵌入矩阵里拿出这些题目的嵌入
        subgraph_proto = q_emb.mean(dim=1)                  # [B, D] 聚合（如平均）得到一个子图表示（或叫知识上下文）

        enhanced_proto = torch.cat([student_proto, subgraph_proto], dim=-1)  # [B, H + D]

        # summary = torch.mean(logit, dim=1, keepdim=True)  # 从  AUC: 79.25%  ACC: 71.53%
        # logit = logit + summary   # 提升到  AUC: 81.65%  ACC: 73.90%

        return logit, enhanced_proto
