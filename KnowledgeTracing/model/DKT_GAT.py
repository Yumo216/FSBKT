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
        self.x_init = nn.Parameter(x_init.to(device), requires_grad=True)
        '''
        # 行为修正模块：猜对和马虎错
        self.guess_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.careless_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        '''
        self.bias_embedder = BiasEmbedder(emb_dim, device)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.gru = nn.RNN(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # ramdom emd
        self.emb_layer = nn.Embedding(2 * C.QUES + 1, emb_dim)

    def forward(self, x, q_seq):  # 现在 q_seq 是 [B, L]，用于提取题目嵌入

        base_emb = self.graph_encoder(self.x_init).float()  # [Q, D]
        full_emb_2Q = self.bias_embedder(base_emb)          # [2Q, D]

        # full_emb_2Q = self.emb_layer.weight  # w/o Graph Embedding

        # full_emb_2Q = self.emb_layer.weight  # shape: [2Q, D]

        x = x.long()
        struct_emb = full_emb_2Q[x - 1]                            # [B, L, D]
        rand_emb = self.emb_layer(x)
        emb = 0.1 * struct_emb + rand_emb                                  # [B, L, D] 题目嵌入 + 随机嵌入
        x_e = self.fc1(emb)                                 # [B, L, H]
        out, _ = self.gru(x_e)                              # [B, L, H]
        logit = self.fc2(out)                               # [B, L, output_dim]

        # ===== 新增部分：提取行为原型和子图嵌入 =====
        student_proto = out[:, -1, :]                       # [B, H]
        student_proto_noisy = student_proto

        # 给行为原型添加噪声
        # laplace_noise = torch.distributions.Laplace(0, scale=0.1).sample(student_proto.shape).to(student_proto.device)
        # student_proto_noisy = student_proto + laplace_noise

        q_seq = q_seq.long()
        q_emb = base_emb[q_seq]                             # [B, L, D] 从嵌入矩阵里拿出这些题目的嵌入
        subgraph_proto = q_emb.mean(dim=1)                  # [B, D] 聚合（如平均）得到一个子图表示（或叫知识上下文）
        '''
        # ===== 行为修正：蒙对 + 马虎错 =====
        guess_score = self.guess_head(student_proto)             # [B, 1]
        careless_score = self.careless_head(student_proto)       # [B, 1]
        # 展开为每个时间步（题目）都加一个行为偏差
        guess_score = guess_score.unsqueeze(1).expand(-1, logit.size(1), -1)  # [B, L, 1]
        careless_score = careless_score.unsqueeze(1).expand(-1, logit.size(1), -1)  # [B, L, 1]

        # 修正 logit（权重可调）
        logit = logit + 0.1 * guess_score - 0.1 * careless_score  # 也可能是 （- 0.1 * guess_score + 0.1 * careless_score）
        '''
        # enhanced_proto = torch.cat([student_proto, subgraph_proto], dim=-1)  # [B, H + D]

        summary = torch.mean(logit, dim=1, keepdim=True)  # 从  AUC: 79.25%  ACC: 71.53%
        logit = logit + summary   # 提升到  AUC: 81.65%  ACC: 73.90%

        return logit, student_proto_noisy, subgraph_proto