import torch
import torch.nn as nn

from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.GCN.Bias import BiasEmbedder


class DKT_GAT(nn.Module):
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        layer_dim,
        output_dim,
        device,
        graph_encoder,
        x_init,
    ):
        super(DKT_GAT, self).__init__()

        self.device = device
        self.graph_encoder = graph_encoder
        self.x_init = nn.Parameter(x_init.to(device), requires_grad=True)

        """
        # Behavior correction module: lucky guesses and careless mistakes
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
        """

        self.bias_embedder = BiasEmbedder(emb_dim, device)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.gru = nn.RNN(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Random embedding
        self.emb_layer = nn.Embedding(2 * C.QUES + 1, emb_dim)

    def forward(self, x, q_seq):  # q_seq is now [B, L] and is used to extract question embeddings
        base_emb = self.graph_encoder(self.x_init).float()  # [Q, D]
        full_emb_2Q = self.bias_embedder(base_emb)  # [2Q, D]

        # full_emb_2Q = self.emb_layer.weight  # w/o Graph Embedding
        # full_emb_2Q = self.emb_layer.weight  # shape: [2Q, D]

        x = x.long()
        struct_emb = full_emb_2Q[x - 1]  # [B, L, D]
        rand_emb = self.emb_layer(x)
        emb = 0.1 * struct_emb + rand_emb  # [B, L, D] question embedding + random embedding
        x_e = self.fc1(emb)  # [B, L, H]
        out, _ = self.gru(x_e)  # [B, L, H]
        logit = self.fc2(out)  # [B, L, output_dim]

        # ===== Added part: extract behavior prototypes and subgraph embeddings =====
        student_proto = out[:, -1, :]  # [B, H]
        student_proto_noisy = student_proto

        # Add noise to the behavior prototypes
        # laplace_noise = torch.distributions.Laplace(0, scale=0.1).sample(student_proto.shape).to(student_proto.device)
        # student_proto_noisy = student_proto + laplace_noise

        q_seq = q_seq.long()
        q_emb = base_emb[q_seq]  # [B, L, D] Retrieve the embeddings of the corresponding questions
        subgraph_proto = q_emb.mean(dim=1)  # [B, D] Aggregate them (e.g., mean pooling) into a subgraph representation or knowledge-context vector

        """
        # ===== Behavior correction: lucky guesses + careless mistakes =====
        guess_score = self.guess_head(student_proto)             # [B, 1]
        careless_score = self.careless_head(student_proto)       # [B, 1]

        # Expand to all time steps (questions) and add a behavior bias at each step
        guess_score = guess_score.unsqueeze(1).expand(-1, logit.size(1), -1)  # [B, L, 1]
        careless_score = careless_score.unsqueeze(1).expand(-1, logit.size(1), -1)  # [B, L, 1]

        # Adjust the logits (with tunable weights)
        logit = logit + 0.1 * guess_score - 0.1 * careless_score
        # Another possible variant:
        # logit = logit - 0.1 * guess_score + 0.1 * careless_score
        """

        # enhanced_proto = torch.cat([student_proto, subgraph_proto], dim=-1)  # [B, H + D]

        summary = torch.mean(logit, dim=1, keepdim=True)
        logit = logit + summary

        return logit, student_proto_noisy, subgraph_proto