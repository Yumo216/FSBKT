import torch
import torch.nn as nn


class BiasEmbedder(nn.Module):
    def __init__(self, emb_dim, device):
        super(BiasEmbedder, self).__init__()
        self.emb_dim = emb_dim
        self.device = device


        self.correct_bias = nn.Parameter(torch.randn(1, emb_dim) * 0.01)
        self.incorrect_bias = nn.Parameter(torch.randn(1, emb_dim) * 0.01)

    def forward(self, base_emb):
        # right_emb = self.correct_proj(base_emb)
        # wrong_emb = self.incorrect_proj(base_emb)
        wrong_emb = base_emb + self.incorrect_bias
        right_emb = base_emb + self.correct_bias
        padding = torch.zeros((1, self.emb_dim), device=self.device)
        full_emb_2Q = torch.cat([wrong_emb, right_emb, padding], dim=0)
        return full_emb_2Q