import torch
import torch.nn as nn

class BiasEmbedder(nn.Module):
    def __init__(self, emb_dim, device):
        super(BiasEmbedder, self).__init__()
        self.emb_dim = emb_dim
        self.device = device
        # self.correct_proj = nn.Sequential(  # 让每道题的两种状态嵌入有更复杂的非线性变换，表达力更强
        #     nn.Linear(emb_dim, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, emb_dim)
        # )
        # self.incorrect_proj = nn.Sequential(
        #     nn.Linear(emb_dim, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, emb_dim)
        # )
        self.correct_bias = nn.Parameter(torch.randn(1, emb_dim) * 0.01)  # 加法方式太弱
        self.incorrect_bias = nn.Parameter(torch.randn(1, emb_dim) * 0.01)


    def forward(self, base_emb):
        # right_emb = self.correct_proj(base_emb)
        # wrong_emb = self.incorrect_proj(base_emb)
        wrong_emb = base_emb + self.incorrect_bias  # 直接加法方式太弱
        right_emb = base_emb + self.correct_bias
        padding = torch.zeros((1, self.emb_dim), device=self.device)
        full_emb_2Q = torch.cat([wrong_emb, right_emb, padding], dim=0)
        return full_emb_2Q