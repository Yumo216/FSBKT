import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn

class BiGNN(nn.Module):
    def __init__(self, embedding_dim):
        super(BiGNN, self).__init__()
        self.conv1 = SAGEConv((-1, -1), embedding_dim)  # 处理二部图 (Q, S)
        self.conv2 = SAGEConv((-1, -1), embedding_dim)  # 第二层传播 Q-S 信息

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))  # 传播 Q-S 信息
        x = self.conv2(x, edge_index)  # 继续传播
        return x  # 返回所有 Q+S embedding
