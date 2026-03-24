import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn

class BiGNN(nn.Module):
    def __init__(self, embedding_dim):
        super(BiGNN, self).__init__()
        self.conv1 = SAGEConv((-1, -1), embedding_dim)  # bipartite graph  (Q, S)
        self.conv2 = SAGEConv((-1, -1), embedding_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x  # Q+S embedding
