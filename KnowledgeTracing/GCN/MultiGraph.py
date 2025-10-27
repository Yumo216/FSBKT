import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv  # 可扩展更多 GNN 模块


class GraphEmbedder(nn.Module):
    """
    通用图嵌入器，支持 GAT 和 GCN，接收 x、edge_index、edge_weight（可选）
    用于图结构增强题目表示（Q-Q 图）
    """
    def __init__(self, model_type, in_dim, hidden_dim, out_dim, device, edge_index, edge_weight=None, dropedge_rate=0.2):
        """
        Args:
            model_type (str): 'GAT' or 'GCN'
            in_dim (int): 输入特征维度
            hidden_dim (int): 中间层维度
            out_dim (int): 输出维度（最终嵌入）
            edge_index (Tensor): PyG 格式图结构 [2, num_edges]
            edge_weight (Tensor, optional): [num_edges] 权重（仅 GCN 用）
        """
        super(GraphEmbedder, self).__init__()
        self.model_type = model_type
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.dropedge_rate = dropedge_rate

        assert (self.edge_index >= 0).all(), "edge_index contains negative values"

        if self.model_type == 'GAT':
            self.gnn1 = GATConv(in_dim, hidden_dim, heads=2, concat=False)
            self.gnn2 = GATConv(hidden_dim, out_dim, heads=1, concat=False)



        elif self.model_type == 'GATv2':
            self.gnn1 = GATv2Conv(in_dim, hidden_dim, heads=2, concat=False)
            self.gnn2 = GATv2Conv(hidden_dim, out_dim, heads=1, concat=False)

        elif self.model_type == 'GCN':
            self.gnn1 = GCNConv(in_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, out_dim)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        """
        Args:
            x (Tensor): 节点初始特征 [num_nodes, in_dim]

        Returns:
            Tensor: 图结构增强后的嵌入 [num_nodes, out_dim]
        """
        # DropEdge: 训练时随机丢弃部分边
        edge_index = self.edge_index
        edge_weight = self.edge_weight
        if self.training and self.dropedge_rate > 0:
            num_edges = edge_index.size(1)
            keep_num = int(num_edges * (1 - self.dropedge_rate))
            perm = torch.randperm(num_edges, device=edge_index.device)
            idx = perm[:keep_num]
            edge_index = edge_index[:, idx]
            if edge_weight is not None:
                edge_weight = edge_weight[idx]

        if self.model_type in ['GAT', 'GATv2']:
            x1 = self.gnn1(x, edge_index)
            x = F.elu(x1)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.gnn2(x, edge_index)

        elif self.model_type == 'GCN':
            x = F.relu(self.gnn1(x, edge_index, edge_weight))
            x = self.gnn2(x, edge_index, edge_weight)

        return x
