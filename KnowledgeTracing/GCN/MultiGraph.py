import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv  # More GNN modules can be added if needed


class GraphEmbedder(nn.Module):
    """
    A general graph embedder that supports GAT and GCN.

    It takes ``x``, ``edge_index``, and optional ``edge_weight`` as input,
    and is used to enhance question representations with graph structure
    over the question-question graph.
    """

    def __init__(
        self,
        model_type,
        in_dim,
        hidden_dim,
        out_dim,
        device,
        edge_index,
        edge_weight=None,
        dropedge_rate=0.2,
    ):
        """
        Args:
            model_type (str): "GAT" or "GCN".
            in_dim (int): Input feature dimension.
            hidden_dim (int): Hidden feature dimension.
            out_dim (int): Output dimension of the final embeddings.
            edge_index (Tensor): Graph structure in PyG format with shape [2, num_edges].
            edge_weight (Tensor, optional): Edge weights with shape [num_edges],
                used only for GCN.
        """
        super(GraphEmbedder, self).__init__()
        self.model_type = model_type
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.dropedge_rate = dropedge_rate

        assert (self.edge_index >= 0).all(), "edge_index contains negative values"

        if self.model_type == "GAT":
            self.gnn1 = GATConv(in_dim, hidden_dim, heads=2, concat=False)
            self.gnn2 = GATConv(hidden_dim, out_dim, heads=1, concat=False)

        elif self.model_type == "GATv2":
            self.gnn1 = GATv2Conv(in_dim, hidden_dim, heads=2, concat=False)
            self.gnn2 = GATv2Conv(hidden_dim, out_dim, heads=1, concat=False)

        elif self.model_type == "GCN":
            self.gnn1 = GCNConv(in_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, out_dim)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        """
        Args:
            x (Tensor): Initial node features with shape [num_nodes, in_dim].

        Returns:
            Tensor: Graph-enhanced node embeddings with shape [num_nodes, out_dim].
        """
        # DropEdge: randomly remove a subset of edges during training.
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

        if self.model_type in ["GAT", "GATv2"]:
            x1 = self.gnn1(x, edge_index)
            x = F.elu(x1)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.gnn2(x, edge_index)

        elif self.model_type == "GCN":
            x = F.relu(self.gnn1(x, edge_index, edge_weight))
            x = self.gnn2(x, edge_index, edge_weight)

        return x