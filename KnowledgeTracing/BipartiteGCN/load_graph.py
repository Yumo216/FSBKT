from KnowledgeTracing.Constant import Constants as C
import pandas as pd
import torch
import numpy as np

def load_graph(path, emb_dim, device):
    adj_matrix = pd.read_csv(path, index_col=0).values
    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index[1] += C.QUES  # skill index 偏移
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)

    # 构造初始节点特征
    node_features = torch.cat([
        torch.eye(C.QUES, emb_dim),
        torch.eye(C.SKILL, emb_dim)
    ], dim=0).to(device)

    return edge_index, node_features
