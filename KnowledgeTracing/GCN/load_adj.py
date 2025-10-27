import json
import numpy as np
import torch
from collections import defaultdict
from KnowledgeTracing.Constant import Constants as C

def build_graph(json_path):
    # 模型&参数选择
    mode = "normalize"  # 可选: "normalize", "binarize", "threshold", "topk"
    threshold = 1       # 仅mode为"threshold"时有效
    topk = 10           # 仅mode为"topk"时有效

    with open(json_path, 'r') as f:
        data = json.load(f)
    # data = data[0:100]  # 如需采样可取消注释

    question_concepts = defaultdict(set)
    for item in data:
        qid = int(item["question_id"])
        skills = str(item["skill"]).split(',')
        question_concepts[qid].update(int(s) for s in skills if s != '')
    Q = max(question_concepts.keys())  # 题号最大即题目个数（已编号）

    adj = np.zeros((Q, Q), dtype=float)

    for i in range(1, Q + 1):
        for j in range(1, Q + 1):
            if i != j:
                common = question_concepts[i] & question_concepts[j]
                adj[i - 1][j - 1] = len(common)

    if mode == "binarize":
        adj = (adj > 0).astype(float)
    elif mode == "threshold":
        adj = (adj >= threshold).astype(float)
    elif mode == "topk":
        for i in range(adj.shape[0]):
            row = adj[i]
            if np.count_nonzero(row) > topk:
                idx = np.argsort(row)[-topk:]
                mask = np.zeros_like(row, dtype=bool)
                mask[idx] = True
                row[~mask] = 0
                adj[i] = row
        row_sum = adj.sum(axis=1, keepdims=True) + 1e-8
        adj = adj / row_sum
    else:  # 默认normalize
        row_sum = adj.sum(axis=1, keepdims=True) + 1e-8
        adj = adj / row_sum

    edge_index = np.array(np.nonzero(adj))
    edge_weight = adj[edge_index[0], edge_index[1]]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return edge_index, edge_weight