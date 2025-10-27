# aggregators/fedavg_proto.py

import torch
from .fedavg import fedavg  # 引用已有参数聚合逻辑

def protoavg(proto_locals, weights=None):
    """
    聚合 prototype 的加权平均
    """
    if weights is None:
        weights = [1 / len(proto_locals)] * len(proto_locals)
    stacked = torch.stack(proto_locals, dim=0)  # [N, D]
    weight_tensor = torch.tensor(weights, device=stacked.device).unsqueeze(1)  # [N, 1]
    proto_glob = torch.sum(stacked * weight_tensor, dim=0) / sum(weights)
    return proto_glob

def fedavg_proto(w_locals, proto_locals, weights=None):
    """
    proto_locals: dict with keys like 'student_proto', 'subgraph_proto', values are proto list
    Returns:
        w_glob, proto_glob (dict with keys)
    """
    w_glob = fedavg(w_locals, weights)

    # 分别聚合
    proto_glob = {}
    for key in proto_locals:
        proto_glob[key] = protoavg(proto_locals[key], weights)

    return w_glob, proto_glob
