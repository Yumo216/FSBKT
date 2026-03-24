# aggregators/fedavg_proto.py

import torch
from .fedavg import fedavg  # 引用已有参数聚合逻辑

def protoavg(proto_locals, weights=None):
    """
    Compute the weighted average of local prototypes.
    """
    if weights is None:
        weights = [1 / len(proto_locals)] * len(proto_locals)

    stacked = torch.stack(proto_locals, dim=0)  # [N, D]
    weight_tensor = torch.tensor(weights, device=stacked.device).unsqueeze(1)  # [N, 1]
    proto_glob = torch.sum(stacked * weight_tensor, dim=0) / sum(weights)

    return proto_glob


def fedavg_proto(w_locals, proto_locals, weights=None):
    """
    Aggregate model parameters and prototype representations.

    Args:
        w_locals: List of local model state_dict objects.
        proto_locals: A dictionary whose keys are prototype names
            (e.g., ``student_proto`` or ``subgraph_proto``), and whose
            values are lists of local prototypes.
        weights: Optional aggregation weights for clients.

    Returns:
        A tuple ``(w_glob, proto_glob)``, where ``w_glob`` is the aggregated
        global model and ``proto_glob`` is a dictionary of aggregated
        global prototypes.
    """
    w_glob = fedavg(w_locals, weights)

    # Aggregate each prototype type separately.
    proto_glob = {}
    for key in proto_locals:
        proto_glob[key] = protoavg(proto_locals[key], weights)

    return w_glob, proto_glob
