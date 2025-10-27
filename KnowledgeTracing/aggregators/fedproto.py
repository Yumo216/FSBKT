import torch

@torch.no_grad()
def fedproto(proto_locals, device=None, weights=None):
    """
    FedProto: 聚合行为原型 (student_proto)
    Args:
        proto_locals: List[Tensor[K, D]]
    Returns:
        dict {"proto": Tensor[K, D]}
    """
    assert proto_locals and len(proto_locals) > 0
    proto_locals = [p.to(device) for p in proto_locals]

    N, K, D = len(proto_locals), *proto_locals[0].shape
    if weights is None:
        weights = torch.ones(N, dtype=torch.float32, device=device) / N
    else:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        weights = weights / weights.sum()

    glob_proto = torch.zeros((K, D), device=device)
    for w, p in zip(weights, proto_locals):
        glob_proto += w * p

    return {"proto": glob_proto}
