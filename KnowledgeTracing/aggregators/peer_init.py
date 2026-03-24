# KnowledgeTracing/aggregators/peer_init.py
import torch
import torch.nn.functional as F
from typing import List, Dict

@torch.no_grad()
def _state_weighted_sum(
    states: List[Dict[str, torch.Tensor]],
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute a weighted sum of state_dict objects.

    All state_dict objects must have exactly the same keys and tensor shapes.
    """
    out = {}
    for k in states[0].keys():
        acc = 0.0
        for i, s in enumerate(states):
            acc = acc + s[k] * weights[i]
        out[k] = acc
    return out


@torch.no_grad()
def peer_init(
    w_locals: List[Dict[str, torch.Tensor]],
    ref_protos: List[torch.Tensor],  # Reference prototypes for similarity scoring (behavior prototypes are recommended)
    topM: int = 8,
    tau: float = 10.0,  # Softmax temperature (larger values focus more on the most similar peers)
    use_cpu: bool = False,  # Set to True when GPU memory is limited: compute similarity and mixing on CPU
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate peer-mixed initialization weights for each client based on
    prototype cosine similarity.

    Returns:
        A list of state_dict objects with the same length as ``w_locals``,
        intended to be used as ``m_locals`` in the next training round.
    """
    device = "cpu" if use_cpu else next(iter(w_locals[0].values())).device
    P = torch.stack([F.normalize(p.to(device), dim=0) for p in ref_protos], dim=0)  # [N, d]
    sim = P @ P.t()  # [N, N] cosine similarity matrix

    N = sim.size(0)
    inits = []

    for i in range(N):
        # Top-M neighbors (including the client itself)
        vals, idx = torch.topk(sim[i], k=min(topM, N))
        w = torch.softmax(tau * vals, dim=0)  # [M]
        states = [w_locals[j] for j in idx.tolist()]

        # When mixing on CPU, temporarily move tensors to the target device
        # so that they are on the same device as the weights.
        if use_cpu:
            states = [{k: v.to(device) for k, v in s.items()} for s in states]

        mix = _state_weighted_sum(states, w)

        # Move mixed parameters back after aggregation to keep consistency
        # with the original ``w_locals`` device placement.
        if use_cpu:
            mix = {k: v.cpu() for k, v in mix.items()}

        inits.append(mix)

    return inits