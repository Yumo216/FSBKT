# KnowledgeTracing/aggregators/peer_init.py
import torch
import torch.nn.functional as F
from typing import List, Dict

@torch.no_grad()
def _state_weighted_sum(states: List[Dict[str, torch.Tensor]], weights: torch.Tensor) -> Dict[str, torch.Tensor]:
    """对一组 state_dict 做按权重加权求和（结构/形状逐键一致）。"""
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
    ref_protos: List[torch.Tensor],   # 用来度量相似度的“参考原型”（推荐用行为原型）
    topM: int = 8,
    tau: float = 10.0,                # softmax 温度（越大越“只看最像的几个”）
    use_cpu: bool = False             # 显存紧张可设 True：在CPU上算相似度与混合
) -> List[Dict[str, torch.Tensor]]:
    """
    根据原型余弦相似度，给每个客户端生成“同伴混合”初始化权重。
    返回：与 w_locals 等长的 state_dict 列表（下一轮作为 m_locals 使用）。
    """
    device = "cpu" if use_cpu else next(iter(w_locals[0].values())).device
    P = torch.stack([F.normalize(p.to(device), dim=0) for p in ref_protos], dim=0)  # [N, d]
    sim = P @ P.t()  # [N, N] 余弦相似度

    N = sim.size(0)
    inits = []
    for i in range(N):
        # Top-M 邻居（包含自己）
        vals, idx = torch.topk(sim[i], k=min(topM, N))
        w = torch.softmax(tau * vals, dim=0)  # [M]
        states = [w_locals[j] for j in idx.tolist()]
        # 如在CPU混合，需把张量临时搬到 device 再混（确保与权重同设备）
        if use_cpu:
            states = [{k: v.to(device) for k, v in s.items()} for s in states]
        mix = _state_weighted_sum(states, w)
        # 混合完成后回原device（与原始 w_locals 保持一致）
        if use_cpu:
            mix = {k: v.cpu() for k, v in mix.items()}
        inits.append(mix)
    return inits
