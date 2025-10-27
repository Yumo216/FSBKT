# KnowledgeTracing/server_proto.py
import torch
import torch.nn.functional as F

# =========================
# 🟣 MoP: 分簇 + 个性化配方
# =========================
def kmeans_cosine(P: torch.Tensor, K: int, iters: int = 20) -> torch.Tensor:
    """
    P: [N, d] (N个客户端的原型)
    返回: centers: [K, d]
    """
    N, d = P.shape
    idx = torch.randperm(N)[:K]
    centers = F.normalize(P[idx], dim=-1)

    for _ in range(iters):
        sim = P @ centers.t()                 # [N, K]
        assign = sim.argmax(dim=1)            # [N]
        new_centers = []
        for k in range(K):
            mk = (assign == k)
            if mk.sum() == 0:
                new_centers.append(centers[k])  # 空簇：保留
            else:
                c = F.normalize(P[mk].mean(dim=0, keepdim=True), dim=-1)
                new_centers.append(c.squeeze(0))
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    return centers

def mop_personalized(P: torch.Tensor, centers: torch.Tensor, alpha: float = 10.0):
    """
    基于簇中心生成“像我”的混合原型
    P:       [N, d]  (N个客户端原型)
    centers: [K, d]  (K个簇中心)
    返回:
      Pmix:  [N, d]  每客户端MoP原型
      pi:    [N, K]  MoP权重
    """
    Pn = F.normalize(P, dim=-1)
    Cn = F.normalize(centers, dim=-1)
    sim = Pn @ Cn.t()                  # [N, K]
    pi  = F.softmax(alpha * sim, -1)   # [N, K]
    Pmix = pi @ centers                # [N, d]
    return Pmix, pi

# =========================
# 🟡 EMA: 跨轮平滑 + 插值
# =========================
def ema_update(prev_ema: torch.Tensor, new_mean: torch.Tensor, beta: float = 0.2):
    """ p_ema = (1-beta)*prev + beta*new """
    if prev_ema is None:
        return new_mean.detach().clone()
    return (1 - beta) * prev_ema + beta * new_mean

def ema_personalize_from_mop(Pmix: torch.Tensor, p_ema: torch.Tensor,
                             alpha: float = 8.0, gmin: float = 0.1, gmax: float = 0.9):
    """
    在“老师(EMA)”与“我的配方(MoP)”之间插值：
      gamma_i = sigmoid(alpha * cos(Pmix_i, p_ema))
      p_tilde_i = (1-gamma_i)*p_ema + gamma_i*Pmix_i
    返回:
      P_tilde: [N, d]  每客户端最终个性化原型
      gamma:   [N]     插值强度（可用于可视化）
    """
    Pm = F.normalize(Pmix, dim=-1)
    Pe = F.normalize(p_ema, dim=-1)
    cos_sim = torch.sum(Pm * Pe.unsqueeze(0), dim=-1)    # [N]
    gamma = torch.sigmoid(alpha * cos_sim)               # [N]
    gamma = torch.clamp(gamma, gmin, gmax)
    P_tilde = (1 - gamma).unsqueeze(-1) * p_ema.unsqueeze(0) + gamma.unsqueeze(-1) * Pmix
    return P_tilde, gamma

# =============== 一站式：MoP + EMA ===============
def server_prototype_update(student_proto_locals, subgraph_proto_locals,
                            prev_stu_ema, prev_sub_ema,
                            K_stu=4, K_sub=4, beta=0.2, alpha_mop=10.0,
                            alpha_ema=8.0, gmin=0.1, gmax=0.9, device="cpu"):
    # 堆叠
    P_stu = torch.stack(student_proto_locals, dim=0).to(device)  # [N, H]
    P_sub = torch.stack(subgraph_proto_locals, dim=0).to(device) # [N, D]

    # 🟣 MoP
    C_stu = kmeans_cosine(P_stu, K_stu)
    C_sub = kmeans_cosine(P_sub, K_sub)
    Pmix_stu, _ = mop_personalized(P_stu, C_stu, alpha=alpha_mop)
    Pmix_sub, _ = mop_personalized(P_sub, C_sub, alpha=alpha_mop)

    # 🟡 EMA（跨轮平滑后的“老师笔记”）
    stu_mean_new = P_stu.mean(dim=0)
    sub_mean_new = P_sub.mean(dim=0)
    stu_ema = ema_update(prev_stu_ema, stu_mean_new, beta=beta)
    sub_ema = ema_update(prev_sub_ema, sub_mean_new, beta=beta)

    # 🟡 EMA插值：老师 vs 我的配方
    Ptilde_stu, gamma_stu = ema_personalize_from_mop(Pmix_stu, stu_ema, alpha=alpha_ema, gmin=gmin, gmax=gmax)
    Ptilde_sub, gamma_sub = ema_personalize_from_mop(Pmix_sub, sub_ema, alpha=alpha_ema, gmin=gmin, gmax=gmax)

    # 打包返回（列表形式，顺序与客户端顺序一致）
    proto_glob_student_list = [Ptilde_stu[i].detach().cpu() for i in range(Ptilde_stu.size(0))]
    proto_glob_subgraph_list = [Ptilde_sub[i].detach().cpu() for i in range(Ptilde_sub.size(0))]
    state = {
        "prev_stu_ema": stu_ema.detach(),
        "prev_sub_ema": sub_ema.detach(),
        "gamma_stu": gamma_stu.detach().cpu(),
        "gamma_sub": gamma_sub.detach().cpu(),
    }
    return proto_glob_student_list, proto_glob_subgraph_list, state
