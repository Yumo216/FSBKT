# KnowledgeTracing/server_proto.py
import torch
import torch.nn.functional as F

# =========================
# 🟣 MoP: Clustering + Personalized Formulation
# =========================
def kmeans_cosine(P: torch.Tensor, K: int, iters: int = 20) -> torch.Tensor:
    """
    P: [N, d] (N clients)
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
                new_centers.append(centers[k])
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
    Generate a hybrid prototype "like me" based on the cluster center.
    P:       [N, d]  (N client prototypes)
    centers: [K, d]  (K cluster centers)
    Return:
      Pmix:  [N, d]  MoP Prototype for Each Client
      pi:    [N, K]  MoP weight
    """
    Pn = F.normalize(P, dim=-1)
    Cn = F.normalize(centers, dim=-1)
    sim = Pn @ Cn.t()                  # [N, K]
    pi  = F.softmax(alpha * sim, -1)   # [N, K]
    Pmix = pi @ centers                # [N, d]
    return Pmix, pi

# =========================
# 🟡 EMA: Cross-wheel smoothing + interpolation
# =========================
def ema_update(prev_ema: torch.Tensor, new_mean: torch.Tensor, beta: float = 0.2):
    """ p_ema = (1-beta)*prev + beta*new """
    if prev_ema is None:
        return new_mean.detach().clone()
    return (1 - beta) * prev_ema + beta * new_mean

def ema_personalize_from_mop(Pmix: torch.Tensor, p_ema: torch.Tensor,
                             alpha: float = 8.0, gmin: float = 0.1, gmax: float = 0.9):
    """
    Interpolate between "Teacher (EMA)" and "My Recipe (MoP)":
    gamma_i = sigmoid(alpha * cos(Pmix_i, p_ema))
    p_tilde_i = (1-gamma_i)*p_ema + gamma_i*Pmix_i
    Returns:
    P_tilde: [N, d] Final personalized prototype per client
    gamma: [N] Interpolation strength (can be used for visualization)
    """
    Pm = F.normalize(Pmix, dim=-1)
    Pe = F.normalize(p_ema, dim=-1)
    cos_sim = torch.sum(Pm * Pe.unsqueeze(0), dim=-1)    # [N]
    gamma = torch.sigmoid(alpha * cos_sim)               # [N]
    gamma = torch.clamp(gamma, gmin, gmax)
    P_tilde = (1 - gamma).unsqueeze(-1) * p_ema.unsqueeze(0) + gamma.unsqueeze(-1) * Pmix
    return P_tilde, gamma

# =============== MoP + EMA ===============
def server_prototype_update(student_proto_locals, subgraph_proto_locals,
                            prev_stu_ema, prev_sub_ema,
                            K_stu=4, K_sub=4, beta=0.2, alpha_mop=10.0,
                            alpha_ema=8.0, gmin=0.1, gmax=0.9, device="cpu"):
    # Stack
    P_stu = torch.stack(student_proto_locals, dim=0).to(device)  # [N, H]
    P_sub = torch.stack(subgraph_proto_locals, dim=0).to(device) # [N, D]

    # MoP
    C_stu = kmeans_cosine(P_stu, K_stu)
    C_sub = kmeans_cosine(P_sub, K_sub)
    Pmix_stu, _ = mop_personalized(P_stu, C_stu, alpha=alpha_mop)
    Pmix_sub, _ = mop_personalized(P_sub, C_sub, alpha=alpha_mop)

    # EMA
    stu_mean_new = P_stu.mean(dim=0)
    sub_mean_new = P_sub.mean(dim=0)
    stu_ema = ema_update(prev_stu_ema, stu_mean_new, beta=beta)
    sub_ema = ema_update(prev_sub_ema, sub_mean_new, beta=beta)

    # EMA插值：Teacher vs. Mine
    Ptilde_stu, gamma_stu = ema_personalize_from_mop(Pmix_stu, stu_ema, alpha=alpha_ema, gmin=gmin, gmax=gmax)
    Ptilde_sub, gamma_sub = ema_personalize_from_mop(Pmix_sub, sub_ema, alpha=alpha_ema, gmin=gmin, gmax=gmax)

    # Return the packaged data (in list format, in the same order as the client's).
    proto_glob_student_list = [Ptilde_stu[i].detach().cpu() for i in range(Ptilde_stu.size(0))]
    proto_glob_subgraph_list = [Ptilde_sub[i].detach().cpu() for i in range(Ptilde_sub.size(0))]
    state = {
        "prev_stu_ema": stu_ema.detach(),
        "prev_sub_ema": sub_ema.detach(),
        "gamma_stu": gamma_stu.detach().cpu(),
        "gamma_sub": gamma_sub.detach().cpu(),
    }
    return proto_glob_student_list, proto_glob_subgraph_list, state
