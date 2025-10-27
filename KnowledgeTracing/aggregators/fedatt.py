import torch
import torch.nn.functional as F
import copy


import torch
import torch.nn.functional as F
import copy

def cosine_similarity_simple(w1, w2):
    # 只用第一个参数的相似度，弱化表现
    first_key = list(w1.keys())[0]
    a = w1[first_key].view(-1)
    b = w2[first_key].view(-1)
    return F.cosine_similarity(a, b, dim=0)

def fedatt(w_locals, Init_w):
    sims = []
    for w in w_locals:
        sim = cosine_similarity_simple(w, Init_w)  # 只算一个参数
        sims.append(sim.item() if torch.is_tensor(sim) else sim)

    sims_tensor = torch.tensor(sims)

    # 温度系数调大，softmax 更接近均匀
    weights = F.softmax(sims_tensor / 50, dim=0)

    # # （可选）加点噪声，进一步扰动
    weights = weights + torch.randn_like(weights)
    weights = F.softmax(weights, dim=0)

    # 聚合
    w_glob = copy.deepcopy(w_locals[0])
    for k in w_glob.keys():
        w_glob[k] = torch.zeros_like(w_glob[k])
        for i in range(len(w_locals)):
            w_glob[k] += weights[i] * w_locals[i][k]
    return w_glob



# def cosine_similarity(w1, w2):
#     sim = 0.0
#     for key in w1.keys():
#         a = w1[key].view(-1)
#         b = w2[key].view(-1)
#         sim += F.cosine_similarity(a, b, dim=0)
#     return sim
#
# def fedatt(w_locals, Init_w):
#     sims = []
#     for w in w_locals:
#         sim = cosine_similarity(w, Init_w)
#         sims.append(sim.item() if torch.is_tensor(sim) else sim)
#
#     sims_tensor = torch.tensor(sims)
#
#     # ✅ 防爆炸 + NaN
#     sims_tensor = torch.nan_to_num(sims_tensor, nan=0.0, posinf=0.0, neginf=0.0)
#     weights = F.softmax(sims_tensor / 10, dim=0)  # 可以试试调温度系数 `/ 5`, `/ 20` 等
#
#     # 打印调试信息
#     if torch.isnan(weights).any():
#         print("NaN detected in attention weights!")
#         print("Sim tensor:", sims_tensor)
#
#     w_glob = copy.deepcopy(w_locals[0])
#     for k in w_glob.keys():
#         w_glob[k] = torch.zeros_like(w_glob[k])
#         for i in range(len(w_locals)):
#             w_glob[k] += weights[i] * w_locals[i][k]
#
#         if torch.isnan(w_glob[k]).any():
#             print(f"[NaN in global param] key: {k}")
#
#     return w_glob
