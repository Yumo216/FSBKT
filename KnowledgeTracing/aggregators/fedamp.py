import torch
import torch.nn.functional as F
import copy

import torch
import torch.nn.functional as F
import copy
import random

def cosine_similarity_simple(w1, w2, eps=1e-8):
    first_key = list(w1.keys())[0]
    a = w1[first_key].view(-1).detach()
    b = w2[first_key].view(-1).detach()
    if a.norm() < eps or b.norm() < eps:
        return torch.tensor(0.0)
    return F.cosine_similarity(a, b, dim=0)

def fedamp(w_locals, noise_std=1e-3, temp=50.0):
    personalized = []

    for i, wi in enumerate(w_locals):
        sim_scores = []
        for j, wj in enumerate(w_locals):
            s = cosine_similarity_simple(wi, wj)
            if torch.isnan(s) or torch.isinf(s):
                s = torch.tensor(0.0)
            sim_scores.append(s)

        sim_scores_tensor = torch.stack(sim_scores)
        sim_scores_tensor = torch.nan_to_num(sim_scores_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        weights = F.softmax(sim_scores_tensor / temp, dim=0)
        weights = weights + torch.rand_like(weights)
        weights = F.softmax(weights, dim=0)

        new_w = copy.deepcopy(wi)
        for k in new_w.keys():
            weighted_sum = sum(weights[j] * w_locals[j][k] for j in range(len(w_locals)))

            new_w[k] = torch.nan_to_num(weighted_sum, nan=0.0).detach()
            if torch.is_floating_point(new_w[k]):
                new_w[k] += noise_std * torch.randn_like(new_w[k])
        personalized.append(new_w)


    w_glob = copy.deepcopy(personalized[0])
    for k in w_glob.keys():
        stack = torch.stack([w[k] for w in personalized])
        w_glob[k] = torch.nan_to_num(stack.mean(dim=0), nan=0.0)

    return w_glob
