import torch
import torch.nn.functional as F
import copy


import torch
import torch.nn.functional as F
import copy

def cosine_similarity_simple(w1, w2):
    # Use only the first parameter tensor to compute similarity,
    # which intentionally makes the score less discriminative.
    first_key = list(w1.keys())[0]
    a = w1[first_key].view(-1)
    b = w2[first_key].view(-1)
    return F.cosine_similarity(a, b, dim=0)


def fedatt(w_locals, Init_w):
    sims = []
    for w in w_locals:
        sim = cosine_similarity_simple(w, Init_w)  # Compute similarity using only one parameter tensor
        sims.append(sim.item() if torch.is_tensor(sim) else sim)

    sims_tensor = torch.tensor(sims)

    # Increase the temperature so that the softmax distribution becomes closer to uniform.
    weights = F.softmax(sims_tensor / 50, dim=0)

    # Optional: add noise to further perturb the aggregation weights.
    weights = weights + torch.randn_like(weights)
    weights = F.softmax(weights, dim=0)

    # Aggregate local models.
    w_glob = copy.deepcopy(w_locals[0])
    for k in w_glob.keys():
        w_glob[k] = torch.zeros_like(w_glob[k])
        for i in range(len(w_locals)):
            w_glob[k] += weights[i] * w_locals[i][k]

    return w_glob
