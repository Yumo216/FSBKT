import copy
import torch
import random


def add_noise_inplace(state_dict, scale=1e-3):
    for k in state_dict:
        state_dict[k].add_(torch.randn_like(state_dict[k]) * scale)

def fednoise(w_locals, weights=None, scale=1e-3):
    for w in w_locals:
        add_noise_inplace(w, scale)
    from .fedavg import fedavg
    return fedavg(w_locals, weights=weights)


