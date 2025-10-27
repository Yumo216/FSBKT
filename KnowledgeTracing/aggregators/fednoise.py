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


'''占用较高'''
# def add_noise_to_weights(state_dict, scale=1e-3):
#     noisy_state = copy.deepcopy(state_dict)
#     for k in noisy_state:
#         noise = torch.randn_like(noisy_state[k]) * scale
#         noisy_state[k] += noise
#     return noisy_state
#
# def fednoise(w_locals, weights=None, scale=1e-3):
#     noisy_w_locals = [add_noise_to_weights(w, scale) for w in w_locals]
#     from .fedavg import fedavg
#     return fedavg(noisy_w_locals, weights)
