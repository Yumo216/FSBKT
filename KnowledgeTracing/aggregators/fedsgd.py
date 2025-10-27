import copy

def fedsgd(grads, weights=None):
    if weights is None:
        weights = [1 / len(grads)] * len(grads)

    avg_grad = copy.deepcopy(grads[0])
    for k in avg_grad.keys():
        avg_grad[k] = sum(g[k] * w for g, w in zip(grads, weights))

    return avg_grad
