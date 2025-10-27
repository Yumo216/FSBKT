# aggregators/fedavg.py

import copy

def fedavg(w_locals, weights=None):
    if weights is None:
        weights = [1 / len(w_locals)] * len(w_locals)

    w_glob = copy.deepcopy(w_locals[0])
    for k in w_glob.keys():
        for i in range(1, len(w_locals)):
            w_glob[k] += w_locals[i][k] * weights[i]
        w_glob[k] = w_glob[k] / sum(weights)
    return w_glob
