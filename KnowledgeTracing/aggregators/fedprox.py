# FedProx only affects local training by adding a proximal regularization term to the loss.
# We still return the FedAvg-style aggregation result for interface consistency.
from .fedavg import fedavg

def fedprox(w_locals, weights=None):
    return fedavg(w_locals, weights)
