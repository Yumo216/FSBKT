# FedProx 只影响本地训练（loss 加惩罚项），不改变聚合逻辑
# 为了接口一致性，我们仍返回 FedAvg 聚合逻辑
from .fedavg import fedavg

def fedprox(w_locals, weights=None):
    return fedavg(w_locals, weights)
