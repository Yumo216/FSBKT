from .fedavg import fedavg
from .fedprox import fedprox
from .fedatt import fedatt
from .fedamp import fedamp
from .fedsgd import fedsgd
from .fednoise import fednoise
from .fedavg_proto import fedavg_proto
from .fedproto import fedproto


def get_aggregator(name):
    if name == "FedAvg":
        return fedavg
    elif name == "FedProx":
        return fedprox
    elif name == "FedAtt":
        return fedatt
    elif name == "FedNoise":
        return fednoise
    elif name == "FedSGD":
        return fedsgd
    elif name == "FedAmp":
        return fedamp
    elif name == "FedAvgProto":
        return fedavg_proto
    elif name == "FedProto":
        return fedproto

    else:
        raise ValueError(f"Unknown aggregator: {name}")
