#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Print p-values (9 lines): dataset × {AUC, ACC, RMSE}
# Star rule: ** if p <= 0.005 AND FSBKT better; * if 0.005 < p <= 0.01 AND FSBKT better; else no star.
# NOTE: This DEMO simulates 5-fold values from mean±std. Replace simulate() outputs with real per-fold arrays for formal tests.

import numpy as np
from scipy.stats import ttest_rel

RNG = np.random.default_rng(42)
N = 5  # folds
ALPHA1 = 0.01
ALPHA2 = 0.005

tbl = {
    "ASSIST": {
        "AUC": {"FedAvg":(76.20,0.34),"Fednoise":(76.04,0.47),"FedProx":(76.06,0.31),"FedAtt":(73.35,0.63),"FedAmp":(72.50,0.33),"Fedproto":(78.90,0.33),"FSBKT":(83.73,0.31)},
        "ACC": {"FedAvg":(69.57,0.20),"Fednoise":(69.52,0.39),"FedProx":(69.53,0.23),"FedAtt":(66.78,0.56),"FedAmp":(65.89,0.49),"Fedproto":(72.03,0.17),"FSBKT":(75.79,0.25)},
        "RMSE":{"FedAvg":(0.4699,0.0042),"Fednoise":(0.4733,0.0043),"FedProx":(0.4747,0.0028),"FedAtt":(0.4614,0.0022),"FedAmp":(0.4643,0.0011),"Fedproto":(0.4628,0.0023),"FSBKT":(0.4436,0.0022)},
    },
    "Eedi": {
        "AUC": {"FedAvg":(64.26,1.64),"Fednoise":(66.54,0.35),"FedProx":(66.54,0.34),"FedAtt":(69.68,0.27),"FedAmp":(70.66,0.29),"Fedproto":(66.68,0.47),"FSBKT":(73.77,0.42)},
        "ACC": {"FedAvg":(61.50,1.41),"Fednoise":(62.89,0.44),"FedProx":(62.91,0.45),"FedAtt":(64.83,0.42),"FedAmp":(65.80,0.43),"Fedproto":(63.03,0.51),"FSBKT":(67.71,0.43)},
        "RMSE":{"FedAvg":(0.5081,0.0056),"Fednoise":(0.5144,0.0031),"FedProx":(0.5144,0.0031),"FedAtt":(0.4683,0.0026),"FedAmp":(0.4638,0.0024),"Fedproto":(0.5158,0.0040),"FSBKT":(0.5193,0.0026)},
    },
    "SLP-all": {
        "AUC": {"FedAvg":(78.81,0.61),"Fednoise":(79.10,0.54),"FedProx":(78.99,0.50),"FedAtt":(83.34,0.15),"FedAmp":(83.55,0.14),"Fedproto":(82.42,0.15),"FSBKT":(84.72,0.10)},
        "ACC": {"FedAvg":(73.40,0.48),"Fednoise":(73.71,0.53),"FedProx":(73.58,0.45),"FedAtt":(76.37,0.11),"FedAmp":(76.68,0.11),"Fedproto":(76.35,0.12),"FSBKT":(77.72,0.16)},
        "RMSE":{"FedAvg":(0.4433,0.0032),"Fednoise":(0.4433,0.0056),"FedProx":(0.4436,0.0037),"FedAtt":(0.4335,0.0012),"FedAmp":(0.4345,0.0014),"Fedproto":(0.4417,0.0021),"FSBKT":(0.4282,0.0021)},
    },
}

def simulate(mean, std, metric, n=N):
    if metric in ("AUC","ACC"):
        m, s = mean/100.0, std/100.0
        x = RNG.normal(m, s, n)
        return np.clip(x, 0.0, 1.0)
    return RNG.normal(mean, std, n)

def best_baseline_name(d, metric):
    d = {k:v for k,v in d.items() if k!="FSBKT"}
    if metric in ("AUC","ACC"):   # higher better
        return max(d.items(), key=lambda kv: kv[1][0])[0]
    else:                         # RMSE lower better
        return min(d.items(), key=lambda kv: kv[1][0])[0]

print("p-values vs best baseline (paired two-tailed t-test, n=5)\n"
      "Legend: ** if p<=0.005 and FSBKT better; * if 0.005<p<=0.01 and FSBKT better; else no mark.\n")

for dataset, metrics in tbl.items():
    for metric, methods in metrics.items():
        best = best_baseline_name(methods, metric)
        fs_mu, fs_sd = methods["FSBKT"]
        bs_mu, bs_sd = methods[best]
        fs = simulate(fs_mu, fs_sd, metric)
        bs = simulate(bs_mu, bs_sd, metric)
        t, p = ttest_rel(fs, bs)  # two-tailed paired t-test
        # Better direction
        fs_mean, bs_mean = fs.mean(), bs.mean()
        better = (fs_mean > bs_mean) if metric in ("AUC","ACC") else (fs_mean < bs_mean)
        # Star label
        label = ""
        if better and p <= ALPHA2:
            label = "**"
        elif better and p <= ALPHA1:
            label = "*"
        # Print ONLY the p-value (4 decimals) + minimal context
        print(f"{dataset} {metric}: p={p:.6f} (vs {best}) {label}")
