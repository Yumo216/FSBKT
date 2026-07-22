# FSBKT: Federated Structural and Behavioral Prototype Learning for Privacy-Preserving Knowledge Tracing

This repository contains the implementation of FSBKT, a federated knowledge tracing framework that addresses cross-school heterogeneity and privacy constraints via structural and behavioral prototype learning.

## Requirements

```
Python >= 3.8
PyTorch >= 1.12
scikit-learn
numpy
```

## Usage

Configure dataset and model in `KnowledgeTracing/Constant.py`, then run:

```bash
python main.py
```

For privacy evaluation (MIA / ReID):

```bash
python mia+reid.py
```


