# main_centralized.py
# 纯“集中式”训练与评测（无联邦聚合），右键运行即可
# 会把多客户端数据 concat 成一个 DataLoader 来训练
import os, sys, copy, numpy as np, torch, torch.nn as nn, torch.optim as optim
from typing import List
sys.path.append(os.path.dirname(__file__) or ".")
from datetime import datetime

# ==== CONFIG（只改这里）====
MODEL_NAME   = "DKT"     # "DKT" / "GDKT" / "GAT"
DEVICE       = "auto"     # "auto" / "cpu" / "cuda:0"
SEED         = 42
LR           = 1e-3
EPOCHS       = 50
PATIENCE     = 5          # val AUC 连续多少轮不提升就 early-stop
BATCH_SIZE   = None       # 用数据集默认; 或者手动写一个数
SAVE_DIR     = "./ckpt"   # ckpt 根目录
# ===========================

# ---- 项目内模块 ----
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.dataloader import getDataLoader
from KnowledgeTracing.GCN.MultiGraph import GraphEmbedder
from KnowledgeTracing.GCN.load_adj import build_graph
from KnowledgeTracing.BipartiteGCN.load_graph import load_graph
from KnowledgeTracing.model.DKT import DKT
from KnowledgeTracing.model.DKT_GAT import DKT_GAT
from KnowledgeTracing.model.GDKT import GDKT

# ✅ 使用新的 centralized 评测与损失（兼容 tuple/Tensor 输出）
from KnowledgeTracing.evaluation import eval_central

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(dev):
    if dev == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

# ====== 构建模型（与你 main.py 基本一致，但统一 output_dim=C.QUES）======
def build_model(model_name: str, device: torch.device):
    edge_index, node_features = load_graph(f"../../KTDataset/{C.DATASET}/adj_matrix.csv", C.EMB_DIM, device)
    edge_ind, edge_weight     = build_graph(f"../../KTDataset/{C.DATASET}/{C.DATASET}.json")
    x_init = torch.randn(C.QUES + 1, C.EMB_DIM, device=device)

    if model_name == "DKT":
        return DKT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.QUES, device=device, edge=edge_index, node=node_features, use_gnn_emb=False
        ).to(device)
    elif model_name == "GDKT":
        graph_model = GraphEmbedder(
            model_type="GAT", in_dim=C.EMB_DIM, hidden_dim=C.EMB_DIM, out_dim=C.EMB_DIM,
            device=device, edge_index=edge_ind, edge_weight=edge_weight
        ).to(device)
        return GDKT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.QUES, device=device, graph_encoder=graph_model, x_init=x_init
        ).to(device)
    elif model_name in ["GAT", "GCN", "GATv2"]:
        graph_model = GraphEmbedder(
            model_type=model_name, in_dim=C.EMB_DIM, hidden_dim=C.EMB_DIM, out_dim=C.EMB_DIM,
            device=device, edge_index=edge_ind, edge_weight=edge_weight
        ).to(device)
        return DKT_GAT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.QUES, device=device, graph_encoder=graph_model, x_init=x_init
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# ====== 把联邦的多个 DataLoader 合并为一个 ======
from torch.utils.data import ConcatDataset, DataLoader
def merge_client_loaders(client_loaders: List[DataLoader], batch_size=None, shuffle=True):
    datasets = [ld.dataset for ld in client_loaders]
    merged = ConcatDataset(datasets)
    collate_fn = getattr(client_loaders[0], "collate_fn", None)
    return DataLoader(merged, batch_size=(batch_size or client_loaders[0].batch_size),
                      shuffle=shuffle, drop_last=False, collate_fn=collate_fn)

def _to_cpu_state(state): return {k: v.detach().cpu() for k, v in state.items()}
def _save_ckpt(state_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"; torch.save(state_dict, tmp); os.replace(tmp, path)

def main():
    set_seed(SEED)
    device = pick_device(DEVICE)
    print(f"[Centralized] dataset={C.DATASET} | model={MODEL_NAME} | device={device}")

    # 取到联邦式加载器，然后合并
    train_clients, val_loader, test_loader = getDataLoader(
        C.BATCH_SIZE, C.QUES, C.MAX_STEP, fold_idx=0
    )
    if isinstance(train_clients, list):
        train_loader = merge_client_loaders(train_clients, batch_size=BATCH_SIZE, shuffle=True)
    else:
        train_loader = train_clients

    if isinstance(val_loader, list):
        val_loader = merge_client_loaders(val_loader, batch_size=BATCH_SIZE, shuffle=False)
    if isinstance(test_loader, list):
        test_loader = merge_client_loaders(test_loader, batch_size=BATCH_SIZE, shuffle=False)

    # 模型与优化器
    model = build_model(MODEL_NAME, device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    # ✅ 使用 centralized 版本损失/评测
    loss_fn = eval_central.lossFunc(device).to(device)

    # 训练循环 + 早停
    best_auc = -1.0; best_state = None; best_epoch = 0; bad = 0
    ckpt_dir = os.path.join(SAVE_DIR, C.DATASET, f"{MODEL_NAME}_Centralized")

    for ep in range(1, EPOCHS + 1):
        tr_loss = eval_central.train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        with torch.no_grad():
            auc, acc, rmse = eval_central.test_epoch(model, val_loader, loss_fn, device)
        improved = auc > best_auc
        print(f"[Ep {ep:03d}] train_loss={tr_loss:.4f} | val AUC={auc:.4f} ACC={acc:.4f} RMSE={rmse:.4f} "
              f"{'**BEST**' if improved else ''}")
        if improved:
            best_auc = auc; best_epoch = ep; best_state = copy.deepcopy(model.state_dict()); bad = 0
            # _save_ckpt(_to_cpu_state(best_state), os.path.join(ckpt_dir, f"fold1_best.pth"))
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[EarlyStop] no AUC improve for {PATIENCE} epochs."); break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    # 测试（用最佳权重）
    model.load_state_dict(best_state)
    with torch.no_grad():
        t_auc, t_acc, t_rmse = eval_central.test_epoch(model, test_loader, loss_fn, device)
    print(f"[TEST] AUC={t_auc*100:.2f}% ACC={t_acc*100:.2f}% RMSE={t_rmse:.4f} @best_ep={best_epoch}")

    final_name = f"{C.DATASET}-{MODEL_NAME}_Centralized-AUC{t_auc*100:.2f}.pth"
    final_path = os.path.join(ckpt_dir, final_name)
    _save_ckpt(_to_cpu_state(best_state), final_path)
    print(f"[SAVE] {final_path}")

if __name__ == "__main__":
    main()
