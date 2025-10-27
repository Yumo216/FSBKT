# mia+reid.py
# 右键运行即可；在“==== CONFIG ====”里改参数。
# MODE:
#   'mia'  -> 训练一个攻击器做成员推断（隐私泄露直测）
#   'reid' -> 分半去匿名化（验证无法指认到具体学生）

import os, sys, numpy as np
from typing import Tuple, Dict, Any, Iterable
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
import KnowledgeTracing.Constant as C
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# ========= CONFIG（只改这里）=========
MODE        = 'mia'      # 'mia' 或 'reid'
MODEL_NAME  = 'DKT'       # 'DKT' / 'GDKT' / 'GAT' / 'GCN' / 'GATv2'
CKPT_PATH   = './ckpt/SLP-all/DKT_Centralized/SLP-all-DKT_Centralized-AUC86.21.pth'          # 模型权重路径；留空则用当前随机权重
FOLD        = 0           # 使用哪一折
DEVICE      = 'auto'      # 'auto' / 'cpu' / 'cuda:0' / 'cuda:1' ...
SEED        = 42
MIN_LEN_REID = 8          # ReID 时有效序列最短长度门槛
EPOCHS_ATK   = 30         # MIA 攻击器训练轮数
# ====================================

# ==== import your project modules (与你的 main.py 一致) ====
sys.path.append(os.path.dirname(__file__) or ".")
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.data.dataloader import getDataLoader
from KnowledgeTracing.GCN.MultiGraph import GraphEmbedder
from KnowledgeTracing.GCN.load_adj import build_graph
from KnowledgeTracing.BipartiteGCN.load_graph import load_graph
from KnowledgeTracing.model.DKT_GAT import DKT_GAT
from KnowledgeTracing.model.DKT import DKT
from KnowledgeTracing.model.GDKT import GDKT

# ----------------------
# Utilities
# ----------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device(dev_cfg: str):
    if dev_cfg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(dev_cfg)

def build_model(model_name: str, device: torch.device):
    """与你的 main.py 保持一致"""
    edge_index, node_features = load_graph(f"../../KTDataset/{C.DATASET}/adj_matrix.csv", C.EMB_DIM, device)
    edge_ind, edge_weight     = build_graph(f"../../KTDataset/{C.DATASET}/{C.DATASET}.json")
    x_init = torch.randn(C.QUES + 1, C.EMB_DIM, device=device)

    if model_name == "DKT":
        return DKT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT, device=device, edge=edge_index, node=node_features, use_gnn_emb=False
        ).to(device)
    elif model_name == "GDKT":
        graph_model = GraphEmbedder(
            model_type="GAT", in_dim=C.EMB_DIM, hidden_dim=C.EMB_DIM, out_dim=C.EMB_DIM,
            device=device, edge_index=edge_ind, edge_weight=edge_weight
        ).to(device)
        return GDKT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT, device=device, graph_encoder=graph_model, x_init=x_init
        ).to(device)
    elif model_name in ["GAT", "GCN", "GATv2"]:
        graph_model = GraphEmbedder(
            model_type=model_name, in_dim=C.EMB_DIM, hidden_dim=C.EMB_DIM, out_dim=C.EMB_DIM,
            device=device, edge_index=edge_ind, edge_weight=edge_weight
        ).to(device)
        return DKT_GAT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT, device=device, graph_encoder=graph_model, x_init=x_init
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def load_ckpt_if_any(model: nn.Module, ckpt_path: str):
    if ckpt_path and os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd)
        print(f"[LOAD] Loaded checkpoint: {ckpt_path}")
    else:
        if ckpt_path:
            print(f"[LOAD] WARNING: ckpt not found: {ckpt_path}. Use current weights.")
        else:
            print(f"[LOAD] No ckpt provided. Using current (likely random) weights.")
    model.eval()

# ----------------------
# Batch parsing（你的 DataLoader: Tensor [B, L, 3]）
# ----------------------
def parse_batch(batch: torch.Tensor, device: torch.device):
    """
    与 eval/train 中一致：
      datas = torch.chunk(batch, 3, dim=2)
      datas[0] -> q_input (题目ID, int, 1-based)
      datas[1] -> x       (交互编码, int)
      datas[2] -> resp    (响应, float; >=1 表示有效，label=resp-1)
    返回: x [B,L] long, q [B,L] long, resp [B,L] float
    """
    if not torch.is_tensor(batch):
        raise RuntimeError("Expected Tensor batch [B,L,3].")
    if batch.dim() != 3 or batch.size(2) < 3:
        raise RuntimeError(f"Unexpected tensor batch shape: {tuple(batch.shape)} (expect [B,L,3])")
    batch = batch.to(device)
    datas = torch.chunk(batch, 3, dim=2)
    q     = datas[0].squeeze(-1).long()
    x     = datas[1].squeeze(-1).long()
    resp  = datas[2].squeeze(-1).float()
    return x, q, resp

def iter_batches(loader_or_list: Any) -> Iterable[torch.Tensor]:
    """兼容：train_loader 可能是 [client_loader1, client_loader2, ...] 的列表"""
    if isinstance(loader_or_list, (list, tuple)):
        for ld in loader_or_list:
            for b in ld:
                yield b
    else:
        for b in loader_or_list:
            yield b

# ----------------------
# 前向 -> gather -> mask（与你的 lossFunc 对齐）
# ----------------------
def forward_to_probs_and_labels(model, x, q, resp):
    """
    输入:
      x    : [B, L] long
      q    : [B, L] long (1-based; 用于下一时刻题目选择)
      resp : [B, L] float (>=1 有效；label=resp-1)
    输出:
      p    : [B, L-1] float  每步概率（按题目 gather 后）
      y    : [B, L-1] float  0/1 标签
      mask : [B, L-1] bool   有效位置
    """
    out = model(x, q)
    logit = out[0] if isinstance(out, (tuple, list)) else out    # [B, L, Q]
    q_target = torch.clamp(q[:, 1:] - 1, min=0, max=C.QUES - 1)  # [B, L-1]
    logits_sel = logit[:, :-1, :]                                # [B, L-1, Q]
    logits_g = torch.gather(logits_sel, dim=2, index=q_target.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    p = torch.sigmoid(logits_g)                                   # [B, L-1]

    p = torch.nan_to_num(p, nan=0.5)  # NaN -> 0.5
    p = p.clamp_(1e-6, 1 - 1e-6)  # 防止 0/1

    y = resp[:, 1:] - 1                                           # [B, L-1]
    mask = resp[:, 1:].ge(1)                                      # [B, L-1]
    return p, y, mask

# ----------------------
# 统计特征（序列级 8 维）
# ----------------------
def seq_feature_stats(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p = np.clip(p, eps, 1. - eps)
    T = len(p)
    mean_p = float(p.mean())
    std_p  = float(p.std())
    max_p  = float(p.max())
    min_p  = float(p.min())
    entropy = float(-np.mean(p*np.log(p) + (1 - p)*np.log(1 - p)))
    loss = float(-np.mean(y*np.log(p) + (1 - y)*np.log(1 - p)))
    top3 = float(np.mean(np.sort(p)[-3:])) if T >= 3 else mean_p
    return np.array([mean_p, std_p, max_p, min_p, entropy, loss, top3, float(T)], dtype=np.float32)

# ----------------------
# MIA：收集攻击特征
# ----------------------
def collect_attack_features(model, loader_or_list, device, is_member_label: int):
    model.eval()
    feats, mem = [], []
    with torch.no_grad():
        for batch in iter_batches(loader_or_list):
            x, q, resp = parse_batch(batch, device)
            p, y, mask = forward_to_probs_and_labels(model, x, q, resp)
            p_np = p.cpu().numpy(); y_np = y.cpu().numpy(); m_np = mask.cpu().numpy().astype(bool)
            B, T = p_np.shape
            for i in range(B):
                vi = m_np[i]
                if not vi.any():
                    continue
                fi = seq_feature_stats(p_np[i, vi], y_np[i, vi])
                feats.append(fi); mem.append(is_member_label)
    if len(feats) == 0:
        return np.zeros((0,8), np.float32), np.zeros((0,), np.int64)
    return np.vstack(feats).astype(np.float32), np.array(mem, dtype=np.int64)

class AttackMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_attack_model(X, y, device, epochs=30, batch=128, lr=1e-3):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    atk = AttackMLP(X.shape[1]).to(device)
    opt = optim.Adam(atk.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    idx = np.arange(len(X_tr))
    for ep in range(epochs):
        np.random.shuffle(idx)
        atk.train()
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]
            xb = torch.tensor(X_tr[b], dtype=torch.float32, device=device)
            yb = torch.tensor(y_tr[b], dtype=torch.float32, device=device).unsqueeze(1)
            pred = atk(xb); loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        atk.eval()
        with torch.no_grad():
            pv = atk(torch.tensor(X_val, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        if ep % 5 == 0 or ep == epochs-1:
            print(f"[ATK] Epoch {ep:02d} | val_auc {roc_auc_score(y_val, pv):.4f}")
    return atk

def run_mia(model, device, fold):
    train_loader, val_loader, test_loader = getDataLoader(C.BATCH_SIZE, C.QUES, C.MAX_STEP, fold_idx=fold)
    print("[MIA] Collecting member/non-member features ...")
    # 训练集=members，测试集=non-members（快速近似）
    X_mem, y_mem = collect_attack_features(model, train_loader, device, is_member_label=1)
    X_non, y_non = collect_attack_features(model, test_loader, device, is_member_label=0)
    if X_mem.shape[0] == 0 or X_non.shape[0] == 0:
        print("[MIA] ERROR: empty features (check masks/batch).");
        return

    # 构造攻击训练/验证集（shadow-lite）
    X_all = np.vstack([X_mem, X_non]); y_all = np.hstack([y_mem, y_non])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.3, random_state=123, stratify=y_all
    )
    atk = train_attack_model(X_tr, y_tr, device, epochs=EPOCHS_ATK)

    # —— 最终评估（阈值无关 + 阈值相关，全面且抗不平衡）——
    atk.eval()
    with torch.no_grad():
        pv = atk(torch.tensor(X_all, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)

    # 统计分布（解释ACC偏高的原因）
    n_pos = int(y_all.sum()); n_neg = int((1 - y_all).sum())
    print(f"[MIA] class counts -> members={n_pos} | non-members={n_neg}")

    # 1) 阈值无关指标（主结论）
    auc = roc_auc_score(y_all, pv)                  # 越接近 0.5 越好
    aupr = average_precision_score(y_all, pv)       # 与正类比例比较
    fpr, tpr, thr = roc_curve(y_all, pv)
    max_adv = float(np.max(tpr - fpr))              # 标准攻击优势 Max(TPR−FPR) 越接近 0 越好
    best_ix = int(np.argmax(tpr - fpr))
    best_thr = float(thr[best_ix]) if best_ix < len(thr) else 0.5

    # 2) 在最佳J点（TPR−FPR最大）上的阈值相关指标
    pred_best = (pv >= best_thr).astype(int)
    bacc_best = balanced_accuracy_score(y_all, pred_best)   # 越接近 0.5 越好
    acc_best  = float((pred_best == y_all).mean())          # 仅参考
    tpr_best  = float(tpr[best_ix]); fpr_best = float(fpr[best_ix])

    # 3) EER（FPR=FNR处的错误率，越接近 50% 越好）
    fnr = 1.0 - tpr
    ix_eer = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[ix_eer] + fnr[ix_eer]) / 2.0)

    # 4) 旧的ACC@0.5（仅给参考，不作为结论）
    acc05 = accuracy_score(y_all, (pv >= 0.5).astype(int))

    # —— 打印（对齐论文口径）——
    print(
        "[MIA] "
        f"AUC={auc:.4f}  | MaxAdv={max_adv:.4f}  | AUPR={aupr:.4f}  | "
        f"EER={eer:.3f}  | BestThr={best_thr:.3f}  | "
        f"BACC@Best={bacc_best:.3f}  | ACC@Best={acc_best:.3f}  | "
        f"TPR@Best={tpr_best:.3f}  | FPR@Best={fpr_best:.3f}  | "
        f"(ACC@0.5={acc05:.3f} for ref)"
    )


# ----------------------
# ReID：分半去匿名化
# ----------------------
def collect_split_half_signatures(model, loader_or_list, device, min_len=8):
    model.eval()
    q_feats, g_feats = [], []
    with torch.no_grad():
        for batch in iter_batches(loader_or_list):
            x, q, resp = parse_batch(batch, device)
            p, y, mask = forward_to_probs_and_labels(model, x, q, resp)
            p_np = p.cpu().numpy(); y_np = y.cpu().numpy(); m_np = mask.cpu().numpy().astype(bool)
            B, T = p_np.shape
            for i in range(B):
                vi = m_np[i]
                if not vi.any(): continue
                pi = p_np[i, vi]; yi = y_np[i, vi]
                L = len(pi)
                if L < min_len: continue
                mid = L // 2
                q_feats.append(seq_feature_stats(pi[:mid], yi[:mid]))
                g_feats.append(seq_feature_stats(pi[mid:], yi[mid:]))
    if len(q_feats)==0:
        return np.zeros((0,8), np.float32), np.zeros((0,8), np.float32)
    return np.vstack(q_feats), np.vstack(g_feats)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def eval_reid(q, g):
    S = cosine_sim(q, g)  # [N,N]
    N = S.shape[0]
    order = np.argsort(-S, axis=1)
    top1 = np.mean(order[:,0] == np.arange(N))
    top5 = np.mean([i in order[i,:min(5,N)] for i in range(N)])
    # MRR
    ranks = []
    for i in range(N):
        pos = int(np.where(order[i]==i)[0][0]) + 1
        ranks.append(1.0/pos)
    mrr = float(np.mean(ranks))
    tau = 0.95
    anon_sizes = (S >= tau).sum(axis=1)
    anon_med = float(np.median(anon_sizes))
    anon_p90 = float(np.percentile(anon_sizes, 90))
    return {"N":N, "top1":top1, "top5":top5, "mrr":mrr,
            "anon_med@0.95":anon_med, "anon_p90@0.95":anon_p90}

def run_reid(model, device, fold):
    _, _, test_loader = getDataLoader(C.BATCH_SIZE, C.QUES, C.MAX_STEP, fold_idx=fold)
    print("[ReID] Collecting split-half signatures on test set ...")
    Q, G = collect_split_half_signatures(model, test_loader, device, min_len=MIN_LEN_REID)
    if len(Q)==0:
        print("[ReID] No valid sequences (too short or all masked)."); return
    m = eval_reid(Q, G)
    print(f"[ReID] N={m['N']} | Top1={m['top1']:.4f} | Top5={m['top5']:.4f} | MRR={m['mrr']:.4f} "
          f"| AnonMed@0.95={m['anon_med@0.95']:.1f} | AnonP90@0.95={m['anon_p90@0.95']:.1f}")
    # 解读：Top1/Top5 越低越好；AnonMed/P90 越大越好（匿名集合更大）

# ----------------------
# Main (无命令行参数)
# ----------------------
def main():
    set_seed(SEED)
    device = pick_device(DEVICE)
    print(f"[SETUP] device = {device} | mode = {MODE} | model = {MODEL_NAME} | fold = {FOLD}")

    model = build_model(MODEL_NAME, device)
    load_ckpt_if_any(model, CKPT_PATH)
    model.eval()

    if MODE == 'mia':
        run_mia(model, device, fold=FOLD)
    elif MODE == 'reid':
        run_reid(model, device, fold=FOLD)
    else:
        raise ValueError("MODE must be 'mia' or 'reid'.")

if __name__ == "__main__":
    main()
