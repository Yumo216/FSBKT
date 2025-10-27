import sys
import os
import copy
import random
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime  # >>> NEW

from KnowledgeTracing.model.DKT import DKT
from KnowledgeTracing.model.DKT_GAT import DKT_GAT
from KnowledgeTracing.model.GDKT import GDKT
from KnowledgeTracing.GCN.MultiGraph import GraphEmbedder
from KnowledgeTracing.GCN.load_adj import build_graph
from KnowledgeTracing.data.dataloader import getDataLoader
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.BipartiteGCN.load_graph import load_graph
from KnowledgeTracing.aggregators import get_aggregator
from KnowledgeTracing.evaluation import eval
from KnowledgeTracing.evaluation import eval_baseline
from KnowledgeTracing.aggregators.proto_enhence import server_prototype_update  # MoP+EMA
from KnowledgeTracing.aggregators.peer_init import peer_init  # PeerInit

# —— 环境与随机种子 ——
torch.autograd.set_detect_anomaly(True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append("../")

# device_id = random.randint(0, 3)
device_id = C.Device_Num
torch.cuda.set_device(device_id)
device = torch.device("cuda")

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# —— 可配置项 ——
model_name = "GAT"         # "DKT", "GDKT", "BiGNN", "GAT", "GCN", "GATv2"
agg_method = "FedAvgProto" # "FedAvg","FedNoise", "FedProx","FedAtt","FedAmp","FedAvgProto","FedProto"
agg_func = get_aggregator(agg_method)

# >>> NEW: 保存目录与工具
CKPT_DIR = f"./ckpt_10.22/{C.DATASET}/{model_name}_{agg_method}"
os.makedirs(CKPT_DIR, exist_ok=True)

def _to_cpu_state(state):
    return {k: v.detach().cpu() for k, v in state.items()}

def _save_ckpt(state_dict, path):
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)  # 原子化替换，防止中途中断写坏文件

# —— 图数据（一次性加载） ——
edge_index, node_features = load_graph(f"../../KTDataset/{C.DATASET}/adj_matrix.csv", C.EMB_DIM, device)
edge_ind, edge_weight = build_graph(f"../../KTDataset/{C.DATASET}/{C.DATASET}.json")
x_init = torch.randn(C.QUES + 1, C.EMB_DIM, device=device)

# —— 模型工厂 ——
def build_model(model_name: str):
    if model_name == "DKT":
        return DKT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT, device=device, edge=edge_index, node=node_features, use_gnn_emb=False
        ).to(device)
    elif model_name == "GDKT":
        graph_backbone = "GAT"
        graph_model = GraphEmbedder(
            model_type=graph_backbone, in_dim=C.EMB_DIM, hidden_dim=C.EMB_DIM, out_dim=C.EMB_DIM,
            device=device, edge_index=edge_ind, edge_weight=edge_weight
        ).to(device)
        return GDKT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT, device=device, graph_encoder=graph_model, x_init=x_init
        ).to(device)
    elif model_name == "BiGNN":
        return DKT(
            emb_dim=C.EMB_DIM, hidden_dim=C.HIDDEN, layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT, device=device, edge=edge_index, node=node_features, use_gnn_emb=True
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

# —— 五折（或一次）交叉验证 ——
results_auc, results_acc, results_rmse = [], [], []

for fold_idx in range(C.K_fold):
    print(f"===== Fold {fold_idx + 1}/{C.K_fold} =====")
    train_loader, val_loader, test_loader = getDataLoader(C.BATCH_SIZE, C.QUES, C.MAX_STEP, fold_idx=fold_idx)

    # 每折重新初始化
    model = build_model(model_name)
    optimizer = optim.Adam(model.parameters(), lr=C.LR, weight_decay=1e-5)
    loss_fn = eval.lossFunc(device).to(device)
    loss_baseline = eval_baseline.lossFunc(device).to(device)
    w_locals = [copy.deepcopy(model.state_dict()) for _ in range(C.CLIENTS)]
    proto_glob = None  # 仅 FedAvgProto 使用

    # early stopping & best tracking
    best_auc = -1.0  # >>> NEW：设为 -1 方便第一轮就能覆盖
    best_acc = 0.0
    best_rmse = 0.0
    best_epoch = 0
    best_state = None  # >>> NEW：保存最佳权重
    patience, counter = 2, 0

    prev_stu_ema, prev_sub_ema = None, None  # MoP+EMA
    proto_student_list = []
    proto_subgraph_list = []
    # >>> NEW：每折的“best.pth”路径（不断覆盖）
    best_ckpt_path = os.path.join(CKPT_DIR, f"fold{fold_idx+1}_best.pth")
    for epoch in range(C.EPOCH):
        print(f"[{model_name} + {agg_method}] GPU:{device_id}  epoch {epoch + 1}")

        if agg_method in ["FedAvgProto", "FedProto"]:
            proto_student = proto_glob["student_proto"] if (proto_glob is not None and "student_proto" in proto_glob) else None
            proto_subgraph = proto_glob["subgraph_proto"] if (proto_glob is not None and "subgraph_proto" in proto_glob) else None
        else:
            proto_student, proto_subgraph = None, None

        # 训练（分流）
        if agg_method in ["FedAvgProto", "FedProto"]:
            model, optimizer, w_locals, init_w, stu_proto_locals, subg_proto_locals = eval.train_epoch(
                model, train_loader, optimizer, loss_fn, w_locals, device,
                proto_student, proto_subgraph, local_epochs=5
            )

            # ===== MoP + EMA：服务器端原型聚合（个性化） =====
            proto_student_list, proto_subgraph_list, ema_state = server_prototype_update(
                stu_proto_locals, subg_proto_locals,
                prev_stu_ema, prev_sub_ema,
                K_stu=4, K_sub=4, beta=0.2, alpha_mop=10.0,
                alpha_ema=8.0, gmin=0.1, gmax=0.9, device=device
            )
            # 更新跨轮EMA状态：
            prev_stu_ema = ema_state["prev_stu_ema"]
            prev_sub_ema = ema_state["prev_sub_ema"]


        else:
            model, optimizer, w_locals, init_w = eval_baseline.train_epoch(
                model, train_loader, optimizer, loss_baseline, w_locals, device
            )
            stu_proto_locals, subg_proto_locals = None, None

        # 聚合
        if agg_method == "FedAtt":
            w_glob = agg_func(w_locals, init_w); proto_glob = None
        elif agg_method in ["FedAvg", "FedProx", "FedNoise"]:
            ratios = eval.calculate_data_ratios(train_loader)
            w_glob = agg_func(w_locals, ratios); proto_glob = None
        elif agg_method == "FedAvgProto":
            ratios = eval.calculate_data_ratios(train_loader)
            proto_locals = {"student_proto": stu_proto_locals, "subgraph_proto": subg_proto_locals}

            # w_glob, proto_glob = agg_func(w_locals, proto_locals, ratios)  # 原型平均,老辈子
            w_glob, _ = agg_func(w_locals, proto_locals, ratios)  # 忽略它的proto输出

            # 用我们的 MoP+EMA 结果覆盖：
            proto_glob = {
                "student_proto": proto_student_list,  # 注意：现在是“列表”，每客户端一份
                "subgraph_proto": proto_subgraph_list
            }

        elif agg_method == "FedProto":
            proto_student_glob = agg_func(proto_locals=stu_proto_locals, device=device)
            proto_glob = {
                "student_proto": proto_student_glob["proto"] if proto_student_glob is not None else None,
                "subgraph_proto": None,
            }
            w_glob = init_w
        else:
            w_glob = agg_func(w_locals); proto_glob = None

        # 原型搬到设备
        if proto_glob is not None:
            for k, v in proto_glob.items():
                if v is not None:
                    # 兼容列表：列表里每个元素再 to(device)
                    if isinstance(v, list):
                        proto_glob[k] = [t.to(device) for t in v]
                    else:
                        proto_glob[k] = v.to(device)

        # 同步全局模型
        model.load_state_dict(w_glob)

        # ======== >>> NEW: PeerInit 个性化下行初始化（参数侧个性化） ========
        # 选择相似度参考原型：推荐用“本轮行为原型”
        ref_protos = stu_proto_locals
        # 若想用 MoP+EMA 的个性化参考来度量相似度，改成：
        # ref_protos = proto_glob["student_proto"]  # 这是一组 per-client 原型

        w_locals = peer_init(
            w_locals=w_locals,  # 本轮各客户端训练后的模型
            ref_protos=ref_protos,  # 相似度参考原型
            topM=8,  # 可调：4/8/12
            tau=10.0,  # 可调：8/10/14
            use_cpu=False  # 显存紧张设 True
        )
        # 说明：我们把“下一轮的 m_locals”直接设为 PeerInit 的混合结果；
        # 你的 train_epoch 里有 `Apply(Init_w, m_locals[idx])`，无需改接口。
        # ======== <<< NEW 结束 ========

        # 验证
        with torch.no_grad():
            if agg_method in ["FedAvgProto", "FedProto"]:
                auc, acc, rmse = eval.test_epoch(model, val_loader, loss_fn, device)
            else:
                auc, acc, rmse = eval_baseline.test_epoch(model, val_loader, loss_baseline, device)

            improved = auc > best_auc
            if improved:
                best_auc, best_acc, best_rmse = auc, acc, rmse
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())  # >>> NEW：保存最佳
                _save_ckpt(_to_cpu_state(best_state), best_ckpt_path)  # >>> NEW：覆盖写 best.pth
                print(f"[SAVE] New best at epoch {best_epoch} | AUC={best_auc:.6f} -> {best_ckpt_path}")

            counter = 0 if improved else counter + 1

        print(f"Best AUC: {best_auc:.6f}  ACC: {best_acc:.6f}  RMSE: {best_rmse:.6f}  @epoch {best_epoch}")
        print("-" * 60)
        if counter >= patience:
            print(f"Early stopping: AUC no improve for {patience} epochs.")
            break

    # >>> NEW：若没有任何提升（极端情况），也保存当前权重
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        _save_ckpt(_to_cpu_state(best_state), best_ckpt_path)
        print(f"[SAVE] No improvement case: saved current state to {best_ckpt_path}")

    # 测试：使用最佳验证 AUC 的权重（更科学）
    print("Testing (with best val-AUC weights)...")
    model.load_state_dict(best_state)  # >>> NEW
    with torch.no_grad():
        if agg_method in ["FedAvgProto", "FedProto"]:
            t_auc, t_acc, t_rmse = eval.test_epoch(model, test_loader, loss_fn, device)
        else:
            t_auc, t_acc, t_rmse = eval_baseline.test_epoch(model, test_loader, loss_baseline, device)
    print(f"[Fold {fold_idx + 1}] Test AUC: {t_auc * 100:.2f}%  ACC: {t_acc * 100:.2f}%  RMSE: {t_rmse:.4f}")

    # >>> NEW：按信息重命名保存一份最终 ckpt（便于后来指定）
    final_ckpt_name = f"{C.DATASET}-{model_name}_{agg_method}-AUC{best_auc*100:.2f}.pth"
    final_ckpt_path = os.path.join(CKPT_DIR, final_ckpt_name)
    _save_ckpt(_to_cpu_state(best_state), final_ckpt_path)
    print(f"[SAVE] Final best checkpoint saved to: {final_ckpt_path}")

    results_auc.append(t_auc)
    results_acc.append(t_acc)
    # 注意：某些实现里 rmse 是 tensor，需要取 .item()
    results_rmse.append(float(t_rmse if isinstance(t_rmse, float) else t_rmse.item()))

# —— 汇总 ——
print("===== Cross-Validation Results =====")
print(f"Model: {model_name}, Aggregator: {agg_method}, Dataset: {C.DATASET}")
print(f"Avg Test AUC : {np.mean(results_auc) * 100:.2f}% ± {np.std(results_auc) * 100:.2f}%")
print(f"Avg Test ACC : {np.mean(results_acc) * 100:.2f}% ± {np.std(results_acc) * 100:.2f}%")
print(f"Avg Test RMSE: {np.mean(results_rmse):.4f} ± {np.std(results_rmse):.4f}")
print(f"AUC list : {[round(x, 4) for x in results_auc]}")
print(f"ACC list : {[round(x, 4) for x in results_acc]}")
print(f"RMSE list: {[round(x, 4) for x in results_rmse]}")
print(f"[CKPT DIR] {CKPT_DIR}")  # >>> NEW：提示保存路径
