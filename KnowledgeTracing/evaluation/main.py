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

# ---- Environment setup and random seeds ----
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

# ---- Configurable options ----
model_name = "GAT"  # "DKT", "GDKT", "BiGNN", "GAT", "GCN", "GATv2"
agg_method = "FedAvgProto"  # "FedAvg", "FedNoise", "FedProx", "FedAtt", "FedAmp", "FedAvgProto", "FedProto"
agg_func = get_aggregator(agg_method)

# >>> NEW: checkpoint directory and helper functions
CKPT_DIR = f"./ckpt_10.22/{C.DATASET}/{model_name}_{agg_method}"
os.makedirs(CKPT_DIR, exist_ok=True)


def _to_cpu_state(state):
    return {k: v.detach().cpu() for k, v in state.items()}


def _save_ckpt(state_dict, path):
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)  # Atomic replacement to avoid corrupted checkpoints if interrupted


# ---- Graph data (loaded once) ----
edge_index, node_features = load_graph(
    f"../../KTDataset/{C.DATASET}/adj_matrix.csv",
    C.EMB_DIM,
    device,
)
edge_ind, edge_weight = build_graph(f"../../KTDataset/{C.DATASET}/{C.DATASET}.json")
x_init = torch.randn(C.QUES + 1, C.EMB_DIM, device=device)


# ---- Model factory ----
def build_model(model_name: str):
    if model_name == "DKT":
        return DKT(
            emb_dim=C.EMB_DIM,
            hidden_dim=C.HIDDEN,
            layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT,
            device=device,
            edge=edge_index,
            node=node_features,
            use_gnn_emb=False,
        ).to(device)
    elif model_name == "GDKT":
        graph_backbone = "GAT"
        graph_model = GraphEmbedder(
            model_type=graph_backbone,
            in_dim=C.EMB_DIM,
            hidden_dim=C.EMB_DIM,
            out_dim=C.EMB_DIM,
            device=device,
            edge_index=edge_ind,
            edge_weight=edge_weight,
        ).to(device)
        return GDKT(
            emb_dim=C.EMB_DIM,
            hidden_dim=C.HIDDEN,
            layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT,
            device=device,
            graph_encoder=graph_model,
            x_init=x_init,
        ).to(device)
    elif model_name == "BiGNN":
        return DKT(
            emb_dim=C.EMB_DIM,
            hidden_dim=C.HIDDEN,
            layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT,
            device=device,
            edge=edge_index,
            node=node_features,
            use_gnn_emb=True,
        ).to(device)
    elif model_name in ["GAT", "GCN", "GATv2"]:
        graph_model = GraphEmbedder(
            model_type=model_name,
            in_dim=C.EMB_DIM,
            hidden_dim=C.EMB_DIM,
            out_dim=C.EMB_DIM,
            device=device,
            edge_index=edge_ind,
            edge_weight=edge_weight,
        ).to(device)
        return DKT_GAT(
            emb_dim=C.EMB_DIM,
            hidden_dim=C.HIDDEN,
            layer_dim=C.RNN_LAYERS,
            output_dim=C.OUTPUT,
            device=device,
            graph_encoder=graph_model,
            x_init=x_init,
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ---- Five-fold (or single-run) cross-validation ----
results_auc, results_acc, results_rmse = [], [], []

for fold_idx in range(C.K_fold):
    print(f"===== Fold {fold_idx + 1}/{C.K_fold} =====")
    train_loader, val_loader, test_loader = getDataLoader(
        C.BATCH_SIZE,
        C.QUES,
        C.MAX_STEP,
        fold_idx=fold_idx,
    )

    # Reinitialize for each fold
    model = build_model(model_name)
    optimizer = optim.Adam(model.parameters(), lr=C.LR, weight_decay=1e-5)
    loss_fn = eval.lossFunc(device).to(device)
    loss_baseline = eval_baseline.lossFunc(device).to(device)
    w_locals = [copy.deepcopy(model.state_dict()) for _ in range(C.CLIENTS)]
    proto_glob = None  # Used only for FedAvgProto

    # Early stopping and best-result tracking
    best_auc = -1.0  # >>> NEW: initialize to -1 so the first epoch can always overwrite it
    best_acc = 0.0
    best_rmse = 0.0
    best_epoch = 0
    best_state = None  # >>> NEW: store the best model weights
    patience, counter = 2, 0

    prev_stu_ema, prev_sub_ema = None, None  # MoP+EMA
    proto_student_list = []
    proto_subgraph_list = []

    # >>> NEW: per-fold best checkpoint path (overwritten when improved)
    best_ckpt_path = os.path.join(CKPT_DIR, f"fold{fold_idx + 1}_best.pth")

    for epoch in range(C.EPOCH):
        print(f"[{model_name} + {agg_method}] GPU:{device_id}  epoch {epoch + 1}")

        if agg_method in ["FedAvgProto", "FedProto"]:
            proto_student = (
                proto_glob["student_proto"]
                if (proto_glob is not None and "student_proto" in proto_glob)
                else None
            )
            proto_subgraph = (
                proto_glob["subgraph_proto"]
                if (proto_glob is not None and "subgraph_proto" in proto_glob)
                else None
            )
        else:
            proto_student, proto_subgraph = None, None

        # Training branch
        if agg_method in ["FedAvgProto", "FedProto"]:
            model, optimizer, w_locals, init_w, stu_proto_locals, subg_proto_locals = eval.train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                w_locals,
                device,
                proto_student,
                proto_subgraph,
                local_epochs=5,
            )

            # ===== MoP + EMA: server-side prototype aggregation for personalization =====
            proto_student_list, proto_subgraph_list, ema_state = server_prototype_update(
                stu_proto_locals,
                subg_proto_locals,
                prev_stu_ema,
                prev_sub_ema,
                K_stu=4,
                K_sub=4,
                beta=0.2,
                alpha_mop=10.0,
                alpha_ema=8.0,
                gmin=0.1,
                gmax=0.9,
                device=device,
            )

            # Update cross-round EMA states
            prev_stu_ema = ema_state["prev_stu_ema"]
            prev_sub_ema = ema_state["prev_sub_ema"]

        else:
            model, optimizer, w_locals, init_w = eval_baseline.train_epoch(
                model,
                train_loader,
                optimizer,
                loss_baseline,
                w_locals,
                device,
            )
            stu_proto_locals, subg_proto_locals = None, None

        # Aggregation
        if agg_method == "FedAtt":
            w_glob = agg_func(w_locals, init_w)
            proto_glob = None
        elif agg_method in ["FedAvg", "FedProx", "FedNoise"]:
            ratios = eval.calculate_data_ratios(train_loader)
            w_glob = agg_func(w_locals, ratios)
            proto_glob = None
        elif agg_method == "FedAvgProto":
            ratios = eval.calculate_data_ratios(train_loader)
            proto_locals = {
                "student_proto": stu_proto_locals,
                "subgraph_proto": subg_proto_locals,
            }

            # w_glob, proto_glob = agg_func(w_locals, proto_locals, ratios)  # Original prototype averaging
            w_glob, _ = agg_func(w_locals, proto_locals, ratios)  # Ignore the prototype output here

            # Override with our MoP+EMA results
            proto_glob = {
                "student_proto": proto_student_list,  # Now a list: one prototype per client
                "subgraph_proto": proto_subgraph_list,
            }

        elif agg_method == "FedProto":
            proto_student_glob = agg_func(proto_locals=stu_proto_locals, device=device)
            proto_glob = {
                "student_proto": proto_student_glob["proto"] if proto_student_glob is not None else None,
                "subgraph_proto": None,
            }
            w_glob = init_w
        else:
            w_glob = agg_func(w_locals)
            proto_glob = None

        # Move prototypes to the target device
        if proto_glob is not None:
            for k, v in proto_glob.items():
                if v is not None:
                    # Support list-style prototypes: move each element individually
                    if isinstance(v, list):
                        proto_glob[k] = [t.to(device) for t in v]
                    else:
                        proto_glob[k] = v.to(device)

        # Synchronize the global model
        model.load_state_dict(w_glob)

        # ======== >>> NEW: PeerInit for personalized downstream initialization ========
        # Choose the reference prototypes for similarity measurement;
        # the behavior prototypes from the current round are recommended.
        ref_protos = stu_proto_locals

        # If you want to use the MoP+EMA personalized references instead, use:
        # ref_protos = proto_glob["student_proto"]  # This is a set of per-client prototypes

        w_locals = peer_init(
            w_locals=w_locals,  # Locally trained models from the current round
            ref_protos=ref_protos,  # Reference prototypes for similarity
            topM=8,  # Tunable: 4/8/12
            tau=10.0,  # Tunable: 8/10/14
            use_cpu=False,  # Set to True when GPU memory is limited
        )

        # We directly use the PeerInit output as the next-round ``m_locals``.
        # Since ``train_epoch`` already applies ``Apply(Init_w, m_locals[idx])``,
        # no interface changes are needed.
        # ======== <<< NEW end ========

        # Validation
        with torch.no_grad():
            if agg_method in ["FedAvgProto", "FedProto"]:
                auc, acc, rmse = eval.test_epoch(model, val_loader, loss_fn, device)
            else:
                auc, acc, rmse = eval_baseline.test_epoch(model, val_loader, loss_baseline, device)

            improved = auc > best_auc
            if improved:
                best_auc, best_acc, best_rmse = auc, acc, rmse
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())  # >>> NEW: save the current best
                _save_ckpt(_to_cpu_state(best_state), best_ckpt_path)  # >>> NEW: overwrite best.pth
                print(f"[SAVE] New best at epoch {best_epoch} | AUC={best_auc:.6f} -> {best_ckpt_path}")

            counter = 0 if improved else counter + 1

        print(
            f"Best AUC: {best_auc:.6f}  ACC: {best_acc:.6f}  RMSE: {best_rmse:.6f}  @epoch {best_epoch}"
        )
        print("-" * 60)

        if counter >= patience:
            print(f"Early stopping: AUC did not improve for {patience} epochs.")
            break

    # >>> NEW: if there was never any improvement (extreme case), save the current weights
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        _save_ckpt(_to_cpu_state(best_state), best_ckpt_path)
        print(f"[SAVE] No-improvement case: saved current state to {best_ckpt_path}")

    # Testing: use the weights with the best validation AUC
    print("Testing (with best val-AUC weights)...")
    model.load_state_dict(best_state)  # >>> NEW

    with torch.no_grad():
        if agg_method in ["FedAvgProto", "FedProto"]:
            t_auc, t_acc, t_rmse = eval.test_epoch(model, test_loader, loss_fn, device)
        else:
            t_auc, t_acc, t_rmse = eval_baseline.test_epoch(model, test_loader, loss_baseline, device)

    print(
        f"[Fold {fold_idx + 1}] Test AUC: {t_auc * 100:.2f}%  ACC: {t_acc * 100:.2f}%  RMSE: {t_rmse:.4f}"
    )

    # >>> NEW: save a final checkpoint with informative naming
    final_ckpt_name = f"{C.DATASET}-{model_name}_{agg_method}-AUC{best_auc * 100:.2f}.pth"
    final_ckpt_path = os.path.join(CKPT_DIR, final_ckpt_name)
    _save_ckpt(_to_cpu_state(best_state), final_ckpt_path)
    print(f"[SAVE] Final best checkpoint saved to: {final_ckpt_path}")

    results_auc.append(t_auc)
    results_acc.append(t_acc)

    # Note: in some implementations rmse may be a tensor, so convert it if needed
    results_rmse.append(float(t_rmse if isinstance(t_rmse, float) else t_rmse.item()))

# ---- Summary ----
print("===== Cross-Validation Results =====")
print(f"Model: {model_name}, Aggregator: {agg_method}, Dataset: {C.DATASET}")
print(f"Avg Test AUC : {np.mean(results_auc) * 100:.2f}% ± {np.std(results_auc) * 100:.2f}%")
print(f"Avg Test ACC : {np.mean(results_acc) * 100:.2f}% ± {np.std(results_acc) * 100:.2f}%")
print(f"Avg Test RMSE: {np.mean(results_rmse):.4f} ± {np.std(results_rmse):.4f}")
print(f"AUC list : {[round(x, 4) for x in results_auc]}")
print(f"ACC list : {[round(x, 4) for x in results_acc]}")
print(f"RMSE list: {[round(x, 4) for x in results_rmse]}")
print(f"[CKPT DIR] {CKPT_DIR}")  # >>> NEW: show the checkpoint directory