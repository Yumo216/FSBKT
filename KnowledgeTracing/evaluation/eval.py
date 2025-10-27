import tqdm
import torch
from KnowledgeTracing.Constant import Constants as C

import torch.nn as nn
from sklearn import metrics

import copy
import numpy as np
from Fed import FedAvg, Apply

'''Updated at 5/29'''

def performance(ground_truth, prediction):
    # 转为 numpy
    y_true = ground_truth.detach().cpu().numpy()
    y_pred = prediction.detach().cpu().numpy()

    # AUC & ACC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(y_true, np.round(y_pred))

    # RMSE
    # rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rmse = torch.sqrt(torch.sum((ground_truth - prediction) ** 2) / len(prediction)).detach().cpu().numpy()


    print(f'auc: {auc:.4f}  acc: {acc:.4f}  rmse: {rmse:.4f}')
    return auc, acc, rmse


class lossFunc(nn.Module):
    def __init__(self, device, proto_weight=0.5):
        super(lossFunc, self).__init__()
        self.loss_fn = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.proto_weight = proto_weight
        self.device = device

    # ========== 可选正则项工具函数 ==========

    # 多样性正则（原型去相关）
    def prototype_diversity_loss(self, proto):
        if proto is None or proto.shape[0] < 2:
            return torch.tensor(0., device=self.device)
        proto_norm = proto / (proto.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(proto_norm, proto_norm.t())  # [K, K]
        off_diag = sim_matrix - torch.eye(proto.shape[0], device=self.device)
        diversity = off_diag.abs().sum() / (proto.shape[0] * (proto.shape[0] - 1))
        return diversity


    # 熵正则（输出分布不过于极端）
    def entropy_regularization(self, prob):
        return -(prob * torch.log(prob + 1e-8) + (1 - prob) * torch.log(1 - prob + 1e-8)).mean()

    # ========== 主损失 ==========
    def forward(self, pred, datas, student_proto=None, subgraph_proto=None,
                proto_glob_student=None, proto_glob_subgraph=None):
        q_input = datas[0].long().squeeze(-1)   # [B, L]
        resp = datas[2].float().squeeze(-1)     # [B, L]
        q_target = torch.clamp(q_input[:, 1:] - 1, min=0, max=C.QUES-1)
        pred = pred[:, :-1, :]  # [B, L-1, Q]
        pred_selected = torch.gather(pred, dim=2, index=q_target.unsqueeze(-1)).squeeze(-1)
        label = resp[:, 1:] - 1
        mask = resp[:, 1:].ge(1)
        pred_filtered = pred_selected[mask]
        label_filtered = label[mask]

        prob = self.sigmoid(pred_filtered)
        task_loss = self.loss_fn(prob, label_filtered.float())

        proto_student_loss = torch.tensor(0.0, device=self.device)
        proto_subgraph_loss = torch.tensor(0.0, device=self.device)
        if proto_glob_student is not None and student_proto is not None:
            proto_student_loss = nn.functional.mse_loss(student_proto, proto_glob_student.detach().expand_as(student_proto))
        if proto_glob_subgraph is not None and subgraph_proto is not None:
            proto_subgraph_loss = nn.functional.mse_loss(subgraph_proto, proto_glob_subgraph.detach().expand_as(subgraph_proto))

        loss = task_loss + self.proto_weight * proto_student_loss + self.proto_weight * proto_subgraph_loss

        # 可选正则项
        diversity_loss = torch.tensor(0., device=self.device)
        entropy_loss = torch.tensor(0., device=self.device)
        diversity_loss = self.prototype_diversity_loss(student_proto)
        entropy_loss = self.entropy_regularization(prob)
        loss += 0.1 * diversity_loss + 0.1 * entropy_loss

        return loss, prob, label_filtered.float(), task_loss.item(), proto_student_loss.item(), proto_subgraph_loss.item()
'''
def train_epoch(model, Loader, optimizer, loss_func, m_locals, device,
                proto_glob_student=None, proto_glob_subgraph=None, local_epochs=5):
    w_locals = []
    student_proto_locals, subgraph_proto_locals = [], []

    Init_w = copy.deepcopy(model.state_dict())  # 初始全局模型参数

    print(f"Number of clients: {len(Loader)}")

    for idx, trainLoader in enumerate(tqdm.tqdm(Loader, desc="Training clients", mininterval=2)):  # each client
        model.load_state_dict(Apply(Init_w, m_locals[Loader.index(trainLoader)]))
        student_proto_batch, subgraph_proto_batch = [], []    # 用于收集该客户端所有学生的原型
        task_loss_sum, proto_student_loss_sum, proto_subgraph_loss_sum, n_batch = 0.0, 0.0, 0.0, 0
        for e in range(local_epochs):
            for batch in trainLoader:  # each student （divide by max_step）
                batch = batch.to(device)
                datas = torch.chunk(batch, 3, 2)

                logit, student_proto, subgraph_proto = model(datas[1].squeeze(-1), datas[0].squeeze(-1))
                student_proto_batch.append(student_proto.detach().cpu())
                subgraph_proto_batch.append(subgraph_proto.detach().cpu())
                # 损失函数分别传student_proto和subgraph_proto及各自的全局proto
                loss, _, _, task_loss, proto_student_loss, proto_subgraph_loss = loss_func(
                    logit, datas, student_proto, subgraph_proto,
                    proto_glob_student, proto_glob_subgraph
                )
                # loss, _, _, task_loss, proto_student_loss, proto_subgraph_loss = loss_func(
                #     logit, datas, student_proto, None,
                #     proto_glob_student, None
                # )  # 1️⃣ w/o Structural Prototype
                # loss, _, _, task_loss, proto_student_loss, proto_subgraph_loss = loss_func(
                #     logit, datas, None, subgraph_proto,
                #     None, proto_glob_subgraph
                # )  # 2️⃣ w/o Behavioral Prototype
                task_loss_sum += task_loss
                proto_student_loss_sum += proto_student_loss
                proto_subgraph_loss_sum += proto_subgraph_loss
                n_batch += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 客户端级原型聚合
        client_student_proto = torch.cat(student_proto_batch, dim=0).mean(dim=0)      # [H]
        client_subgraph_proto = torch.cat(subgraph_proto_batch, dim=0).mean(dim=0)    # [D]
        student_proto_locals.append(client_student_proto)
        subgraph_proto_locals.append(client_subgraph_proto)

        w_locals.append(copy.deepcopy(model.state_dict()))
        model.load_state_dict(Init_w)

    print(f"Client {len(w_locals)} model added.")
    # 分别返回本地student_proto和subgraph_proto

    return model, optimizer, w_locals, Init_w, student_proto_locals, subgraph_proto_locals
'''
def train_epoch(model, Loader, optimizer, loss_func, m_locals, device,
                proto_glob_student=None, proto_glob_subgraph=None, local_epochs=5):
    import copy, tqdm, torch
    #  10.17 更新MoP+EMA版的train_epoch
    w_locals = []
    student_proto_locals, subgraph_proto_locals = [], []

    Init_w = copy.deepcopy(model.state_dict())  # 初始全局模型参数

    print(f"Number of clients: {len(Loader)}")

    for idx, trainLoader in enumerate(tqdm.tqdm(Loader, desc="Training clients", mininterval=2)):  # each client
        model.load_state_dict(Apply(Init_w, m_locals[Loader.index(trainLoader)]))

        # >>> NEW: 解析“每客户端个性化原型”或“单一全局原型”
        if isinstance(proto_glob_student, list):
            p_stu_i = proto_glob_student[idx].detach().to(device)
        else:
            p_stu_i = proto_glob_student.detach().to(device) if proto_glob_student is not None else None

        if isinstance(proto_glob_subgraph, list):
            p_sub_i = proto_glob_subgraph[idx].detach().to(device)
        else:
            p_sub_i = proto_glob_subgraph.detach().to(device) if proto_glob_subgraph is not None else None
        # <<< NEW end

        student_proto_batch, subgraph_proto_batch = [], []    # 用于收集该客户端所有学生的原型
        task_loss_sum, proto_student_loss_sum, proto_subgraph_loss_sum, n_batch = 0.0, 0.0, 0.0, 0

        for e in range(local_epochs):
            for batch in trainLoader:  # each student （divide by max_step）
                batch = batch.to(device)
                datas = torch.chunk(batch, 3, 2)

                logit, student_proto, subgraph_proto = model(
                    datas[1].squeeze(-1),    # x
                    datas[0].squeeze(-1)     # q_seq
                )

                # 收集本客户端该轮所有学生的原型
                student_proto_batch.append(student_proto.detach().cpu())
                subgraph_proto_batch.append(subgraph_proto.detach().cpu())

                # >>> 修改点：把“对齐到全局proto”换成“对齐到本客户端的 p_stu_i / p_sub_i”
                loss, _, _, task_loss, proto_student_loss, proto_subgraph_loss = loss_func(
                    logit, datas, student_proto, subgraph_proto,
                    p_stu_i, p_sub_i
                )
                # <<< 修改点结束

                task_loss_sum += task_loss
                proto_student_loss_sum += proto_student_loss
                proto_subgraph_loss_sum += proto_subgraph_loss
                n_batch += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 客户端级原型聚合（把该客户端内所有学生/批次的原型做平均）
        client_student_proto = torch.cat(student_proto_batch, dim=0).mean(dim=0)      # [H]
        client_subgraph_proto = torch.cat(subgraph_proto_batch, dim=0).mean(dim=0)    # [D]
        student_proto_locals.append(client_student_proto)
        subgraph_proto_locals.append(client_subgraph_proto)

        w_locals.append(copy.deepcopy(model.state_dict()))
        model.load_state_dict(Init_w)

    print(f"Client {len(w_locals)} model added.")
    # 返回：模型、优化器、各客户端权重、初始化全局权重、各客户端的行为/结构原型（供服务器做 MoP+EMA）
    return model, optimizer, w_locals, Init_w, student_proto_locals, subgraph_proto_locals

def test_epoch(model, testLoader, loss_func, device):
    all_preds, all_labels = [], []

    for batch in tqdm.tqdm(testLoader, desc='Testing', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, dim=2)
        logit, *_ = model(datas[1].squeeze(-1), datas[0].squeeze(-1))
        loss, pred, label, *_ = loss_func(logit, datas)

        all_preds.append(pred)
        all_labels.append(label)

    # 拼接所有 batch 的预测与标签
    prediction = torch.cat(all_preds)
    ground_truth = torch.cat(all_labels)
    # print("Prediction values:", prediction.detach().cpu().numpy())
    return performance(ground_truth, prediction)



def calculate_data_ratios(Loader):
    client_data_sizes = [len(trainLoader.dataset) for trainLoader in Loader]  # 获取每个客户端的数据大小
    total_data_size = sum(client_data_sizes)  # 总数据量
    data_ratios = [size / total_data_size for size in client_data_sizes]  # 计算比例
    return data_ratios

