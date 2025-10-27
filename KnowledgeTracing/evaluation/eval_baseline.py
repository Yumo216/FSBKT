import tqdm
import torch
from KnowledgeTracing.Constant import Constants as C

import torch.nn as nn
from sklearn import metrics

import copy
import numpy as np
from Fed import FedAvg, Apply


def performance(ground_truth, prediction):
    # 转为 numpy
    y_true = ground_truth.detach().cpu().numpy()
    y_pred = prediction.detach().cpu().numpy()

    # AUC & ACC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(y_true, np.round(y_pred))

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f'auc: {auc:.4f}  acc: {acc:.4f}  rmse: {rmse:.4f}')
    return auc, acc, rmse


class lossFunc(nn.Module):
    def __init__(self, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.q = C.QUES
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, pred, datas):   # pred [BS, L, q]
        qshft = datas[0] -1 # [BS,L,1] 0 ~ C.Ques

        qshft_adjusted = torch.where(qshft < 0, torch.tensor(0, device=self.device),
                                            qshft)
        # 取出 pred 的前 49 个步骤
        pred_first_49 = pred[:, :C.MAX_STEP - 1, :]  # [BS, 49, C.Ques]

        # 取出 qshft_adjusted 的后 49 个元素
        qshft_last_49 = qshft_adjusted[:, 1:C.MAX_STEP, :]  # [BS, 199, 1]
        pred_one = torch.gather(pred_first_49, dim=-1, index=(qshft_last_49).long())


        target = datas[2]
        target = target[:, 1:, :]
        # pred_one = pred_one[:, :-1, :]
        target_1d = target.reshape(-1, 1)  # [batch_size * seq_len, 1]  [bs*199]
        mask = target_1d.ge(1)  # [batch_size * seq_len, 1]
        pred_1d = pred_one.reshape(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)  # [batch_size * seq_len - be masked, 1]
        filtered_target = torch.masked_select(target_1d, mask) - 1

        pred = self.sig(filtered_pred)

        loss = self.crossEntropy(pred, filtered_target.float())

        return loss, pred, filtered_target.float()




def train_epoch(model, Loader, optimizer, loss_func, m_locals, device):
    w_locals = []

    Init_w = copy.deepcopy(model.state_dict())  # 保存初始模型的权重
    E = 5

    print(f"Number of clients: {len(Loader)}")

    for idx, trainLoader in enumerate(tqdm.tqdm(Loader, desc="Training clients", mininterval=2)):  # 每个 trainLoader 对应一个客户端的数据。
        # print(f"Now running client {idx + 1}")

        model.load_state_dict(Apply(Init_w, m_locals[Loader.index(trainLoader)]))

        loss = torch.tensor([], device=device)
        for e in range(E):  # 在一次全局epoch下，每个客户端的epoch
            for batch in trainLoader:
                batch = batch.to(device)
                # shape of a batch:[batch_size, max_step, 3]
                datas = torch.chunk(batch, 3, 2)  # [q,q+r,r]

                logit = model(datas[1].squeeze(-1), datas[0].squeeze(-1))

                loss, _, _ = loss_func(logit, datas)  # 正确解包返回的元组
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        w_locals.append(copy.deepcopy(model.state_dict()))
        model.load_state_dict(Init_w)
    print(f"Client {len(w_locals)} model added.")

    print(f"w_locals length: {len(w_locals)}")
    print('loss:', loss.item())
    return model, optimizer, w_locals, Init_w


def test_epoch(model, testLoader, loss_func, device):
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, 2)

        logit = model(datas[1].squeeze(-1), datas[0].squeeze(-1))  # random initial emb
        loss, p, a = loss_func(logit, datas)
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, a])
    return performance(ground_truth, prediction)

def calculate_data_ratios(Loader):
    client_data_sizes = [len(trainLoader.dataset) for trainLoader in Loader]  # 获取每个客户端的数据大小
    total_data_size = sum(client_data_sizes)  # 总数据量
    data_ratios = [size / total_data_size for size in client_data_sizes]  # 计算比例
    return data_ratios
