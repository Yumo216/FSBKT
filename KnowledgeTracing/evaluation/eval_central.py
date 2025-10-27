# eval_central.py
import tqdm, torch, numpy as np
import torch.nn as nn
from sklearn import metrics
from KnowledgeTracing.Constant import Constants as C

def performance(y_true_t, y_pred_t):
    y_true = y_true_t.detach().cpu().numpy()
    y_pred = y_pred_t.detach().cpu().numpy()
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(y_true, np.round(y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f'auc: {auc:.4f}  acc: {acc:.4f}  rmse: {rmse:.4f}')
    return auc, acc, rmse

def _ensure_logit_3d(pred):
    """把模型输出规整为 [B, L, Q] 的 logit 张量；若不满足，给出明确报错。"""
    # 允许 tuple/list：取第一个为 logit
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if not torch.is_tensor(pred):
        raise TypeError(f"Model forward must return Tensor or (Tensor,...), got {type(pred)}")
    if pred.dim() == 3:
        B,L,Q = pred.shape
        if Q < C.QUES:
            raise RuntimeError(f"logit last dim {Q} < C.QUES {C.QUES}. "
                               f"请将模型的 output_dim 设为 C.QUES（题目数）。")
        # 若比题目数还大，裁到前 C.QUES
        if Q != C.QUES:
            pred = pred[:, :, :C.QUES]
        return pred
    elif pred.dim() == 2:
        # 常见于误设 output_dim=1 的情况
        raise RuntimeError(f"得到二维输出 {tuple(pred.shape)}，需要 [B,L,Q]。"
                           f"请把模型的输出维设为题目数：output_dim=C.QUES。")
    else:
        raise RuntimeError(f"不支持的输出维度：{pred.dim()}，需要 [B,L,Q]。")

class lossFunc(nn.Module):
    """集中式训练/评测统一的 BCE 损失（与 baseline/gather 逻辑一致）"""
    def __init__(self, device):
        super().__init__()
        self.bce = nn.BCELoss()
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, pred, datas):
        """
        pred:   模型输出（Tensor 或 tuple），将被规整为 [B, L, Q]
        datas:  torch.chunk(batch, 3, dim=2) 得到的 (q_input, x, resp)
        """
        pred = _ensure_logit_3d(pred)                     # [B, L, Q]
        q_input, _, resp = datas                          # [B,L,1], [B,L,1], [B,L,1]
        q_target = torch.clamp(q_input[:, 1:, :] - 1, 0, C.QUES - 1).long()  # [B, L-1, 1]
        logits_sel = pred[:, :-1, :]                      # [B, L-1, Q]
        logits_one = torch.gather(logits_sel, dim=2, index=q_target).squeeze(-1)  # [B, L-1]
        prob = self.sig(logits_one)                       # [B, L-1]

        tgt = resp[:, 1:, :].float()                      # [B, L-1, 1]
        tgt_1d = tgt.reshape(-1, 1)                       # [B*(L-1),1]
        mask = tgt_1d.ge(1)
        y = (tgt_1d - 1.0)                                # {0,1}
        p = prob.reshape(-1, 1)

        p_masked = torch.masked_select(p, mask)           # [M]
        y_masked = torch.masked_select(y, mask)           # [M]
        loss = self.bce(p_masked, y_masked.float())
        return loss, p_masked, y_masked.float()

@torch.no_grad()
def test_epoch(model, loader, loss_fn, device):
    model.eval()
    y_all = torch.tensor([], device=device)
    p_all = torch.tensor([], device=device)
    for batch in tqdm.tqdm(loader, desc='Testing', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, dim=2)
        pred = model(datas[1].squeeze(-1), datas[0].squeeze(-1))
        loss, p, y = loss_fn(pred, datas)
        p_all = torch.cat([p_all, p])
        y_all = torch.cat([y_all, y])
    return performance(y_all, p_all)

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, n = 0.0, 0
    for batch in tqdm.tqdm(loader, desc='Train', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, 3, dim=2)
        pred = model(datas[1].squeeze(-1), datas[0].squeeze(-1))
        loss, _, _ = loss_fn(pred, datas)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()); n += 1
    return total / max(1, n)
