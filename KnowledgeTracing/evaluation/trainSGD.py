import torch
import copy
import tqdm

def train_epoch_sgd(model, Loader, optimizer, loss_func, device, batch_limit=5):
    grads = []
    Init_w = copy.deepcopy(model.state_dict())

    print(f"Number of clients: {len(Loader)}")

    for idx, trainLoader in enumerate(tqdm.tqdm(Loader, desc="FedSGD clients", mininterval=2)):
        model.load_state_dict(Init_w)
        model.train()
        optimizer.zero_grad()

        client_grad = None
        used_batches = 0

        for batch in trainLoader:
            batch = batch.to(device)
            datas = torch.chunk(batch, 3, 2)
            logit = model(datas[1].squeeze(-1), datas[0].squeeze(-1))
            loss, _, _ = loss_func(logit, datas)
            loss.backward()

            # 累积梯度
            current_grad = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    current_grad[name] = param.grad.clone()
                else:
                    current_grad[name] = torch.zeros_like(param.data)

            if client_grad is None:
                client_grad = current_grad
            else:
                for name in client_grad:
                    client_grad[name] += current_grad[name]

            optimizer.zero_grad()
            used_batches += 1
            if used_batches >= batch_limit:
                break

        # 平均这个客户端的多 batch 梯度
        for name in client_grad:
            client_grad[name] /= used_batches

        grads.append(client_grad)

    return grads
