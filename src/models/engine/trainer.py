import time
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from ..metrics import accuracy


def train_one_epoch(model, loader, opt, device, amp):
    model.train()
    ce = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=amp)
    acc_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(x)
            loss = ce(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        acc_sum += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return acc_sum / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    acc_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        acc_sum += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return acc_sum / n


def timed_train(model, loaders, optimizer, epochs, device, amp):
    train_loader, val_loader, _ = loaders
    history = {}
    t0 = time.perf_counter()
    best_val, best_state = -1, None
    for ep in range(1, epochs+1):
        t_ep0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, optimizer, device, amp)
        va = validate(model, val_loader, device)
        dt = time.perf_counter() - t_ep0
        history[ep] = {"train_acc": tr, "val_acc": va, "epoch_time_s": dt}
        if va > best_val:
            best_val, best_state = va, {k: v.cpu()
                                        for k, v in model.state_dict().items()}
    total = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, total
