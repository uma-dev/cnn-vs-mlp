import torch
from ..metrics import sklearn_report

CIFAR10_CLASSES = ["airplane", "automobile", "bird",
                   "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        all_logits.append(model(x).cpu())
        all_y.append(y)
    logits = torch.cat(all_logits, 0)
    y = torch.cat(all_y, 0)
    report, cm = sklearn_report(logits, y, CIFAR10_CLASSES)
    return report, cm
