import torch
from sklearn.metrics import classification_report, confusion_matrix


def accuracy(logits, y):
    pred = logits.argmax(1)
    return (pred == y).float().mean().item()


def sklearn_report(logits, y, target_names):
    y_true = y.cpu().numpy()
    y_pred = logits.argmax(1).cpu().numpy()
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm
