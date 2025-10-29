import matplotlib.pyplot as plt
import numpy as np


def plot_curves(history, out_png):
    # history: dict(epoch-> {"train_acc":..., "val_acc":...})
    epochs = sorted(history.keys())
    tr = [history[e]["train_acc"] for e in epochs]
    va = [history[e]["val_acc"] for e in epochs]
    plt.figure()
    plt.plot(epochs, tr, label="train_acc")
    plt.plot(epochs, va, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_confusion(cm, classes, out_png):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick = np.arange(len(classes))
    plt.xticks(tick, classes, rotation=45)
    plt.yticks(tick, classes)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
