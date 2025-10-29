import argparse
import json
import os
import torch
from torchvision import datasets, transforms
from models.engine.evaluate import evaluate_full, CIFAR10_CLASSES
from models.viz import plot_confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    # infer model type from run_dir name
    run = os.path.basename(args.run_dir)
    model_type = "cnn" if run.startswith("cnn") else "mlp"

    # reconstruct model (look for config snapshot if you save it)
    if model_type == "cnn":
        from models.architectures.cnn import SimpleCNN
        model = SimpleCNN(blocks=[
            {"out_channels": 32, "kernel_size": 3,
                "pool": True, "bn": True, "dropout": 0.0},
            {"out_channels": 64, "kernel_size": 3,
                "pool": True, "bn": True, "dropout": 0.0},
            {"out_channels": 128, "kernel_size": 3,
                "pool": True, "bn": True, "dropout": 0.5},
        ], num_classes=10)
        # build classifier for 32x32
        # -> downsampled 3 pools => 4x4 with 128ch => 2048
        model.build_classifier(128*4*4, [128])
    else:
        from models.architectures.mlp import MLP
        model = MLP(3072, [512, 256, 128], 10, 0.3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(os.path.join(
        args.run_dir, "best.pt"), map_location=device))
    model.to(device)

    test_t = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(
                                 (0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))]
                                )
    test = datasets.CIFAR10("./data", train=False,
                            download=True, transform=test_t)
    from torch.utils.data import DataLoader
    loader = DataLoader(test, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=False)

    report, cm = evaluate_full(model, loader, device)
    with open(os.path.join(args.run_dir, "test_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot_confusion(cm, CIFAR10_CLASSES, os.path.join(
        args.run_dir, "confusion_matrix.png"))
    print("Saved metrics and confusion matrix.")


if __name__ == "__main__":
    main()
