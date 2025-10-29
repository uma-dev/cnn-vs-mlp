import argparse
import json
import os
import torch

from models.utils import load_config, set_seed, device_from, timestamp, ensure_dir, count_params
from models.data import build_loaders
from models.architectures.mlp import MLP
from models.architectures.cnn import SimpleCNN
from models.engine.trainer import timed_train
from models.viz import plot_curves


def build_model(cfg, device, sample_shape=(1, 3, 32, 32)):
    mcfg = cfg["model"]
    if mcfg["name"] == "mlp":
        model = MLP(input_dim=mcfg["input_dim"], hidden_layers=mcfg["hidden_layers"],
                    num_classes=mcfg["num_classes"], dropout=mcfg.get("dropout", 0.0))
    elif mcfg["name"] == "cnn":
        model = SimpleCNN(blocks=mcfg["blocks"],
                          num_classes=mcfg["num_classes"])
        # probe flatten dim
        with torch.no_grad():
            x = torch.zeros(sample_shape).to(device)
            feats = model.features(x).view(1, -1)
            flatten_dim = feats.size(1)
        model.build_classifier(flatten_dim, mcfg.get("fc", [128]))
    else:
        raise ValueError("Unknown model")
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = device_from(cfg["train"]["device"])
    loaders = build_loaders(cfg)

    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(
    ), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    history, total_time = timed_train(
        model, loaders, opt, cfg["train"]["epochs"], device, cfg["train"]["amp"])

    # create en experiment directory
    run_name = cfg["model"]["name"] + "-" + timestamp()
    out_dir = os.path.join(cfg["logging"]["out_dir"], run_name)
    ensure_dir(out_dir)
    torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    # metrics summary as json
    n_params = count_params(model)
    with open(os.path.join(out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {"params": n_params, "total_train_time_s": total_time}, f, indent=2)
    plot_curves(history, os.path.join(out_dir, "learning_curves.png"))
    print(f"Saved run -> {out_dir}")


if __name__ == "__main__":
    main()
