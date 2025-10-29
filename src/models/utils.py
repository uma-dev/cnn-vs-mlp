import os
import random
import time
import yaml
import torch
import numpy as np


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    # poor-man "defaults_from"
    if isinstance(cfg, dict) and cfg.get("defaults_from"):
        base = load_config(cfg["defaults_from"])
        base.update({k: v for k, v in cfg.items() if k != "defaults_from"})
        return base
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from(cfg_device):
    return torch.device(
        "cuda" if (cfg_device == "cuda" and torch.cuda.is_available())
        else "cpu"
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
