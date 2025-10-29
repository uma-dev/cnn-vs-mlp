from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms


def build_transforms(aug):
    t = []
    if aug.get("random_crop", True):
        t.append(transforms.RandomCrop(32, padding=4))
    if aug.get("random_hflip", True):
        t.append(transforms.RandomHorizontalFlip())
    t += [transforms.ToTensor()]
    if aug.get("normalize", True):
        t += [transforms.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2470, 0.2435, 0.2616))]
    return transforms.Compose(t)


def build_loaders(cfg):
    aug = build_transforms(cfg["data"]["aug"])
    test_t = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    root = cfg["data"]["root"]
    full = datasets.CIFAR10(root=root, train=True,
                            download=True, transform=aug)
    test = datasets.CIFAR10(root=root, train=False,
                            download=True, transform=test_t)

    splits = cfg["data"]["splits"]
    n = len(full)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    n_rest = n - n_train - n_val
    train, val, _ = random_split(
        full, [n_train, n_val, n_rest], generator=None)  # ignore rest
    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    pin = bool(cfg["data"].get("pin_memory", False))
    return (
        DataLoader(train, batch_size=bs, shuffle=True,
                   num_workers=nw, pin_memory=pin),
        DataLoader(val, batch_size=bs, shuffle=False,
                   num_workers=nw, pin_memory=pin),
        DataLoader(test, batch_size=bs, shuffle=False,
                   num_workers=nw, pin_memory=pin),
    )
