import torch.nn as nn


def conv_block(in_ch, out_ch, k=3, pool=True, bn=True, p_dropout=0.0):
    layers = [nn.Conv2d(in_ch, out_ch, k, padding=k//2), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    if pool:
        layers.append(nn.MaxPool2d(2))
    if p_dropout > 0:
        layers.append(nn.Dropout(p_dropout))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    def __init__(self, blocks, num_classes: int):
        super().__init__()
        in_ch = 3
        stages = []
        for b in blocks:
            stages.append(conv_block(in_ch, b["out_channels"], b.get("kernel_size", 3),
                                     b.get("pool", True), b.get("bn", True), b.get("dropout", 0.0)))
            in_ch = b["out_channels"]
        self.features = nn.Sequential(*stages)
        # infer flatten dim with a dummy pass in scripts/train.py
        self.classifier = None
        self.num_classes = num_classes
        self._flatten_dim = None

    def build_classifier(self, flatten_dim, fc):
        layers, prev = [], flatten_dim
        for h in fc:
            layers += [nn.Linear(prev, h),
                       nn.ReLU(inplace=True), nn.Dropout(0.5)]
            prev = h
        layers.append(nn.Linear(prev, self.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        if self.classifier is None:
            raise RuntimeError(
                "Classifier not built; call build_classifier(flatten_dim, fc) after probing shape.")
        return self.classifier(x)
