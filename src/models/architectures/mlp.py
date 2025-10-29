import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers, num_classes: int, dropout: float = 0.3, activation="relu"):
        super().__init__()
        act = nn.ReLU if activation.lower() == "relu" else nn.GELU
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 3, 32, 32) -> flatten
        x = torch.flatten(x, 1)
        return self.net(x)
