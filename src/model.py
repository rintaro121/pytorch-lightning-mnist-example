import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
