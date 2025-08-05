"""Custom classification heads for DocShield models."""

import torch.nn as nn


class MLPHead(nn.Module):
    """A simple multi‑layer perceptron head for classification."""

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)