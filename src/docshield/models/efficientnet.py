"""EfficientNet helper functions."""

from __future__ import annotations

import torch.nn as nn

try:
    from torchvision import models
except ImportError:
    models = None  # type: ignore


def build_efficientnet(
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.2,
    head_width: int = 128,
) -> tuple[nn.Module, int]:
    """Construct an EfficientNet-B0 model with a custom classification head.

    Returns the model and feature dimension just before the head.
    """
    if models is None:
        raise ImportError("torchvision is required to build EfficientNet")
    base = models.efficientnet_b0(pretrained=pretrained)
    # Replace classifier with custom head
    in_features = base.classifier[1].in_features
    head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, head_width),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(head_width, num_classes),
    )
    base.classifier = head
    return base, in_features