"""Vision Transformer helper functions."""

from __future__ import annotations

from typing import Dict, Tuple

import torch.nn as nn

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except ImportError:
    vit_b_16 = None  # type: ignore
    ViT_B_16_Weights = None  # type: ignore


def build_vit(config: Dict) -> Tuple[nn.Module, int]:
    """Construct a Vision Transformer (ViT) model.

    Args:
        config: A dictionary containing model configuration.  Must include
            `num_classes` and optionally `pretrained` and `dropout`.

    Returns:
        A tuple `(model, feature_dim)`.
    """
    if vit_b_16 is None:
        raise ImportError("torchvision>=0.14 is required for ViT models")
    num_classes = int(config.get("num_classes", 6))
    pretrained = bool(config.get("pretrained", True))
    dropout = float(config.get("dropout", 0.1))
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
    else:
        weights = None
    model = vit_b_16(weights=weights)
    # Replace the head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model, in_features