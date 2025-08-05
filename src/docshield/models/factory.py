"""Model factory for DocShield.

This module exposes a single function `create_model` which constructs
a deep learning model based on a configuration dictionary.  Supported
backbones include EfficientNet and Vision Transformer (ViT).  The
function returns the model and the feature dimension prior to the
classification head.
"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn

from .efficientnet import build_efficientnet
from .vit import build_vit


def create_model(config: dict) -> Tuple[nn.Module, int]:
    """Create a model according to the given configuration.

    Args:
        config: A configuration dictionary with at least the keys
            'name' (model name), 'num_classes' and optionally 'pretrained',
            'dropout', 'head_width', etc.

    Returns:
        A tuple `(model, feature_dim)` where `model` is an `nn.Module`
        ready for training and `feature_dim` is the dimension of the
        features before the final classification head.

    Raises:
        ValueError: If the model name is not supported.
    """
    name = config.get("name", "efficientnet").lower()
    num_classes = int(config.get("num_classes", 6))
    pretrained = bool(config.get("pretrained", True))
    dropout = float(config.get("dropout", 0.0))
    head_width = int(config.get("head_width", 128))

    if name in {"efficientnet", "efficientnet-b0", "efficientnet_b0"}:
        model, feature_dim = build_efficientnet(num_classes=num_classes, pretrained=pretrained, dropout=dropout, head_width=head_width)
    elif name in {"vit", "vit-base", "vision_transformer"}:
        model, feature_dim = build_vit(config)
    else:
        raise ValueError(f"Unsupported model name: {name}")

    return model, feature_dim