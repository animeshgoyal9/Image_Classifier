"""Image transformation pipelines using Albumentations.

This module defines functions to construct training and validation
augmentation pipelines based on a YAML configuration.  It uses the
Albumentations library for composing complex transformations and
automatically converts PIL images into tensors compatible with PyTorch.
"""

from __future__ import annotations

from typing import List, Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_transforms(config: Dict[str, List[Dict[str, Any]]], image_size: int = 224) -> Dict[str, A.BasicTransform]:
    """Build training and validation transforms from a configuration.

    Args:
        config: A dictionary with keys 'train' and 'val', each mapping to a
            list of transform definitions.  Each definition must contain
            a 'name' key (matching an Albumentations class) and
            optionally 'params' specifying keyword arguments for that
            transform.
        image_size: Default image size if not specified in the config.

    Returns:
        A dictionary with keys 'train' and 'val' containing composed
        Albumentations transforms.
    """
    def compose(transforms_list: List[Dict[str, Any]]) -> A.Compose:
        transforms: List[A.BasicTransform] = []
        for tconf in transforms_list:
            name = tconf.get("name")
            if name is None:
                raise ValueError("Transform config must have a 'name' field")
            params = tconf.get("params", {})
            if name == "RandomResizedCrop" and "height" not in params:
                params.setdefault("height", image_size)
                params.setdefault("width", image_size)
            if name == "Resize" and "height" not in params:
                params.setdefault("height", image_size)
                params.setdefault("width", image_size)
            # Dynamically get transform class from Albumentations
            try:
                TransformClass = getattr(A, name)
            except AttributeError as e:
                raise ValueError(f"Unknown transform {name}: {e}") from e
            transforms.append(TransformClass(**params))
        # Always convert to tensor at the end
        transforms.append(ToTensorV2())
        # Add normalization to convert to float
        transforms.append(A.Normalize(mean=[0, 0, 0], std=[255, 255, 255], max_pixel_value=255))
        return A.Compose(transforms)

    train_transforms = compose(config.get("train", []))
    val_transforms = compose(config.get("val", []))
    return {"train": train_transforms, "val": val_transforms}