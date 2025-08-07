"""Training utilities for DocShield."""

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'name': 'efficientnet',
            'num_classes': 6,
            'pretrained': True,
            'dropout': 0.2,
            'head_width': 128
        }
    }, path)
    print(f"Model saved to {path}")


def load_checkpoint(model: nn.Module, path: Path) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('model_config', {})
