from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - torch may be optional during import
    torch = None  # type: ignore
    nn = Any  # type: ignore


def set_seed(seed: int) -> None:
    """Seed random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Save a model's state dictionary to ``path``."""
    if torch is None:
        raise RuntimeError("PyTorch is required to save checkpoints")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
