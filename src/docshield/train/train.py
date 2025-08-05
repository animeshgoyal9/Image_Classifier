"""Training script for DocShield models.

This module provides a command‑line entry point to train a document
classification model from scratch or from a pretrained backbone.  It
reads hyperparameters and dataset paths from a YAML configuration
file and supports optional overrides via the command line.

Example usage:

```bash
python -m src.docshield.train.train --config configs/train.yaml
python -m src.docshield.train.train --config configs/train.yaml optim.lr=1e-4 model.name=vit
```
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None  # type: ignore

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError:
    torch = None  # type: ignore
    DataLoader = None  # type: ignore

from ..data.datasets import DocClassificationDataset, create_class_to_idx
from ..data.transforms import build_transforms
from ..models.factory import create_model
from .metrics import compute_metrics
from .utils import set_seed, save_checkpoint


def parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parse key=value pairs into a nested dictionary."""
    cfg: Dict[str, Any] = {}
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to cast values
        if val.lower() in {"true", "false"}:
            casted: Any = val.lower() == "true"
        else:
            try:
                casted = int(val)
            except ValueError:
                try:
                    casted = float(val)
                except ValueError:
                    casted = val
        d[keys[-1]] = casted
    return cfg


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionary b into a (in place)."""
    for k, v in b.items():
        if isinstance(v, dict) and k in a and isinstance(a[k], dict):
            merge_dicts(a[k], v)
        else:
            a[k] = v
    return a


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Create training and validation dataloaders based on the configuration."""
    train_dir = Path(cfg["dataset"]["train_dir"])
    val_dir = Path(cfg["dataset"]["val_dir"])
    # Build class mapping
    class_to_idx = create_class_to_idx(train_dir)
    # Load augmentations
    aug_path = cfg["augmentations"]["train"]
    with open(aug_path, "r", encoding="utf-8") as f:
        aug_cfg = yaml.safe_load(f)
    transforms = build_transforms(aug_cfg)
    train_ds = DocClassificationDataset(train_dir, transform=transforms["train"], class_to_idx=class_to_idx)
    val_ds = DocClassificationDataset(val_dir, transform=transforms["val"], class_to_idx=class_to_idx)
    # Compute sample weights for WeightedRandomSampler
    import numpy as np
    labels = [label for _, label in train_ds.samples]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    batch_size = int(cfg["optim"]["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, class_to_idx


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> Tuple[float, float]:
    """Train the model for one epoch.  Returns (loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    """Validate the model and compute metrics."""
    model.eval()
    running_loss = 0.0
    preds_list: List[int] = []
    labels_list: List[int] = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())
    val_loss = running_loss / len(labels_list)
    metrics = compute_metrics(preds_list, labels_list)
    metrics["loss"] = val_loss
    return val_loss, metrics


def train_main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    overrides = parse_overrides(args.overrides or [])
    cfg = merge_dicts(cfg, overrides)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    if not torch:
        raise RuntimeError("PyTorch is not installed.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, val_dl, class_to_idx = build_dataloaders(cfg)
    cfg["dataset"]["num_classes"] = len(class_to_idx)
    cfg["model"]["num_classes"] = len(class_to_idx)
    model_cfg = cfg["model"]
    model, _ = create_model(model_cfg)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["optim"]["epochs"])
    best_f1 = 0.0
    patience = cfg.get("early_stopping", {}).get("patience", 5)
    es_counter = 0
    checkpoint_dir = Path(cfg.get("logging", {}).get("checkpoint_dir", "models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["optim"]["epochs"]):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        scheduler.step()
        val_loss, metrics = validate(model, val_dl, criterion, device)
        val_f1 = metrics.get("f1_macro", 0.0)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{cfg['optim']['epochs']} "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Elapsed: {elapsed:.1f}s"
        )
        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            es_counter = 0
            save_checkpoint(model, checkpoint_dir / "best.ckpt")
        else:
            es_counter += 1
        if es_counter >= patience:
            print("Early stopping.")
            break
    print(f"Training complete. Best F1: {best_f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DocShield model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML")
    parser.add_argument("overrides", nargs="*", help="Configuration overrides (key=value)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_main(args)