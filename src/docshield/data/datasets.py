"""Dataset definitions for document classification.

Provides PyTorch Dataset classes for loading images and (optionally) PDFs
from a directory structure.  Each subdirectory under the root directory
corresponds to a class name.  The dataset returns samples as
(image_tensor, label) pairs.

Example directory structure:

```
data/train/
  ssn_real/
    img1.jpg
    img2.png
  ssn_fake/
    img3.jpg
  dl_real/
    ...
  ...
```
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Tuple, Dict

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # Provide dummy Dataset for type hints if PyTorch is unavailable
    class Dataset:  # type: ignore
        pass

from .pdf_to_image import pdf_to_images


def list_image_files(root_dir: Path) -> List[Tuple[Path, str]]:
    """Recursively list image and PDF files under the root directory.

    Returns a list of tuples (file_path, class_name).
    """
    samples: List[Tuple[Path, str]] = []
    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for file_path in class_dir.rglob("*"):
            if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}:
                samples.append((file_path, class_name))
    return samples


class DocClassificationDataset(Dataset):
    """PyTorch Dataset for document classification.

    Args:
        root_dir: Directory containing class subdirectories.
        transform: A callable transform that takes a PIL image and returns a tensor.
        class_to_idx: Optional mapping from class name to integer index.  If not
            provided, the mapping is inferred from subdirectory names in
            alphabetical order.

    Each item in the dataset is a tuple `(image_tensor, label)`.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[Image.Image], any],
        class_to_idx: Dict[str, int] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Build list of (file_path, class_name)
        samples = list_image_files(self.root_dir)
        if not samples:
            raise ValueError(f"No image or PDF files found in {root_dir}")
        # Infer class_to_idx if not provided
        if class_to_idx is None:
            classes = sorted({cls for _, cls in samples})
            self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        # Store samples as (file_path, label_idx)
        self.samples: List[Tuple[Path, int]] = []
        for file_path, class_name in samples:
            if class_name not in self.class_to_idx:
                continue
            self.samples.append((file_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[any, int]:
        file_path, label = self.samples[index]
        # If PDF, convert to image and take the first page
        if file_path.suffix.lower() == ".pdf":
            images = pdf_to_images(str(file_path))
            if not images:
                raise RuntimeError(f"Failed to convert PDF {file_path} to images")
            image = images[0]
        else:
            image = Image.open(file_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_class_to_idx(root_dir: Path) -> Dict[str, int]:
    """Create a class_to_idx mapping from subdirectories under root."""
    classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
    return {cls: idx for idx, cls in enumerate(classes)}