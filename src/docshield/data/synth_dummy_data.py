"""Synthetic dummy data generator for DocShield.

This script creates a small synthetic dataset with obvious text and
watermarks to validate the end‑to‑end pipeline without using real PII.
Each document type (ssn, dl, bankstmt) has two classes: `*_real` and
`*_fake`.  Images are generated with random background colors and
overlaid text indicating the class.  Optionally, PDF files can be
produced by saving images as single‑page PDFs.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont


CLASSES = [
    "ssn_real",
    "ssn_fake",
    "dl_real",
    "dl_fake",
    "bankstmt_real",
    "bankstmt_fake",
]


def generate_image(text: str, size: int = 256) -> Image.Image:
    """Generate a synthetic image with colored background and text."""
    # Random pastel background
    bg_color = tuple(random.randint(200, 255) for _ in range(3))
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size // 10)
    except IOError:
        font = ImageFont.load_default()
    text_color = (0, 0, 0)
    try:
        # Pillow < 10.0
        w, h = draw.textsize(text, font=font)  # type: ignore[attr-defined]
    except AttributeError:
        # Pillow >= 10.0: use textbbox to compute bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

    # Center text
    position = ((size - w) // 2, (size - h) // 2)
    draw.text(position, text, fill=text_color, font=font)
    # Add faint watermark
    watermark = "fake" if "fake" in text else "real"
    wm_color = (150, 150, 150)
    draw.text((5, size - 20), watermark, fill=wm_color, font=font)
    return img


def generate_dataset(output_dir: Path, num_samples: int = 10, create_pdf: bool = False) -> None:
    """Generate synthetic dataset into the specified directory.

    Args:
        output_dir: Root directory where class subdirectories are created.
        num_samples: Number of images per class.
        create_pdf: Whether to also save each image as a PDF file.
    """
    for cls in CLASSES:
        cls_dir = output_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(num_samples):
            img = generate_image(cls)
            img_path = cls_dir / f"{cls}_{i}.jpg"
            img.save(img_path)
            if create_pdf:
                pdf_path = cls_dir / f"{cls}_{i}.pdf"
                img.save(pdf_path, "PDF", resolution=100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dummy data for DocShield")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for synthetic data")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per class")
    parser.add_argument("--pdf", action="store_true", help="Also generate PDF files for each image")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    generate_dataset(output_dir, args.num_samples, args.pdf)
    print(f"Synthetic data generated at {output_dir}")


if __name__ == "__main__":
    main()