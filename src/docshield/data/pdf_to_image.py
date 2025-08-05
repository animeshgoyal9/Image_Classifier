"""Utilities for converting PDF files to PIL images.

This module provides a single function `pdf_to_images` which converts
pages of a PDF into PIL images.  It tries to use `pdf2image` if
available, falling back to `PyMuPDF` (fitz) if installed.  If neither
is available, it raises an ImportError.
"""

from __future__ import annotations

from typing import List

from PIL import Image

def pdf_to_images(path: str) -> List[Image.Image]:
    """Convert a PDF file into a list of PIL images.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of PIL.Image objects, one per page.

    Raises:
        ImportError: If neither pdf2image nor PyMuPDF are available.
        RuntimeError: If conversion fails.
    """
    images: List[Image.Image] = []
    # Try pdf2image
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(path)
        return images
    except Exception:
        pass
    # Try PyMuPDF (fitz)
    try:
        import fitz  # type: ignore

        doc = fitz.open(path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap()
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception:
        raise ImportError(
            "PDF conversion requires either pdf2image or PyMuPDF.  Install one of these packages."
        )