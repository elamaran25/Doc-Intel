"""
Document ingestion: load a PDF (or set of images), render pages, and produce
a single multi-image payload with `--- PAGE N ---` markers — same pattern
used in production EOB extraction, refactored here from per-page calls to
one multi-image call per document.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image


@dataclass
class PageImage:
    page_number: int  # 1-indexed
    image: Image.Image

    def to_base64(self, fmt: str = "PNG") -> str:
        buf = io.BytesIO()
        self.image.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_pdf_pages(pdf_path: str | Path, dpi: int = 200) -> list[PageImage]:
    """Render every page of a PDF to a PIL image."""
    images = convert_from_path(str(pdf_path), dpi=dpi)
    return [PageImage(page_number=i + 1, image=img) for i, img in enumerate(images)]


def load_image_files(paths: list[str | Path]) -> list[PageImage]:
    """Load a list of pre-split page images (e.g. scanned .png/.jpg per page)."""
    pages = []
    for i, p in enumerate(paths):
        pages.append(PageImage(page_number=i + 1, image=Image.open(p).convert("RGB")))
    return pages


def build_multi_page_payload(pages: list[PageImage]) -> list[dict]:
    """
    Build the content list for a single multi-image chat completion call.
    Each image is preceded by a `--- PAGE N ---` text marker so the model can
    attribute extracted fields to a specific page (feeds `source_page` in
    ExtractedField).
    """
    content: list[dict] = []
    for page in pages:
        content.append({"type": "text", "text": f"--- PAGE {page.page_number} ---"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{page.to_base64()}"},
            }
        )
    return content


def preprocess_document(path: str | Path) -> list[dict]:
    """Entry point used by the extraction stage. Handles PDF or single image."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        pages = load_pdf_pages(path)
    else:
        pages = load_image_files([path])
    return build_multi_page_payload(pages)
