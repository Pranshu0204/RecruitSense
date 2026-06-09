"""PDF text extraction via PyMuPDF (fitz).

Supports filesystem paths or in-memory bytes (e.g., from a FastAPI
``UploadFile``). Pages are joined with double newlines. Image-only / scanned
PDFs raise :class:`PDFParseError` since OCR is not configured.
"""

from pathlib import Path

import fitz

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF cannot be opened or yields no extractable text."""


def extract_text_from_pdf(source: str | Path | bytes) -> str:
    """Extract concatenated plain text from all pages of a PDF.

    Args:
        source: filesystem path (``str`` or ``Path``) or raw PDF ``bytes``.

    Returns:
        Plain text with pages joined by double newlines.

    Raises:
        PDFParseError: if the PDF cannot be opened or yields no extractable
            text (commonly the case for scanned/image-only PDFs).
    """
    try:
        if isinstance(source, str | Path):
            doc = fitz.open(str(source))
        else:
            doc = fitz.open(stream=source, filetype="pdf")
    except Exception as exc:
        raise PDFParseError(f"Failed to open PDF: {exc}") from exc

    pages: list[str] = []
    try:
        for i, page in enumerate(doc):
            text = (page.get_text("text") or "").strip()
            if text:
                pages.append(text)
            else:
                logger.warning("pdf_page_empty", page_index=i)
    finally:
        doc.close()

    if not pages:
        raise PDFParseError(
            "PDF yielded no extractable text "
            "(likely a scanned or image-only PDF; OCR is not enabled)."
        )

    return "\n\n".join(pages)
