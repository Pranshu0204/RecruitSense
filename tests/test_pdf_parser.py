"""PDF parser tests — happy path + error cases."""

import pytest

from backend.utils.pdf_parser import PDFParseError, extract_text_from_pdf


def test_extract_from_valid_pdf_bytes(tiny_pdf_bytes: bytes) -> None:
    """A real (in-memory) PDF round-trips through ``extract_text_from_pdf``."""
    text = extract_text_from_pdf(tiny_pdf_bytes)
    assert "Jane Smith" in text
    assert "FastAPI" in text


def test_invalid_bytes_raises() -> None:
    """Garbage input must raise ``PDFParseError``, not crash."""
    with pytest.raises(PDFParseError):
        extract_text_from_pdf(b"not a pdf at all")


def test_missing_path_raises() -> None:
    with pytest.raises(PDFParseError):
        extract_text_from_pdf("/nonexistent/path/to.pdf")


def test_image_only_pdf_raises() -> None:
    """A PDF with no extractable text (blank page) must raise ``PDFParseError``."""
    import fitz

    doc = fitz.open()
    doc.new_page()  # blank page, no text inserted
    blank_bytes = doc.tobytes()
    doc.close()
    with pytest.raises(PDFParseError):
        extract_text_from_pdf(blank_bytes)
