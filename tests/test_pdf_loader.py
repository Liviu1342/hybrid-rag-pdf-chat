"""Tests for reading PDF bytes into sections."""

from io import BytesIO

from pypdf import PdfWriter

from pdf_loader import load_pdf


def test_pdf_without_text_gives_no_sections() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)  # like a scanned page: no text layer
    buffer = BytesIO()
    writer.write(buffer)
    assert load_pdf(buffer.getvalue()) == []
