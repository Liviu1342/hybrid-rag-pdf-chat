"""Tests for Word and PowerPoint files, built in memory (no files on disk)."""

from io import BytesIO

from docx import Document
from pptx import Presentation
from pptx.util import Inches

from documents import PARAGRAPHS_PER_SECTION, Section
from office_loader import load_docx, load_pptx


def test_docx_is_split_at_headings() -> None:
    document = Document()
    document.add_paragraph("Welcome to our shop.")
    document.add_heading("Refund policy", level=1)
    document.add_paragraph("Refunds are accepted within 30 days.")
    buffer = BytesIO()
    document.save(buffer)

    assert load_docx(buffer.getvalue()) == [
        Section("Welcome to our shop.", "paragraph 1"),
        Section("Refund policy\nRefunds are accepted within 30 days.", "Refund policy, paragraph 2"),
    ]


def test_long_docx_without_headings_is_still_split() -> None:
    document = Document()
    for number in range(1, 46):  # 45 paragraphs
        document.add_paragraph(f"Paragraph {number}.")
    buffer = BytesIO()
    document.save(buffer)

    locations = [section.location for section in load_docx(buffer.getvalue())]
    assert PARAGRAPHS_PER_SECTION == 20
    assert locations == ["paragraph 1", "paragraph 21", "paragraph 41"]


def test_pptx_gives_one_section_per_slide_with_notes() -> None:
    presentation = Presentation()
    first = presentation.slides.add_slide(presentation.slide_layouts[1])  # title + content
    title = first.shapes.title
    notes = first.notes_slide.notes_text_frame
    assert title is not None and notes is not None
    title.text = "Shipping"
    notes.text = "Mention express delivery."
    presentation.slides.add_slide(presentation.slide_layouts[6])  # blank slide: no text
    third = presentation.slides.add_slide(presentation.slide_layouts[6])
    third.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1)).text_frame.text = "Refunds"
    buffer = BytesIO()
    presentation.save(buffer)

    assert load_pptx(buffer.getvalue()) == [
        Section("Shipping\nSpeaker notes: Mention express delivery.", "slide 1"),
        Section("Refunds", "slide 3"),
    ]