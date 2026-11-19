"""Load Word (.docx) and PowerPoint (.pptx) files."""

from io import BytesIO

from docx import Document
from pptx import Presentation
from pptx.shapes.autoshape import Shape

from documents import Section

PARAGRAPHS_PER_SECTION = 20  # a long chapter without sub-headings is still split


def load_docx(data: bytes) -> list[Section]:
    """Return sections split at every heading, located as "Refund policy, paragraph 12".

    Word has no fixed pages (they depend on the printer), so we cite the
    nearest heading plus the paragraph number, which works in any viewer.
    """
    document = Document(BytesIO(data))
    sections: list[Section] = []
    heading = ""  # the last heading seen; empty before the first one
    texts: list[str] = []
    first_paragraph = 0
    number = 0  # counts only paragraphs that contain text

    def flush() -> None:
        """Turn the collected paragraphs into one section."""
        if texts:
            where = f"paragraph {first_paragraph}"
            location = f"{heading}, {where}" if heading else where
            sections.append(Section("\n".join(texts), location))
            texts.clear()

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        number += 1
        style = getattr(paragraph.style, "name", None) or ""  # e.g. "Heading 1", "Normal"
        if style.startswith(("Heading", "Title")):
            flush()  # a new heading starts a new section
            heading = text
        elif len(texts) >= PARAGRAPHS_PER_SECTION:
            flush()
        if not texts:
            first_paragraph = number
        texts.append(text)
    flush()
    return sections


def load_pptx(data: bytes) -> list[Section]:
    """Return one section per slide (its text and speaker notes), located as "slide 7"."""
    presentation = Presentation(BytesIO(data))
    sections: list[Section] = []
    for slide_number, slide in enumerate(presentation.slides, start=1):
        texts = [
            shape.text_frame.text.strip()
            for shape in slide.shapes
            if isinstance(shape, Shape) and shape.text_frame.text.strip()  # skip pictures, charts
        ]
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes is not None and notes.text.strip():
                texts.append("Speaker notes: " + notes.text.strip())
        if texts:  # skip slides that are only pictures
            sections.append(Section("\n".join(texts), f"slide {slide_number}"))
    return sections