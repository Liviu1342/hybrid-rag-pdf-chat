"""Load Word (.docx) and PowerPoint (.pptx) files."""

from io import BytesIO

from docx import Document
from pptx import Presentation
from pptx.shapes.autoshape import Shape

from documents import Section, group_paragraphs

def load_docx(data: bytes) -> list[Section]:
    """Return sections split at every heading, located as "Refund policy, paragraph 12"."""
    document = Document(BytesIO(data))
    paragraphs: list[tuple[str, bool]] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:  # skip empty lines
            style = getattr(paragraph.style, "name", None) or ""  # e.g. "Heading 1", "Normal"
            paragraphs.append((text, style.startswith(("Heading", "Title"))))
    return group_paragraphs(paragraphs)


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