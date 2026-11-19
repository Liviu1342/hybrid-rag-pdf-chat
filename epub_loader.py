"""Load .epub e-books: one chapter after another, in reading order."""

from io import BytesIO

from ebooklib import epub

from documents import Section, group_paragraphs
from html_text import html_to_paragraphs


def load_epub(data: bytes) -> list[Section]:
    """Return sections located as "Chapter 3: The Return, paragraph 12"."""
    book = epub.read_epub(BytesIO(data))
    sections: list[Section] = []
    chapter_number = 0
    for item_id, _linear in book.spine:  # the spine is the reading order
        item = book.get_item_with_id(item_id)
        if not isinstance(item, epub.EpubHtml) or isinstance(item, epub.EpubNav):
            continue  # skip images, styles and the table of contents
        paragraphs = html_to_paragraphs(item.get_content().decode("utf-8", errors="replace"))
        if paragraphs:
            chapter_number += 1
            # Until the chapter's own heading appears, cite it by its number
            sections.extend(group_paragraphs(paragraphs, first_heading=f"chapter {chapter_number}"))
    return sections