"""Load .epub e-books: one chapter after another, in reading order."""

from html.parser import HTMLParser
from io import BytesIO

from ebooklib import epub

from documents import Section, group_paragraphs

BLOCK_TAGS = {"p", "div", "li", "blockquote", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIPPED_TAGS = {"head", "script", "style"}  # their text is not part of the book


class ParagraphParser(HTMLParser):
    """Collect the visible text of an XHTML chapter as (text, is_heading) paragraphs.

    Uses the HTML parser from the standard library: it only reads text and
    never loads external files or runs scripts found inside the book.
    """

    def __init__(self) -> None:
        super().__init__()
        self.paragraphs: list[tuple[str, bool]] = []
        self._words: list[str] = []
        self._in_heading = False
        self._skip_depth = 0  # > 0 while inside head, script or style tags

    def _end_paragraph(self) -> None:
        text = " ".join(" ".join(self._words).split())  # collapse spaces and new lines
        if text:
            self.paragraphs.append((text, self._in_heading))
        self._words = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._end_paragraph()  # a new block closes the previous one
            self._in_heading = tag in HEADING_TAGS

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_TAGS:
            self._end_paragraph()
            self._in_heading = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._words.append(data)

    def close(self) -> None:
        super().close()
        self._end_paragraph()  # text after the last tag


def load_epub(data: bytes) -> list[Section]:
    """Return sections located as "Chapter 3: The Return, paragraph 12"."""
    book = epub.read_epub(BytesIO(data))
    sections: list[Section] = []
    chapter_number = 0
    for item_id, _linear in book.spine:  # the spine is the reading order
        item = book.get_item_with_id(item_id)
        if not isinstance(item, epub.EpubHtml) or isinstance(item, epub.EpubNav):
            continue  # skip images, styles and the table of contents
        parser = ParagraphParser()
        parser.feed(item.get_content().decode("utf-8", errors="replace"))
        parser.close()
        if parser.paragraphs:
            chapter_number += 1
            # Until the chapter's own heading appears, cite it by its number
            sections.extend(group_paragraphs(parser.paragraphs, first_heading=f"chapter {chapter_number}"))
    return sections