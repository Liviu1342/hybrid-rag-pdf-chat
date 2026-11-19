"""Turn HTML into plain-text paragraphs, using only the standard library."""

from html.parser import HTMLParser

BLOCK_TAGS = {"p", "div", "li", "blockquote", "tr", "br", "h1", "h2", "h3", "h4", "h5", "h6"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
SKIPPED_TAGS = {"head", "script", "style"}  # their text is not part of the content


class ParagraphParser(HTMLParser):
    """Collect the visible text of an HTML document as (text, is_heading) paragraphs.

    It only reads text: it never loads external files or runs scripts.
    skipped_tags lets a caller hide more parts, e.g. menus on a web page.
    """

    def __init__(self, skipped_tags: set[str] = SKIPPED_TAGS) -> None:
        super().__init__()
        self._skipped_tags = skipped_tags
        self.paragraphs: list[tuple[str, bool]] = []
        self._words: list[str] = []
        self._in_heading = False
        self._skip_depth = 0  # > 0 while inside a skipped tag

    def _end_paragraph(self) -> None:
        text = " ".join(" ".join(self._words).split())  # collapse spaces and new lines
        if text:
            self.paragraphs.append((text, self._in_heading))
        self._words = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skipped_tags:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._end_paragraph()  # a new block closes the previous one
            self._in_heading = tag in HEADING_TAGS

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skipped_tags:
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


def html_to_paragraphs(html: str, skipped_tags: set[str] = SKIPPED_TAGS) -> list[tuple[str, bool]]:
    """Return the visible text of an HTML document as (text, is_heading) paragraphs."""
    parser = ParagraphParser(skipped_tags)
    parser.feed(html)
    parser.close()
    return parser.paragraphs