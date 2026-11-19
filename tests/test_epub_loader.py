"""Tests for .epub e-books, built in memory with ebooklib."""

from io import BytesIO

from ebooklib import epub

from documents import Section
from epub_loader import ParagraphParser, load_epub


def make_epub(chapters: list[str]) -> bytes:
    """Create a small e-book whose chapters have the given XHTML bodies."""
    book = epub.EpubBook()
    book.set_identifier("test-book")
    book.set_title("Test book")
    book.set_language("en")
    items = []
    for number, body in enumerate(chapters, start=1):
        chapter = epub.EpubHtml(title=f"Chapter {number}", file_name=f"chapter{number}.xhtml")
        chapter.content = body
        book.add_item(chapter)
        items.append(chapter)
    book.toc = items
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]  # the table of contents comes first, like in real books
    buffer = BytesIO()
    epub.write_epub(buffer, book)
    return buffer.getvalue()


def test_parser_keeps_visible_text_and_marks_headings() -> None:
    parser = ParagraphParser()
    parser.feed("<head><title>Ignored</title></head><h1>Refunds</h1><p>Within <b>30</b>\n days.</p>")
    parser.close()
    assert parser.paragraphs == [("Refunds", True), ("Within 30 days.", False)]


def test_chapters_are_read_in_order_without_the_table_of_contents() -> None:
    data = make_epub(["<h1>The Start</h1><p>Hello.</p>", "<p>No heading here.</p>"])
    assert load_epub(data) == [
        Section("The Start\nHello.", "The Start, paragraph 1"),
        Section("No heading here.", "chapter 2, paragraph 1"),
    ]