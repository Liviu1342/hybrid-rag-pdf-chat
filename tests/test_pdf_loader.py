"""Tests for splitting page text into chunks (no real PDF needed)."""

from pdf_loader import split_into_chunks


def test_every_chunk_keeps_its_page_number() -> None:
    pages = [(2, "Text on page two."), (5, "Text on page five.")]
    chunks = split_into_chunks(pages, source="book.pdf")
    assert [(chunk.page, chunk.source) for chunk in chunks] == [(2, "book.pdf"), (5, "book.pdf")]


def test_long_page_is_split_into_chunks_no_longer_than_chunk_size() -> None:
    long_text = "word " * 500  # 2500 characters
    chunks = split_into_chunks([(1, long_text)], source="a.pdf", chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 1000 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(400))
    chunks = split_into_chunks([(1, text)], source="a.pdf", chunk_size=300, chunk_overlap=100)
    last_word_of_first = chunks[0].text.split()[-1]
    assert last_word_of_first in chunks[1].text  # the end of one chunk starts the next


def test_no_pages_gives_no_chunks() -> None:
    assert split_into_chunks([], source="empty.pdf") == []