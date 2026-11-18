"""Tests for splitting sections into chunks (no real file needed)."""

from documents import Section, split_into_chunks


def test_every_chunk_keeps_its_location() -> None:
    sections = [Section("Text on page two.", "page 2"), Section("Text at minute five.", "05:00")]
    chunks = split_into_chunks(sections, source="book.pdf")
    assert [(chunk.location, chunk.source) for chunk in chunks] == [
        ("page 2", "book.pdf"),
        ("05:00", "book.pdf"),
    ]


def test_long_section_is_split_into_chunks_no_longer_than_chunk_size() -> None:
    long_text = "word " * 500  # 2500 characters
    chunks = split_into_chunks([Section(long_text, "page 1")], source="a.pdf", chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 1000 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(400))
    chunks = split_into_chunks([Section(text, "page 1")], source="a.pdf", chunk_size=300, chunk_overlap=100)
    last_word_of_first = chunks[0].text.split()[-1]
    assert last_word_of_first in chunks[1].text  # the end of one chunk starts the next


def test_no_sections_gives_no_chunks() -> None:
    assert split_into_chunks([], source="empty.pdf") == []
