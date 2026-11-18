"""Tests for reading plain text files into blocks of lines."""

from documents import Section
from text_loader import LINES_PER_SECTION, load_text


def test_short_text_is_one_section_with_its_line_range() -> None:
    assert load_text(b"first line\nsecond line") == [Section("first line\nsecond line", "lines 1-2")]


def test_long_text_is_split_into_blocks_of_lines() -> None:
    data = "\n".join(f"line {n}" for n in range(1, 101)).encode()  # 100 lines
    locations = [section.location for section in load_text(data)]
    assert LINES_PER_SECTION == 40
    assert locations == ["lines 1-40", "lines 41-80", "lines 81-100"]


def test_empty_file_gives_no_sections() -> None:
    assert load_text(b"\n\n  \n") == []
