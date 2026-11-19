"""Tests for detecting the file type and choosing the loader."""

import pytest

from documents import Section
from loaders import SUPPORTED_EXTENSIONS, UnsupportedFileError, load_document


def test_loader_is_chosen_by_extension_ignoring_case() -> None:
    srt = b"1\n00:00:05,000 --> 00:00:06,000\nHello\n"
    assert load_document("talk.SRT", srt) == [Section("Hello", "00:05")]
    assert load_document("notes.txt", b"Hello") == [Section("Hello", "lines 1-1")]


def test_uploader_list_contains_every_supported_type() -> None:
    assert SUPPORTED_EXTENSIONS == ["pdf", "srt", "vtt", "txt", "docx", "pptx"]


def test_unknown_extension_is_refused() -> None:
    with pytest.raises(UnsupportedFileError, match="not supported"):
        load_document("virus.exe", b"MZ...")


def test_fake_pdf_is_refused() -> None:
    with pytest.raises(UnsupportedFileError, match="does not look like a real .pdf"):
        load_document("invoice.pdf", b"MZ this is really a program")


def test_fake_word_file_is_refused() -> None:
    with pytest.raises(UnsupportedFileError, match="does not look like a real .docx"):
        load_document("report.docx", b"%PDF-1.7 a PDF renamed to .docx")


def test_binary_file_renamed_to_txt_is_refused() -> None:
    with pytest.raises(UnsupportedFileError, match="binary"):
        load_document("photo.txt", b"\x89PNG\r\n\x00\x00")