"""Tests for turning HTML into paragraphs."""

from html_text import SKIPPED_TAGS, html_to_paragraphs


def test_visible_text_is_kept_and_headings_are_marked() -> None:
    html = "<head><title>Ignored</title></head><h1>Refunds</h1><p>Within <b>30</b>\n days.</p>"
    assert html_to_paragraphs(html) == [("Refunds", True), ("Within 30 days.", False)]


def test_extra_tags_can_be_skipped() -> None:
    html = "<nav><a>Home</a> <a>Contact</a></nav><p>The article.</p>"
    assert html_to_paragraphs(html, SKIPPED_TAGS | {"nav"}) == [("The article.", False)]