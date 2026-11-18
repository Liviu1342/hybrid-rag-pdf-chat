"""Tests for reading .srt / .vtt subtitles into one-minute sections."""

from documents import Section
from subtitle_loader import format_timestamp, load_subtitles, parse_timestamp

SRT = """1
00:00:01,000 --> 00:00:04,000
Welcome to the course.

2
00:00:30,500 --> 00:00:33,000
Today we talk about <i>refunds</i>.

3
00:01:05,000 --> 00:01:08,000
Now, shipping.
"""

VTT = """WEBVTT

NOTE this comment is ignored

01:02:03.000 --> 01:02:05.000
Late in a long video.
"""


def test_timestamps_are_parsed_with_or_without_hours() -> None:
    assert parse_timestamp("00:01:02,500") == 62
    assert parse_timestamp("01:02.500") == 62


def test_timestamps_are_formatted_like_a_video_player() -> None:
    assert format_timestamp(330) == "05:30"
    assert format_timestamp(3930) == "1:05:30"


def test_srt_cues_are_grouped_per_minute_without_tags() -> None:
    assert load_subtitles(SRT.encode()) == [
        Section("Welcome to the course. Today we talk about refunds.", "00:01"),
        Section("Now, shipping.", "01:05"),
    ]


def test_vtt_header_and_notes_are_skipped() -> None:
    assert load_subtitles(VTT.encode()) == [Section("Late in a long video.", "1:02:03")]


def test_repeated_auto_caption_lines_are_kept_once() -> None:
    srt = "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n2\n00:00:02,000 --> 00:00:03,000\nhello\n"
    assert load_subtitles(srt.encode()) == [Section("hello", "00:01")]
