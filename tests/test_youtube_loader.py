"""Tests for the YouTube loader, with a fake transcript API (no internet needed)."""

import pytest
from youtube_transcript_api import FetchedTranscriptSnippet, TranscriptsDisabled

import youtube_loader
from documents import Section
from youtube_loader import TranscriptError, extract_video_id, load_youtube


class FakeTranscriptApi:
    """Stands in for YouTubeTranscriptApi and returns three timed lines."""

    def fetch(self, video_id: str, languages: tuple[str, ...]) -> list[FetchedTranscriptSnippet]:
        return [
            FetchedTranscriptSnippet(text="[Music]", start=0.0, duration=2.0),
            FetchedTranscriptSnippet(text="Hello and welcome.", start=2.4, duration=3.0),
            FetchedTranscriptSnippet(text="Refunds take 30 days.", start=75.9, duration=3.0),
        ]


class DisabledTranscriptApi:
    """A video whose owner turned subtitles off."""

    def fetch(self, video_id: str, languages: tuple[str, ...]) -> list[FetchedTranscriptSnippet]:
        raise TranscriptsDisabled(video_id)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s",
        "https://youtu.be/dQw4w9WgXcQ?si=abc",
        "youtube.com/shorts/dQw4w9WgXcQ",
    ],
)
def test_video_id_is_found_in_common_links(url: str) -> None:
    assert extract_video_id(url) == "dQw4w9WgXcQ"


def test_other_links_are_rejected() -> None:
    assert extract_video_id("https://example.com/watch?v=dQw4w9WgXcQ") is None


def test_transcript_keeps_timestamps_and_drops_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(youtube_loader, "YouTubeTranscriptApi", FakeTranscriptApi)
    assert load_youtube("dQw4w9WgXcQ") == [
        Section("Hello and welcome.", "00:02"),
        Section("Refunds take 30 days.", "01:15"),
    ]


def test_disabled_subtitles_give_a_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(youtube_loader, "YouTubeTranscriptApi", DisabledTranscriptApi)
    with pytest.raises(TranscriptError, match="disabled"):
        load_youtube("dQw4w9WgXcQ")