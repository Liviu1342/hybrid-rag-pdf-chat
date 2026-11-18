"""Load a YouTube transcript with timestamps, one section per minute of video.

extract_video_id and clean_transcript come from Project 1 (YouTube summarizer).
"""

import re
from urllib.parse import parse_qs, urlparse

from requests.exceptions import RequestException  # installed together with the library
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from documents import Section
from subtitle_loader import group_into_windows

# A YouTube video ID is always exactly 11 characters: letters, digits, "-" and "_"
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
# Matches subtitle annotations such as [Music], [Applause] or [♪♪♪]
ANNOTATION_PATTERN = re.compile(r"\[[^\]]*\]")


class TranscriptError(Exception):
    """Error with a user-friendly message that can be shown directly in the UI."""


def extract_video_id(url: str) -> str | None:
    """Return the 11-character video ID from a YouTube link, or None if it is not one."""
    url = url.strip()
    if "://" not in url:  # also accept "youtube.com/..." without https
        url = "https://" + url
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    video_id = None
    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif host == "youtube.com":
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith(("/shorts/", "/embed/", "/live/")):
            video_id = parsed.path.split("/")[2]
    if video_id and VIDEO_ID_PATTERN.match(video_id):
        return video_id
    return None


def clean_transcript(text: str) -> str:
    """Remove subtitle noise (annotations, music notes, extra spaces)."""
    text = ANNOTATION_PATTERN.sub(" ", text)
    text = text.replace("♪", " ")
    return " ".join(text.split())


def fetch_cues(video_id: str, languages: tuple[str, ...] = ("ro", "en")) -> list[tuple[int, str]]:
    """Download the subtitles and return (start_second, text) for every line.

    Unlike Project 1, the timestamps are kept: they become the citations.
    """
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    except TranscriptsDisabled:
        raise TranscriptError("Subtitles are disabled for this video.") from None
    except NoTranscriptFound:
        raise TranscriptError(f"No subtitles found in: {', '.join(languages)}.") from None
    except VideoUnavailable:
        raise TranscriptError("The video does not exist or is private.") from None
    except CouldNotRetrieveTranscript as error:
        raise TranscriptError(f"YouTube refused the request ({type(error).__name__}).") from error
    except RequestException as error:
        raise TranscriptError("Could not connect to YouTube. Check your internet connection.") from error

    cues: list[tuple[int, str]] = []
    for snippet in fetched:
        text = clean_transcript(snippet.text)
        if text:  # a line that was only "[Music]" is now empty
            cues.append((int(snippet.start), text))
    return cues


def load_youtube(video_id: str) -> list[Section]:
    """Return one section per minute of the video, located as "05:30"."""
    return group_into_windows(fetch_cues(video_id))


if __name__ == "__main__":
    import sys

    found_id = extract_video_id(sys.argv[1])
    if found_id is None:
        sys.exit("Not a YouTube link.")
    for section in load_youtube(found_id)[:3]:
        print(f"[{section.location}] {section.text[:120]}")