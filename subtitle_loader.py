"""Load .srt / .vtt subtitles, one section per minute of video."""

import re

from documents import Section

WINDOW_SECONDS = 60  # one section per minute: precise citations, enough context

# "00:01:02,500" (.srt) or "01:02.500" (.vtt, hours optional)
TIMESTAMP = re.compile(r"(?:(\d+):)?(\d{2}):(\d{2})[,.]\d{3}")
TAG = re.compile(r"<[^>]+>")  # formatting tags such as <i> or <c.colorE5E5E5>


def parse_timestamp(text: str) -> int:
    """Turn "00:01:02,500" or "01:02.500" into whole seconds (62)."""
    match = TIMESTAMP.search(text)
    if match is None:
        raise ValueError(f"Not a subtitle timestamp: {text!r}")
    hours, minutes, seconds = match.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)


def format_timestamp(total_seconds: int) -> str:
    """Turn 330 into "05:30", and 3930 into "1:05:30"."""
    hours, rest = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_cues(text: str) -> list[tuple[int, str]]:
    """Return (start_second, text) for every subtitle cue."""
    cues: list[tuple[int, str]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = block.strip().split("\n")
        # The timing line looks like "00:00:01,000 --> 00:00:04,000"
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue  # the WEBVTT header, a NOTE, or a stray number
        start = parse_timestamp(lines[timing_index].split("-->")[0])
        words = " ".join(TAG.sub("", line).strip() for line in lines[timing_index + 1 :])
        if words.strip():
            cues.append((start, words.strip()))
    return cues


def group_into_windows(cues: list[tuple[int, str]]) -> list[Section]:
    """Join consecutive cues into sections of about WINDOW_SECONDS each."""
    sections: list[Section] = []
    window_start: int | None = None
    texts: list[str] = []
    for start, text in cues:
        if window_start is not None and start - window_start >= WINDOW_SECONDS:
            sections.append(Section(" ".join(texts), format_timestamp(window_start)))
            window_start, texts = None, []
        if window_start is None:
            window_start = start
        if not texts or texts[-1] != text:  # auto-captions often repeat a line
            texts.append(text)
    if window_start is not None:
        sections.append(Section(" ".join(texts), format_timestamp(window_start)))
    return sections


def load_subtitles(data: bytes) -> list[Section]:
    """Read .srt or .vtt bytes and return one section per minute of speech."""
    text = data.decode("utf-8-sig", errors="replace")  # "-sig" drops a BOM if present
    return group_into_windows(parse_cues(text))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    for section in load_subtitles(Path(sys.argv[1]).read_bytes())[:3]:
        print(f"[{section.location}] {section.text[:120]}")
