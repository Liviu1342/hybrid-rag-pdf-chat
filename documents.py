"""Source-independent building blocks: sections of a document and their chunks."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Section:
    """A piece of a source with one location: a PDF page, a minute of video, a chapter."""

    text: str
    location: str  # human-readable, e.g. "page 12", "05:30", "chapter 3"


@dataclass(frozen=True)
class Chunk:
    """A piece of text small enough to embed, plus where it came from."""

    text: str
    location: str  # copied from its section, shown in citations
    source: str  # file name (or video title), shown in citations


PARAGRAPHS_PER_SECTION = 20  # a long chapter without sub-headings is still split


def group_paragraphs(
    paragraphs: list[tuple[str, bool]], first_heading: str = ""
) -> list[Section]:
    """Group (text, is_heading) paragraphs into sections, located as "Refunds, paragraph 12".

    Used for formats without fixed pages (Word, EPUB): the page count changes
    with the font size, so we cite the nearest heading plus the paragraph number.
    A new section starts at every heading and after PARAGRAPHS_PER_SECTION paragraphs.
    """
    sections: list[Section] = []
    heading = first_heading  # the last heading seen
    texts: list[str] = []
    first_paragraph = 0

    def flush() -> None:
        """Turn the collected paragraphs into one section."""
        if texts:
            where = f"paragraph {first_paragraph}"
            location = f"{heading}, {where}" if heading else where
            sections.append(Section("\n".join(texts), location))
            texts.clear()

    for number, (text, is_heading) in enumerate(paragraphs, start=1):
        if is_heading:
            flush()  # a new heading starts a new section
            heading = text
        elif len(texts) >= PARAGRAPHS_PER_SECTION:
            flush()
        if not texts:
            first_paragraph = number
        texts.append(text)
    flush()
    return sections


def split_into_chunks(
    sections: list[Section],
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split each section into overlapping chunks, keeping its location."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Chunk] = []
    for section in sections:
        for piece in splitter.split_text(section.text):
            chunks.append(Chunk(text=piece, location=section.location, source=source))
    return chunks