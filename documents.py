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
