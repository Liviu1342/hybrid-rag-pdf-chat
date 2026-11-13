"""Load a PDF and split it into overlapping text chunks with page metadata."""

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    """A piece of text from the PDF, plus where it came from."""

    text: str
    page: int  # 1-based page number, as a human reads it
    source: str  # file name, shown later in citations


def load_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) for every page that contains text."""
    reader = PdfReader(pdf_path)
    pages: list[tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip empty pages (e.g. scanned images)
            pages.append((page_number, text))
    return pages


def split_into_chunks(
    pages: list[tuple[int, str]],
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Split each page into overlapping chunks, keeping its page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks: list[Chunk] = []
    for page_number, text in pages:
        for piece in splitter.split_text(text):
            chunks.append(Chunk(text=piece, page=page_number, source=source))
    return chunks


if __name__ == "__main__":
    import sys

    pdf_path = Path(sys.argv[1])
    pages = load_pages(pdf_path)
    chunks = split_into_chunks(pages, source=pdf_path.name)
    print(f"Pages with text: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(chunks[0])