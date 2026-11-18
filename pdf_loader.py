"""Load the text of a PDF, one section per page."""

from io import BytesIO

from pypdf import PdfReader

from documents import Section


def load_pdf(data: bytes) -> list[Section]:
    """Return one section per page that contains text, located as "page N"."""
    reader = PdfReader(BytesIO(data))  # read from memory, no file path needed
    sections: list[Section] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip empty pages (e.g. scanned images)
            sections.append(Section(text=text, location=f"page {page_number}"))
    return sections


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from documents import split_into_chunks

    pdf_path = Path(sys.argv[1])
    sections = load_pdf(pdf_path.read_bytes())
    chunks = split_into_chunks(sections, source=pdf_path.name)
    print(f"Pages with text: {len(sections)}")
    print(f"Chunks: {len(chunks)}")
    print(chunks[0])
