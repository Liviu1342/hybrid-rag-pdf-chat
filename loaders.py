"""Pick the right loader for an uploaded file, based on its extension and content."""

from collections.abc import Callable
from pathlib import Path

from documents import Section
from office_loader import load_docx, load_pptx
from pdf_loader import load_pdf
from subtitle_loader import load_subtitles
from text_loader import load_text

# Every loader has the same shape: bytes in, sections out
Loader = Callable[[bytes], list[Section]]

LOADERS: dict[str, Loader] = {
    ".pdf": load_pdf,
    ".srt": load_subtitles,
    ".vtt": load_subtitles,
    ".txt": load_text,
    ".docx": load_docx,
    ".pptx": load_pptx,
}

SUPPORTED_EXTENSIONS = [extension.lstrip(".") for extension in LOADERS]  # for the uploader

# "Magic bytes": how the content of a real file of this type starts
SIGNATURES: dict[str, bytes] = {
    ".pdf": b"%PDF",
    ".docx": b"PK\x03\x04",  # Office files are ZIP archives of XML files
    ".pptx": b"PK\x03\x04",
}
TEXT_EXTENSIONS = {".srt", ".vtt", ".txt"}


class UnsupportedFileError(Exception):
    """Raised when a file type is unknown or its content does not match its extension."""


def detect_extension(name: str, data: bytes) -> str:
    """Return the file's extension, after checking that the content matches it."""
    extension = Path(name).suffix.lower()  # "Notes.TXT" -> ".txt"
    if extension not in LOADERS:
        supported = ", ".join(LOADERS)
        raise UnsupportedFileError(f"'{extension or name}' files are not supported. Use: {supported}.")
    signature = SIGNATURES.get(extension)
    if signature is not None and not data.startswith(signature):
        raise UnsupportedFileError(f"'{name}' does not look like a real {extension} file.")
    if extension in TEXT_EXTENSIONS and b"\x00" in data[:1024]:
        raise UnsupportedFileError(f"'{name}' looks like a binary file, not text.")
    return extension


def load_document(name: str, data: bytes) -> list[Section]:
    """Detect the file type and load it with the matching loader."""
    return LOADERS[detect_extension(name, data)](data)