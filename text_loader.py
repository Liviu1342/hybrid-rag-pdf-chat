"""Load plain .txt files, one section per block of lines."""

from documents import Section

LINES_PER_SECTION = 40  # roughly one printed page


def load_text(data: bytes) -> list[Section]:
    """Return one section per LINES_PER_SECTION lines, located as "lines 1-40"."""
    lines = data.decode("utf-8-sig", errors="replace").splitlines()
    sections: list[Section] = []
    for start in range(0, len(lines), LINES_PER_SECTION):
        block = lines[start : start + LINES_PER_SECTION]
        text = "\n".join(block)
        if text.strip():  # skip blocks that are only empty lines
            end = start + len(block)
            sections.append(Section(text, f"lines {start + 1}-{end}"))
    return sections
