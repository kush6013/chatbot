import re

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return True
    return stripped.count("|") >= 2


def _normalize_section(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


_HEADING_CONTACT_RE = re.compile(
    r"(@|\bwww\.|^\s*tel:?|^\s*email:?|^\s*web:?|\+[0-9 ])",
    re.IGNORECASE,
)

_KNOWN_SECTIONS = {
    "company details",
    "contact details",
    "contact information",
    "working hours",
    "office timings",
    "leave policy",
    "time off policy",
    "holidays calendar",
    "holiday calendar",
    "work from home policy",
    "internship policy",
    "internship certificate",
    "general policy",
    "resignation policy",
    "termination policy",
    "non disclosure policy",
    "punctuality policy",
    "employee behaviour policy",
    "reference release form",
    "code of conduct during meetings",
    "major offenses",
    "cell phone company policy",
}


def _looks_like_heading(line: str, next_line: str = None) -> bool:
    """True when `line` reads like a section heading.

    Conservative on purpose: title-cased, short, not a bullet, not a
    contact/table line, no sentence punctuation. A heading must be
    followed by real content (not another short title-cased line which
    usually means we are inside a data list such as a holiday table).
    """
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _is_table_line(s):
        return False
    if s.startswith(("•", "●", "-", "–", "·")) or re.match(r"^\d+\.\s", s):
        return False
    if s.endswith((".", ",", ":", ";", "-")):
        return False
    if _HEADING_CONTACT_RE.search(s):
        return False
    words = s.split()
    if len(words) > 8:
        return False
    if len(words) == 1 and len(s) <= 5:
        return False

    if _normalize_section(s) in _KNOWN_SECTIONS:
        return True

    if len(words) < 2:
        return False

    uppercase = sum(1 for word in words if word[:1].isupper())
    if uppercase < max(1, (len(words) + 1) // 2):
        return False

    if next_line:
        n = next_line.strip()
        next_is_content = (
            len(n) >= 40
            or n.startswith(("•", "●", "-", "–", "·"))
            or len(n.split()) > 8
        )
        if not next_is_content:
            return False

    return True


def _head_tail_split_lines(lines):
    """Split a paragraph's first title-cased line off as a heading.

    Returns (heading_or_None, joined_remaining_text).
    """
    if not lines:
        return None, ""

    first = lines[0]
    if _looks_like_heading(first, lines[1] if len(lines) > 1 else None):
        remaining = " ".join(lines[1:]).strip()
        return first.strip(), remaining

    return None, " ".join(lines).strip()


def _split_into_blocks(text: str):
    """Split text into heading, table and paragraph blocks.

    A table block is a run of consecutive pipe-delimited rows; it is
    kept as one block so rows (Date | Festival | Day) stay together.
    A title-cased line at the start of a paragraph (optionally followed
    directly by bullets/content) is flagged as a heading block so
    retrieval can use it as a section signal.
    """
    blocks = []
    current_table = []
    current_paragraph = []

    def flush_paragraph():
        heading, remaining = _head_tail_split_lines(current_paragraph)
        if heading:
            blocks.append(("heading", heading))
        if remaining:
            blocks.append(("paragraph", remaining))
        current_paragraph.clear()

    def flush_table():
        if current_table:
            table_text = "\n".join(current_table).strip()
            if table_text:
                blocks.append(("table", table_text))
        current_table.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_table()
            continue
        if _is_table_line(line):
            flush_paragraph()
            current_table.append(line)
        else:
            flush_table()
            current_paragraph.append(line)

    flush_paragraph()
    flush_table()

    return blocks


def _split_sentences(paragraph: str):
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [part.strip() for part in parts if part.strip()]


def _nearest_boundary(text: str, target: int, min_boundary: int):
    """Find the last sentence/whitespace near `target` to cut at."""
    boundary = -1
    for match in re.finditer(r"[.!?]\s", text):
        if match.start() + 1 > target:
            break
        if match.start() + 1 >= min_boundary:
            boundary = match.start() + 1
    if boundary == -1:
        boundary = text.rfind(" ", min_boundary, target)
    if boundary == -1:
        boundary = target
    return boundary


def _chunk_long_paragraph(paragraph: str, chunk_size: int, overlap: int):
    sentences = _split_sentences(paragraph)
    if not sentences:
        return []

    chunks = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= chunk_size:
            current = sentence
            continue
        for start in range(0, len(sentence), chunk_size - overlap):
            chunk = sentence[start:start + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        current = ""
    if current:
        chunks.append(current)

    return chunks


def _chunk_table(table_text: str, chunk_size: int):
    if len(table_text) <= chunk_size:
        return [table_text]

    rows = table_text.splitlines()
    header = rows[0] if rows else ""

    chunks = []
    current = ""
    for row in rows:
        candidate = f"{current}\n{row}".strip() if current else row
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(row) <= chunk_size:
            current = row
            continue
        for start in range(0, len(row), chunk_size):
            piece = row[start:start + chunk_size].strip()
            if piece:
                chunks.append(piece)
        current = header if row != header else ""
    if current:
        chunks.append(current)

    return chunks


def _chunk_blocks(blocks, chunk_size: int, overlap: int):
    current_heading = ""
    out = []

    for kind, block in blocks:
        if kind == "heading":
            current_heading = block
            continue

        if kind == "table":
            texts = _chunk_table(block, chunk_size)
        else:
            texts = _chunk_long_paragraph(block, chunk_size, overlap)

        for text in texts:
            prefix = f"{current_heading}\n\n" if current_heading else ""
            out.append({
                "text": prefix + text,
                "type": kind,
                "heading": current_heading
            })

    return out


def chunk_text_with_meta(text: str):
    """Heading- and type-aware chunking returning metadata dicts.

    Each returned dict:
        {"text": str, "type": "table"|"paragraph", "heading": str}
    The heading is kept both as metadata and as a text prefix so the
    section label participates in embedding/keyword retrieval.
    """
    text = re.sub(r"[ \t]+", " ", text)

    if not text.strip():
        return []

    chunk_size = max(50, int(CHUNK_SIZE))
    overlap = max(0, min(int(CHUNK_OVERLAP), chunk_size - 1))

    return [
        chunk for chunk in _chunk_blocks(
            _split_into_blocks(text),
            chunk_size,
            overlap
        )
        if chunk["text"].strip()
    ]


def chunk_text(text: str):
    return [
        chunk["text"]
        for chunk in chunk_text_with_meta(text)
    ]