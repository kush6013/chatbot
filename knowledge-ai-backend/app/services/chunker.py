from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def chunk_text(text: str):

    text = " ".join(
        text.split()
    )

    if not text:
        return []

    chunks = []

    start = 0

    text_length = len(text)

    chunk_size = max(50, CHUNK_SIZE)
    overlap = max(0, min(CHUNK_OVERLAP, chunk_size - 1))

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        if end >= text_length:

            break

        start = end - overlap

    return chunks

