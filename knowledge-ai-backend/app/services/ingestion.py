import uuid

from pathlib import Path

from app.config import document_id_for

from app.services.document_loader import (
    load_document
)

from app.services.chunker import (
    chunk_text_with_meta
)

from app.services.embeddings import (
    embedding_service
)

from app.services.vector_store import (
    vector_store
)

EMBED_BATCH_SIZE = 64


def _embed_in_batches(texts):
    embeddings = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        batch_embeddings = embedding_service.generate(batch)
        embeddings.extend(batch_embeddings)
    return embeddings


def process_document(
    file_path: str
):

    filename = Path(
        file_path
    ).name

    pages = load_document(
        file_path
    )

    all_chunks = []

    metadata = []

    for page_data in pages:

        chunks = chunk_text_with_meta(
            page_data["text"]
        )

        for chunk in chunks:

            all_chunks.append(
                chunk["text"]
            )

            metadata.append({

                "id": str(
                    uuid.uuid4()
                ),

                "document_id": document_id_for(
                    filename
                ),

                "source": filename,

                "page": page_data[
                    "page"
                ],

                "text": chunk["text"],

                "type": chunk["type"],

                "heading": chunk["heading"]

            })

    if not all_chunks:

        raise ValueError(
            "No readable text found "
            "in the document."
        )

    embeddings = _embed_in_batches(
        all_chunks
    )

    vector_store.add(
        embeddings,
        metadata
    )

    return {
        "filename": filename,
        "chunks": len(
            all_chunks
        )
    }