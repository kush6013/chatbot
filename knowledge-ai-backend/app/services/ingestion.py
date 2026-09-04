import uuid

from pathlib import Path

from app.services.document_loader import (
    load_document
)

from app.services.chunker import (
    chunk_text
)

from app.services.embeddings import (
    embedding_service
)

from app.services.vector_store import (
    vector_store
)


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

        chunks = chunk_text(
            page_data["text"]
        )

        for chunk in chunks:

            all_chunks.append(
                chunk
            )

            metadata.append({

                "id": str(
                    uuid.uuid4()
                ),

                "source": filename,

                "page": page_data[
                    "page"
                ],

                "text": chunk

            })

    if not all_chunks:

        raise ValueError(
            "No readable text found "
            "in the document."
        )

    embeddings = (
        embedding_service.generate(
            all_chunks
        )
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
