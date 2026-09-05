import logging
import os
import shutil

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.config import (
    UPLOAD_DIR,
    MAX_UPLOAD_SIZE
)

from app.database import (
    add_document,
    list_documents,
    delete_document,
    reconcile_documents
)

from app.services.ingestion import (
    process_document
)

logger = logging.getLogger("knowledge_ai")


router = APIRouter(
    prefix="/api/documents"
)


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}


@router.get("/")
def get_documents():

    reconcile_documents()

    documents = list_documents()

    return {
        "documents": [
            {
                "document_id": item["document_id"],
                "filename": item["filename"],
                "size": item["size"],
                "type": item["file_type"],
                "uploaded_at": item[
                    "uploaded_at"
                ]
            }
            for item in documents
        ]
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Only PDF, DOCX, and TXT "
                "documents are supported."
            )
        )

    filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    size = 0
    try:
        with open(file_path, "wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "File is too large. "
                            "Maximum allowed size is "
                            f"{MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
                        )
                    )
                output.write(chunk)
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as error:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error("Upload failed for %s: %s", filename, error)
        raise HTTPException(
            status_code=500,
            detail="Failed to save the uploaded file."
        )

    if size == 0:
        os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    existing = [
        item for item in list_documents()
        if item["filename"] == filename
    ]

    if existing:
        from app.services.vector_store import vector_store
        vector_store.delete_source(filename)

    try:
        result = process_document(file_path)
    except Exception as error:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error("Document processing failed for %s: %s", filename, error)
        raise HTTPException(
            status_code=500,
            detail=(
                "The document could not be processed. "
                "It may be corrupt, empty, or in an unsupported layout. "
                "Details are logged on the server."
            )
        )

    add_document(
        filename,
        size,
        extension.replace(".", "")
    )

    return {
        "success": True,
        "message": (
            "Document uploaded "
            "and indexed successfully."
        ),
        "filename": filename,
        "chunks": result["chunks"]
    }


@router.delete("/{filename}")
def remove_document(
    filename: str
):

    filename = os.path.basename(
        filename
    )

    from app.services.vector_store import (
        vector_store
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    file_exists = os.path.exists(
        file_path
    )

    rows = [
        item for item in list_documents()
        if item["filename"] == filename
        or item["document_id"] == filename
    ]

    sources = {
        item.get("source")
        for item in vector_store.all_metadata()
    }

    if not file_exists and not rows and filename not in sources:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    if file_exists:
        os.remove(file_path)

    if rows:
        delete_document(filename)

    vector_store.delete_source(
        filename
    )

    return {
        "success": True,
        "message": "Document deleted."
    }