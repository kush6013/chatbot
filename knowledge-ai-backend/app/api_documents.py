import os
import shutil

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from app.config import UPLOAD_DIR

from app.database import (
    add_document,
    list_documents,
    delete_document
)

from app.services.ingestion import (
    process_document
)


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

    documents = list_documents()

    return {
        "documents": [
            {
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
            status_code=400,
            detail=(
                "Only PDF, DOCX, and TXT "
                "documents are supported."
            )
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    filename = os.path.basename(
        file.filename
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    try:

        with open(
            file_path,
            "wb"
        ) as output:

            shutil.copyfileobj(
                file.file,
                output
            )

        size = os.path.getsize(
            file_path
        )

        result = process_document(
            file_path
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

    except Exception as error:

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


from app.services.vector_store import (
    vector_store
)


@router.delete("/{filename}")
def remove_document(
    filename: str
):

    filename = os.path.basename(
        filename
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    if not os.path.exists(
        file_path
    ):

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    os.remove(
        file_path
    )

    delete_document(
        filename
    )

    vector_store.delete_source(
        filename
    )

    return {
        "success": True,
        "message": "Document deleted."
    }

