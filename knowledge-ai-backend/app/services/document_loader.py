from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from docx import Document


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx"
}


def load_pdf(file_path: str):

    pages = []

    try:
        pdf = fitz.open(file_path)
    except Exception as err:
        raise ValueError(f"Failed to open PDF file: {str(err)}")

    try:

        for page_number, page in enumerate(
            pdf,
            start=1
        ):

            text = page.get_text(
                "text"
            ).strip()

            if text:

                pages.append({
                    "text": text,
                    "page": page_number
                })

    finally:

        pdf.close()

    return pages


def load_docx(file_path: str):

    try:
        document = Document(
            file_path
        )
    except Exception as err:
        raise ValueError(f"Failed to open DOCX file: {str(err)}")

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)

    text = "\n".join(paragraphs)

    if not text:

        return []

    return [
        {
            "text": text,
            "page": None
        }
    ]


def load_document(file_path: str):

    extension = Path(
        file_path
    ).suffix.lower()

    if extension == ".pdf":

        return load_pdf(
            file_path
        )

    if extension == ".docx":

        return load_docx(
            file_path
        )

    raise ValueError(
        "Only PDF and DOCX files are supported."
    )

