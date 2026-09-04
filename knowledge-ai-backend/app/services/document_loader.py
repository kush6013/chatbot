from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    import fitz


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt"
}


def load_pdf(file_path: str):
    pages = []
    try:
        pdf = fitz.open(file_path)
    except Exception as err:
        raise ValueError(f"Failed to open PDF file: {str(err)}")

    try:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
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
        from docx import Document
    except ImportError:
        raise ValueError(
            "python-docx is not installed. "
            "Add 'python-docx' to requirements.txt."
        )
    try:
        document = Document(file_path)
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


def load_txt(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
    except Exception as err:
        raise ValueError(f"Failed to read TXT file: {str(err)}")

    if not text:
        return []

    return [
        {
            "text": text,
            "page": None
        }
    ]


def load_document(file_path: str):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    raise ValueError("Only PDF, DOCX, and TXT files are supported.")
