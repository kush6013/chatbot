import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.3-70b-instruct:free"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "800")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "120")
)

TOP_K = int(
    os.getenv("TOP_K", "5")
)

SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SIMILARITY_THRESHOLD",
        "0.15"
    )
)

FETCH_K = int(
    os.getenv("FETCH_K", "40")
)

SUMMARY_FETCH_K = int(
    os.getenv("SUMMARY_FETCH_K", "24")
)

MIN_FINAL_SCORE = float(
    os.getenv("MIN_FINAL_SCORE", "0.75")
)

SEMANTIC_FLOOR = float(
    os.getenv("SEMANTIC_FLOOR", "0.30")
)

GENERAL_SEMANTIC_FLOOR = float(
    os.getenv("GENERAL_SEMANTIC_FLOOR", "0.34")
)

RAG_DEBUG = os.getenv(
    "RAG_DEBUG",
    "false"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on"
}

MAX_UPLOAD_SIZE = int(
    os.getenv(
        "MAX_UPLOAD_SIZE",
        str(15 * 1024 * 1024)
    )
)

PORT = int(
    os.getenv("PORT", "8000")
)


UPLOAD_DIR = BASE_DIR / "data" / "uploads"

VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

DATABASE_PATH = BASE_DIR / "data" / "knowledge_ai.db"


def ensure_data_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


def document_id_for(filename):
    """Stable internal document identifier derived from the filename.

    Used as the canonical document key across the database, the vector
    store metadata and retrieval results (the filename alone is not a
    reliable identifier — uploads can be renamed or re-uploaded).
    """
    return hashlib.sha1(
        str(filename).encode("utf-8")
    ).hexdigest()[:16]