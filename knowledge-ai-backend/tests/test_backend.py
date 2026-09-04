import os
import shutil
import tempfile
from fastapi.testclient import TestClient


from app.database import (
    initialize_database,
    add_document,
    list_documents,
    delete_document,
    create_conversation,
    add_message,
    get_messages,
    get_conversations,
    delete_conversation
)
from app.services.chunker import chunk_text
from app.services.vector_store import VectorStore
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "Knowledge AI"}


def test_chunker_basic():
    text = "Hello world! " * 100
    chunks = chunk_text(text)
    assert len(chunks) > 0
    assert isinstance(chunks, list)


def test_chunker_overlap_guard():
    text = "Testing sentence overlapping logic."
    chunks = chunk_text(text)
    assert len(chunks) >= 1


def test_database_crud():
    initialize_database()
    add_document("test_doc.pdf", 1024, "pdf")
    docs = list_documents()
    assert any(d["filename"] == "test_doc.pdf" for d in docs)
    delete_document("test_doc.pdf")
    docs_after = list_documents()
    assert not any(d["filename"] == "test_doc.pdf" for d in docs_after)


def test_conversation_db():
    cid = "test-conv-123"
    create_conversation(cid, "Test Title")
    add_message(cid, "user", "Hello")
    add_message(cid, "assistant", "Hi there!")
    msgs = get_messages(cid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    convs = get_conversations()
    assert any(c["conversation_id"] == cid for c in convs)
    delete_conversation(cid)
    assert len(get_messages(cid)) == 0


def test_vector_store_delete_source():
    store = VectorStore()
    mock_embeddings = [[0.1] * 384, [0.2] * 384]
    mock_meta = [
        {"id": "1", "source": "docA.pdf", "page": 1, "text": "sample text 1"},
        {"id": "2", "source": "docB.pdf", "page": 1, "text": "sample text 2"}
    ]
    store.add(mock_embeddings, mock_meta)
    assert len(store.metadata) >= 2
    store.delete_source("docA.pdf")
    sources = [m["source"] for m in store.metadata]
    assert "docA.pdf" not in sources
