# Knowledge AI — Document RAG Chatbot

A document-grounded chatbot that answers questions from uploaded company documents (PDF, DOCX, TXT) using retrieval-augmented generation.

- **Frontend**: `knowledge-ai-frontend` — pure static HTML/CSS/JS, no build step.
- **Backend**: `knowledge-ai-backend` — Python FastAPI + SQLite + FAISS + fastembed (ONNX embeddings) + OpenAI/OpenRouter.

## Architecture

```
User -> Frontend (static JS)
      -> FastAPI /api/chat
      -> query normalize + intent detection (spelling-tolerant) + query expansion
      -> query embedding (fastembed / ONNX all-MiniLM-L6-v2)
      -> FAISS vector search (+ lexical intent fallback for contact/address/timings blocks)
      -> hybrid rerank (semantic + keyword + phrase + heading + intent + table + time)
      -> dedup + relevance gate -> prompt (retrieved context + history)
      -> LLM (OpenAI -> OpenRouter -> heuristic fallback)
      -> answer + sources -> Frontend renders markdown/tables + source chips
```

Document upload flow: `FormData -> UploadFile -> disk (data/uploads) -> text/table extraction (PyMuPDF find_tables + python-docx tables) -> chunking (paragraph/table/sentence aware) -> embeddings -> FAISS index + SQLite metadata`.

## Features

- Document upload/listing/deletion (PDF, DOCX, TXT; 15 MB cap)
- Table-aware extraction and answers (rows preserved, returned as Markdown tables)
- Heading/section-aware chunking and retrieval (sections boosted for their own questions)
- Query normalization + typo tolerance (`working hors` -> `working hours`) and intent detection (contact, address, working hours, leave policy, holiday, summary, …)
- Hybrid reranking (semantic + keyword + phrase + heading + intent + table + time signals)
- Document-structure summaries (one representative chunk per section) plus "not found" gating
- Near-duplicate chunk removal (kills repeated headers/footers in answers)
- Conversation memory (last 6 turns, bounded)
- Source/reference display (filename + page)
- Hallucination guard ("I couldn't find this information…")
- Clear chat without deleting documents
- Loading states and friendly error messages
- Render + Vercel deployment support

## Technology stack

| Layer        | Tech |
|--------------|------|
| Backend      | FastAPI, Uvicorn, pydantic |
| Storage      | SQLite (metadata/conversations), FAISS (vectors) |
| Embeddings   | fastembed (ONNX Runtime) `all-MiniLM-L6-v2` |
| Extraction   | PyMuPDF, python-docx |
| LLM          | OpenAI (primary), OpenRouter (fallback) |
| Frontend     | Vanilla HTML/CSS/JS (no frameworks) |

## Folder structure

```
knowledge-ai-backend/
  app/
    main.py               FastAPI app, CORS, /health
    config.py             env loading + module-relative data paths
    api_chat.py           /api/chat, history, clear chat
    api_documents.py      upload / list / delete documents
    database.py           SQLite CRUD
    services/
      document_loader.py  PDF/DOCX/TXT extraction (tables preserved)
      chunker.py          paragraph/table/sentence-aware chunking
      embeddings.py       lazy-loaded embedding model
      vector_store.py     FAISS index + metadata, delete/reindex
      ingestion.py        doc -> chunks -> embeddings -> store (batched)
      rag.py              retrieval, dedup, prompt, LLM, heuristic
  data/                   runtime data (uploads/, vector_store/, .db) — gitignored
  tests/test_backend.py   pytest suite
  requirements.txt
knowledge-ai-frontend/
  index.html, app.js, style.css, config.js
  vercel.json
render.yaml               Render blueprint
```

## Local setup

```bash
# Backend
cd knowledge-ai-backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Frontend: nothing to install
```


`config.py` resolves data paths relative to the backend folder, so the backend works from any working directory.

```bash
# Backend startup (from knowledge-ai-backend/)
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
# or
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Startup takes ~20–30 s (embedding model load). Health check: `curl http://127.0.0.1:8000/api/health`.

```bash
# Frontend startup
cd knowledge-ai-frontend
python3 -m http.server 5173
# open http://localhost:5173
```

