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

### Environment variables (.env in knowledge-ai-backend/)

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `OPENAI_API_KEY` | yes* | — | OpenAI key. Without it the heuristic answerer is used. |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | |
| `OPENROUTER_API_KEY` | no | — | Optional LLM fallback |
| `OPENROUTER_MODEL` | no | `meta-llama/llama-3.3-70b-instruct:free` | |
| `EMBEDDING_MODEL` | no | `all-MiniLM-L6-v2` | |
| `CHUNK_SIZE` | no | `800` | |
| `CHUNK_OVERLAP` | no | `120` | |
| `TOP_K` | no | `5` | |
| `FETCH_K` | no | `40` | Semantic candidates fetched before hybrid rerank |
| `MIN_FINAL_SCORE` | no | `0.75` | Hybrid acceptance bar for specific-intent questions |
| `SEMANTIC_FLOOR` | no | `0.30` | Minimum embedding similarity for specific-intent chunks |
| `GENERAL_SEMANTIC_FLOOR` | no | `0.34` | Similarity floor for general questions (also need keyword evidence) |
| `RAG_DEBUG` | no | `false` | Enables `POST /api/chat/debug` (retrieval transparency helper) |
| `SIMILARITY_THRESHOLD` | no | `0.15` | Legacy low bar; superseded by `MIN_FINAL_SCORE`/floors |
| `MAX_UPLOAD_SIZE` | no | `15728640` | bytes (15 MB) |
| `PORT` | no | `8000` | Render injects this |

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

The frontend uses `http://127.0.0.1:8000` by default (`config.js` left empty). Set `window.RAG_API_BASE` in `config.js` to point at a deployed backend.

## Document upload process

1. Frontend sends `multipart/form-data` field `file` to `POST /api/documents/upload`.
2. Backend streams the file to `data/uploads` (bounded by `MAX_UPLOAD_SIZE`).
3. `load_document` extracts text; PDF and DOCX tables are detected and serialized as pipe-delimited rows so `Date | Festival | Day` relationships survive.
4. `chunk_text` groups paragraphs, keeps table blocks whole, splits long paragraphs at sentence boundaries.
5. Embeddings are generated in small batches and stored in FAISS with filename/page metadata.
6. Re-uploading the same filename replaces (not duplicates) the old source vectors.

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health`, `/api/health` | Liveness |
| POST | `/api/chat` | Ask a question (`message`, `conversation_id`, `language`, `model`) |
| POST | `/api/chat/debug` | Retrieval transparency (per-chunk scores); only when `RAG_DEBUG=true`, else 404 |
| GET | `/api/chat/history` | List conversations |
| GET | `/api/chat/history/{id}` | Conversation messages |
| DELETE | `/api/chat/{id}` | Clear a conversation (does not delete documents) |
| GET | `/api/documents/` | List uploaded documents |
| POST | `/api/documents/upload` | Upload + index |
| DELETE | `/api/documents/{filename}` | Delete document + vectors |

## RAG pipeline

1. Pure conversational/small-talk turns (greetings, "how are you", thanks, goodbyes) are answered by a lightweight router — no retrieval, no sources.
2. Normalize the question (typo map, plural handling) and detect intents (contact, address, company info, working hours, leave policy, holidays, document summary, …). Bare "company detail" implies address + contact.
3. Document-aware routing: if the question names a file (exact filename, extension-less name, or a single distinctive word like `holidays`/`policies`), retrieval is restricted to that document. "Summarize this document" uses the most recently referenced file in the conversation; with several documents and no reference it asks which file to summarize instead of guessing.
4. Expand the query with canonical intent terms, then embed it with the same model used at ingestion.
5. FAISS search fetches `FETCH_K` candidates, plus a lexical intent fallback that pulls letterhead/contact/address chunks even when they rank past the embedding window.
6. Each chunk is hybrid-scored: `semantic + keyword + phrase + heading + intent + table + time − duplicate`, with per-intent pattern boosts (Tel/+/@/www for contact, S7B/road/PIN for address, `10:00 am` for timings). A "medium gate" (email/phone/website/address named in the question) promotes the chunk that has it and demotes chunks that clearly don't.
7. Acceptance gate: specific-intent questions need `MIN_FINAL_SCORE` or a strong pattern-backed intent plus `SEMANTIC_FLOOR`; general questions need `GENERAL_SEMANTIC_FLOOR` plus a keyword hit. Anything below returns a clear "not found".
8. Near-duplicate chunks are removed (word-containment overlap, so genuinely different content survives).
9. Context is built from deduped chunks, capped at 5000 chars; the last 6 conversation messages are included for follow-ups.
10. `document_summary` questions take a structural path: one representative chunk per (source, heading) so summaries span the whole document rather than top-k fragments. Boilerplate/letterhead lines are stripped so summaries do not repeat a contact block on every page.
11. LLM answers only from context (Markdown table for tabular data, bullets for lists, short paragraphs otherwise) and refuses when the answer is not present.
12. If no LLM is configured/fails, a heuristic answers from the top chunks with the same not-found behavior. Line selection is chunk-order-aware and pattern-aware (contact/address/timing lines verify against the query intent, not a bare embedding cosine), skips blank form fields, and trims trailing document clauses off letterhead lines.
13. Sources are the chunks the answer actually used (deduped) and carry a stable `document_id`; TXT sources have `page: null` (no fake "p. ?" in the UI).

## Testing

```bash
cd knowledge-ai-backend
./.venv/bin/python -m pytest tests/ -v
```

`test_vector_store_delete_source` uses a temporary directory (no pollution of dev data). The DB and uploads can still be touched by endpoint tests; the suite cleans up its own rows/files. The retrieval tests auto-skip when the dev fixture (`data/uploads/Company_Policies_25-26.pdf`) is absent, and index both the PDF and `data/uploads/company_holidays.txt`. JavaScript syntax: `node --check knowledge-ai-frontend/app.js`; the table/markdown parsers are covered by `node knowledge-ai-frontend/tests/table-parser.test.js`.

Recommended questions (with the Company Policies document indexed; it includes the holidays calendar):

```
Show me the holidays.
When is Independence Day?
What is the leave policy?
What are the working hours?          (typo: "working hors" also works)
What is the company's address?       / "email address?" / "contact number?"
Summarize this document.
What is the salary of the CEO of NASA?   -> clean "not found"
```

## Deploying the backend to Render

- Repo includes `render.yaml` (blueprint): `rootDir: knowledge-ai-backend`, start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Or create a Web Service in the dashboard: root directory `knowledge-ai-backend`, build `pip install -r requirements.txt`, start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
- Set `OPENAI_API_KEY` (and optionally `OPENROUTER_API_KEY`) as environment variables. Never commit `.env`.
- CORS is `allow_origins=["*"]`; note the dataset: data lives in the ephemeral instance disk (`data/`) and is lost on redeploy/restart. For persistence you would need a managed store (e.g. hosted SQLite/PG for metadata and object storage for files/vectors).

**Memory (measured):** the previous stack (sentence-transformers + CUDA torch build) peaked at **~824 MB RSS** and imported torch at ~740 MB. Replaced with **fastembed (ONNX Runtime)** using the same `all-MiniLM-L6-v2` model (same 384-d vectors, existing index remains usable). Measured on this machine with the model loaded and a chat performed: **peak RSS 355 MB** (VmHWM). A Render 512 MB instance has ~355 MB + headroom for requests/LLM client traffic, so it is realistic — but the exact Render figure depends on running it there; uploads keep only 1 model instance and embeddings streams in batches of 32.

## Deploying the frontend to Vercel

- Import the `knowledge-ai-frontend` folder (or repo) as a Vercel project. It needs no build step; `vercel.json` sets `outputDirectory: "."`.
- API base is not hardcoded throughout `app.js`. Set `window.RAG_API_BASE` in `config.js` to your deployed backend, e.g. `"https://your-app.onrender.com"` (this is the export-config equivalent of `VITE_API_BASE_URL` for this build-free app). For a code-based alternative you can edit `config.js`.
- Because the frontend is build-free, a `VITE_*` env var is not inlined; the `config.js` global is the mechanism.
- Backend CORS already allows all origins, so the Vercel origin can call Render.

## Known limitations

- Without a configured/working LLM, answers fall back to a semantic-extraction heuristic (bullet/table snippets, no natural-language reasoning). Retrieval itself is intent-aware, but exact pronoun resolution still needs the LLM.
- "Summarize this document" lists the document's sections structurally, matching a company-policies style document; prose documents without section headings fall back to a top-k synthesis.
- Table detection on PDFs depends on PyMuPDF's `find_tables`; if the source PDF does not expose table structure, pipe-delimited text is preserved as best effort.
- All data is local to the instance disk — not durable across redeploys.
- `requirements.txt` is pinned; GPU/accelerator wheels are not installed (CPU only).

## Security

- `.env` contains live secrets and is gitignored — never commit it.
- The git remote URL contains a personal access token; do not expose `.git/config`.
- Backend error responses are user-friendly; details go to server logs.