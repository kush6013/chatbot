# AGENTS.md

Two sibling apps in one repo, no monorepo tooling. README.md at root is current; keep it in sync.

## Components

- `knowledge-ai-backend/` — Python FastAPI RAG chatbot (SQLite + FAISS + fastembed/ONNX embeddings + OpenAI/OpenRouter). Deployed to Render via `render.yaml`.
- `knowledge-ai-frontend/` — pure static HTML/CSS/JS (no npm, no build step). Has `config.js` + `vercel.json`.

## Running the backend

`config.py` resolves `.env` and `data/` relative to the module (no longer CWD-dependent), but still launch from `knowledge-ai-backend/` so imports resolve.

```bash
cd knowledge-ai-backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Startup takes **~20–30s** (embedding model load). Not a hang — wait for "Application startup complete".
- Frontend defaults to `http://127.0.0.1:8000`; port 8000 is required locally. `config.js` overrides via `window.RAG_API_BASE`.
- Health: `curl http://127.0.0.1:8000/api/health` (also `/health`).
- LLM: OpenAI first, OpenRouter fallback, then heuristic extraction. The current OpenAI key has **no credits** (429), so the heuristic path is what actually runs locally.

## Virtualenvs (easy to get wrong)

- Backend `.venv/` — runtime deps **and** pytest. Use this one.
- Backend `venv/` — runtime deps only, **no pytest**.
- Root `./.venv` — exists but empty/unused, ignore it.
- `requirements.txt` is now pinned.

## Tests

```bash
cd knowledge-ai-backend
./.venv/bin/python -m pytest tests/ -v
```

`test_vector_store_delete_source` now uses a temp dir (no dev-data pollution). Endpoint tests (upload/delete) briefly touch real `data/`. **LLM-dependent chat tests will hit the 429 path** — don't assert specific LLM phrasing.

## Frontend

```bash
cd knowledge-ai-frontend
python3 -m http.server 5173
```

- `app.js` reads `window.RAG_API_BASE || 'http://127.0.0.1:8000'` — no more Render-URL routing based on hostname.
- Messages render light Markdown (tables, bullets, `**bold**`, `*(source)*`) via DOM APIs — XSS-safe, keep it that way.
- JS syntax check: `node --check knowledge-ai-frontend/app.js` (node v22 available).

## Memory / Render

- Embeddings were switched from sentence-transformers+CUDA torch (measured **~824 MB RSS**) to **fastembed/ONNX** (same `all-MiniLM-L6-v2`, same 384-d vectors). Measured locally with model loaded + a chat: **~355 MB VmHWM** — realistically fits Render's 512 MB free tier, but that claim is only a measurement here, not on Render. `requirements.txt` must be reinstalled on any existing env that still has the old torch stack.
- `EMBEDDING_MODEL` accepts `all-MiniLM-L6-v2` (mapped to `sentence-transformers/all-MiniLM-L6-v2`) or `BAAI/bge-small-en-v1.5` — both 384-d. Changing the model invalidates `data/vectors/` (re-upload docs or delete `data/`).
- `data/` is instance-local disk (uploads, vectors, SQLite) — lost on redeploy.

## Security notes

- GitHub PAT is embedded in the git remote URL in `.git/config` — don't push it or expose `.git/`.
- `.env` (gitignored) holds a live OpenAI key. Never commit it.
- Deployed tests use `data/uploads/Company_Policies_25-26.pdf` (contains contact, timings, leave policy, and a page-10 holidays calendar) plus `data/uploads/company_holidays.txt` (the holiday table, recreated by the test fixture if absent). There is no `sample_knowledge/` anymore.

## Pipeline notes (agent-relevant)

- `chunker.py` treats consecutive pipe-delimited lines as atomic table blocks — don't break that. `chunk_text_with_meta` also attaches a `heading` + `type` (paragraph/table) to each chunk; `chunk_text` stays backward-compatible.
- `rag.py` is a hybrid pipeline: conversational router (greetings/thanks/bye bypass RAG entirely) → `normalize_query` (typo/plural) → `detect_intents` (contact, address, working hours, leave, holiday, summary, …) → document-aware filter (named file / distinctive token like `holidays` → restrict retrieval to that doc) → `_expand_query` → embed → FAISS fetch + lexical intent fallback (`_extra_intent_candidates` pulls letterhead/contact/address chunks past the embedding window) → `rerank_results` (semantic + keyword + phrase + heading + intent + table + time − duplicate) → `_select_answer_chunks` gate → near-duplicate removal (word-containment 0.85).
- The duplicate penalty keys on the chunk's *normalized text*, not (source, page, heading) — distinct chunks sharing an empty heading on one page are NOT duplicates.
- The answer gate: specific-intent questions need `MIN_FINAL_SCORE` (0.75) OR a strong pattern-backed intent, plus `SEMANTIC_FLOOR`; general questions need `GENERAL_SEMANTIC_FLOOR` + keyword evidence. Below those, `answer_question` returns the exact "I couldn't find this information…" string with empty sources.
- Contact/address/timings chunks sometimes sit far down the FAISS ranking (e.g. a repeated letterhead); the medium gate + intent extras + lexical fallback keep them reachable. `_medium_presence` treats `address` as a *pattern* (S7B/road/PIN), not the literal word.
- `document_summary` intent takes `_retrieve_summary_structure` (one chunk per source+heading) — it bypasses top-k entirely. With several uploaded documents and no file reference, "summarize this document" asks which file rather than guessing; `_resolve_summary_target` falls back to the most recently referenced file in the conversation.
- Sources returned by `/api/chat` are the chunks the answer actually used (heuristic path), deduped by `(document_id, source, page)`; TXT chunks report `page: null`. `document_id` is `sha1(filename)[:16]`, computed in `config.document_id_for` and kept in DB rows and vector metadata.
- `extract_concise_answer` picks lines with a pattern bonus (contact/@/+91/www, S7B/road/PIN, clock times) on top of embedding cosine; don't remove that or letterhead answers regress.
- `POST /api/chat/debug` (returns per-chunk scores) only exists when `RAG_DEBUG=true`; it must stay gated (default 404).
- Changing the embeddings index requires metadata with the new keys — re-upload or delete `data/`.
- `api_documents.py` streams uploads to disk, caps at `MAX_UPLOAD_SIZE`, and re-upload of same filename replaces old vectors (delete BEFORE process — order matters).