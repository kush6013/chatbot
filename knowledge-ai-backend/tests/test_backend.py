import os
import re
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
from app.services.chunker import chunk_text, chunk_text_with_meta, _looks_like_heading
from app.services.vector_store import VectorStore
from app.services.rag import (
    deduplicate_results,
    normalize_query,
    detect_intents,
    retrieve,
    debug_retrieval,
    answer_question
)
from app.main import app

import app.services.vector_store as vector_store_module

client = TestClient(app)

import app.services.vector_store as vector_store_module
import pytest


@pytest.fixture(scope="module", autouse=True)
def ensure_rag_fixtures():
    """Rebuild the vector store from dev fixtures so retrieval tests are
    deterministic. Indexes BOTH the policy PDF and the holiday calendar so
    document-aware routing/selection is exercised. Skips silently (with a
    clear reason) when the policy document is not present locally."""
    policy_pdf = os.path.join("data", "uploads", "Company_Policies_25-26.pdf")
    holidays_txt = os.path.join("data", "uploads", "company_holidays.txt")
    if not os.path.exists(policy_pdf):
        pytest.skip("Policy PDF fixture not present; skipping retrieval tests")

    with open(holidays_txt, "w", encoding="utf-8") as handle:
        handle.write(
            "Company Holiday Calendar\n\n"
            "Date | Festival/Holiday | Day\n"
            "15 Aug 2026 | Independence Day | Saturday\n"
            "26 Jan 2027 | Republic Day | Tuesday\n"
            "14 Nov 2026 | Children's Day | Saturday\n"
            "25 Dec 2026 | Christmas | Friday\n"
        )

    from app.config import VECTOR_STORE_DIR
    import app.services.vector_store as vstore
    from app.services.ingestion import process_document
    for f in ("index.faiss", "metadata.json"):
        p = os.path.join(str(VECTOR_STORE_DIR), f)
        if os.path.exists(p):
            os.remove(p)
    vstore.vector_store.index = None
    vstore.vector_store.metadata = []
    vstore.vector_store.save()
    process_document(policy_pdf)
    process_document(holidays_txt)


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


def test_chunker_preserves_table_rows():
    table = (
        "Company Holiday Calendar\n\n"
        "| Date | Festival/Holiday | Day |\n"
        "| 15 Aug 2026 | Independence Day | Saturday |\n"
        "| 26 Jan 2027 | Republic Day | Tuesday |"
    )
    chunks = chunk_text(table)
    joined = " ".join(chunks)
    assert "15 Aug 2026" in joined
    assert "Independence Day" in joined
    assert "Republic Day" in joined
    assert "26 Jan 2027" in joined


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


def test_vector_store_delete_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        vector_store_module, "VECTOR_STORE_DIR", str(tmp_path)
    )
    monkeypatch.setattr(
        vector_store_module, "INDEX_FILE", str(tmp_path / "index.faiss")
    )
    monkeypatch.setattr(
        vector_store_module, "METADATA_FILE", str(tmp_path / "metadata.json")
    )

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


def test_deduplicate_results_removes_near_duplicates():
    results = [
        {
            "source": "doc.pdf",
            "page": 1,
            "score": 0.9,
            "text": (
                "Tel: +91 94251 51787 Email: reachus@gravitinfosystems.com "
                "Web: www.gravitinfosystems.com GRAVIT INFOSYSTEMS PRIVATE "
                "LIMITED S7B Second Floor Elixir MK City Square Sirol Road"
            )
        },
        {
            "source": "doc.pdf",
            "page": 2,
            "score": 0.88,
            "text": (
                "Tel: +91 94251 51787 Email: reachus@gravitinfosystems.com "
                "Web: www.gravitinfosystems.com GRAVIT INFOSYSTEMS PRIVATE "
                "LIMITED S7B Second Floor Elixir MK City Square"
            )
        },
        {
            "source": "doc.pdf",
            "page": 3,
            "score": 0.7,
            "text": (
                "All employees must use their full working hours for company "
                "business as described in the job description."
            )
        }
    ]
    kept = deduplicate_results(results)
    assert len(kept) == 2
    assert kept[1]["text"].startswith("All employees")


def test_upload_unsupported_type():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.pdf.exe", b"fake", "application/octet-stream")}
    )
    assert response.status_code == 415
def test_upload_empty_file():
    response = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Query normalization / typo tolerance
# ---------------------------------------------------------------------------

def test_normalize_query_fixes_typos_and_plurals():
    assert normalize_query("working hors") == "working hours"
    assert normalize_query("compny details") == "company detail"
    assert normalize_query("adress of the company") == "address of the company"
    assert normalize_query("holidy") == "holiday"
    assert normalize_query("leave polcy") == "leave policy"
    assert normalize_query("this doc") == "this doc"


def test_intent_detection():
    assert "leave_policy" in detect_intents(normalize_query("what is the leave policy"))
    assert "working_hours" in detect_intents(normalize_query("working hours"))
    assert "holiday" in detect_intents(normalize_query("show me holidays"))
    assert "contact_information" in detect_intents(
        normalize_query("what is the contact number")
    )
    assert "address" in detect_intents(normalize_query("what is the address"))
    assert "document_summary" in detect_intents(normalize_query("summarize this doc"))
    assert detect_intents(normalize_query("what is the weather")) == ["general_question"]


# ---------------------------------------------------------------------------
# Heading-aware chunking
# ---------------------------------------------------------------------------

def test_chunker_detects_headings_and_types():
    text = (
        "Office Timings\n"
        "Office timing is from 10:00 am to 6:30 pm.\n"
        "Lunch break is from 2 pm to 2.45 pm.\n\n"
        "Date | Festival/Holiday | Day\n"
        "15 Aug 2026 | Independence Day | Saturday"
    )
    metas = chunk_text_with_meta(text)
    metas = [m for m in metas if m["text"].strip()]
    headings = {m.get("heading") for m in metas}
    types = {m.get("type") for m in metas}
    assert "Office Timings" in headings
    assert "table" in types
    assert any("10:00 am" in m["text"] for m in metas)
    # The table block must not be marked as a heading.
    table_meta = [m for m in metas if m.get("type") == "table"]
    assert table_meta and table_meta[0].get("heading") not in ("Date", "")


def test_chunker_data_rows_not_mistaken_for_headings():
    # Two title-cased short lines in a row = data list, not headings.
    text = "Independence Day\n15th August 2025\nRepublic Day\n26th January 2026"
    metas = chunk_text_with_meta(text)
    headings = {m.get("heading") for m in metas if m.get("heading")}
    assert "Republic Day" not in headings
    assert "Independence Day" not in headings


# ---------------------------------------------------------------------------
# Hybrid retrieval ranking
# ---------------------------------------------------------------------------

def _top_retrieved(query, n=5):
    return retrieve(query)["accepted"][:n]


def test_company_address_outranks_policy():
    results = _top_retrieved("company details like address, contact detail", n=4)
    assert results
    top = results[0]
    assert "S7B" in top.get("text", "")
    assert not any(
        "Non-Disclosure" in (r.get("text") or "") or "medical" in (r.get("text") or "").lower()
        for r in results
    )


def test_company_address_direct_question():
    results = _top_retrieved("what is the company address?", n=4)
    assert results
    assert "S7B" in results[0].get("text", "")


def test_contact_number_returns_phone():
    results = _top_retrieved("what is the contact number?", n=4)
    assert results
    assert "+91" in results[0].get("text", "") or "Tel" in results[0].get("text", "")


def test_working_hours_returns_office_timings():
    results = _top_retrieved("working hors", n=4)
    assert results
    texts = " ".join(r.get("text", "") for r in results)
    assert re.search(r"\d{1,2}[:.]\d{2}\s*(am|pm)", texts)  # e.g. 10:00 am / 6:30 pm


def test_leave_policy_heading_boosted():
    results = _top_retrieved("leave policy", n=4)
    assert results
    texts = " ".join(r.get("text", "") for r in results)
    assert re.search(r"\bleave\b|\bprobation\b|\btime off\b", texts, re.IGNORECASE)


def test_holiday_table_boosted():
    results = _top_retrieved("show me holidays", n=4)
    assert results
    top = results[0]
    assert top.get("type") == "table"
    assert "Date" in top.get("text", "") or "Holiday" in top.get("text", "")


def test_not_found_salary_nasa():
    results = _top_retrieved("What is the salary of the CEO of NASA?", n=4)
    assert results == []


# ---------------------------------------------------------------------------
# End-to-end answers (heuristic fallback runs locally)
# ---------------------------------------------------------------------------

def _only_sources(r):
    return [s["source"] for s in r["sources"]]


def test_answer_greeting_no_rag():
    r = answer_question("How are you?", [])
    assert "couldn't find" not in r["answer"]
    assert r["answer"].lower() != r["answer"]
    assert r["sources"] == []


def test_answer_company_details():
    r = answer_question("company details like address, contact detail", [])
    assert "S7B" in r["answer"]
    assert "couldn't find" not in r["answer"]
    assert "medical" not in r["answer"].lower()
    assert "non-disclosure" not in r["answer"].lower()
    assert r["sources"] and r["sources"][0]["source"] == "Company_Policies_25-26.pdf"


def test_answer_company_address():
    r = answer_question("what is the company address?", [])
    assert "S7B" in r["answer"] or "Sirol" in r["answer"]
    assert _only_sources(r) and all(s == "Company_Policies_25-26.pdf" for s in _only_sources(r))


def test_answer_contact_number():
    r = answer_question("what is the contact number?", [])
    assert "+91" in r["answer"] or "Tel" in r["answer"]
    assert _only_sources(r) and all(s == "Company_Policies_25-26.pdf" for s in _only_sources(r))


def test_answer_email():
    r = answer_question("what is the email address?", [])
    assert re.search(r"\S+@\S+", r["answer"])


def test_answer_working_hours():
    r = answer_question("what are the working hours?", [])
    assert "10:00" in r["answer"] or "6:30" in r["answer"]


def test_answer_leave_policy_no_form_junk():
    r = answer_question("what is the leave policy?", [])
    assert "couldn't find" not in r["answer"]
    assert "Duration: from" not in r["answer"]
    assert re.search(r"\bleave\b|\btime off\b|\bprobation\b",
                     r["answer"].lower())


def test_answer_holidays_table_3_columns():
    r = answer_question("show me holidays", [])
    assert "| Date | Festival/Holiday | Day |" in r["answer"]
    assert "Independence Day" in r["answer"]
    assert "Christmas" in r["answer"]
    assert "Saturday" in r["answer"]


def test_show_me_holidays_sources_txt_only():
    r = answer_question("show me holidays", [])
    assert _only_sources(r) == ["company_holidays.txt"]


def test_summarize_company_holidays_txt():
    r = answer_question("summarize company_holidays.txt", [])
    assert "company_holidays.txt" in r["answer"] or "holiday" in r["answer"].lower()
    assert _only_sources(r) == ["company_holidays.txt"]


def test_summarize_policy_pdf():
    r = answer_question("summarize Company_Policies_25-26.pdf", [])
    assert "couldn't find" not in r["answer"]
    assert "Which document" not in r["answer"]
    assert _only_sources(r)
    assert all(s == "Company_Policies_25-26.pdf" for s in _only_sources(r))


def test_summarize_ambiguous_clarification():
    r = answer_question("summarize this document", [])
    assert "Which document" in r["answer"]
    assert r["sources"] == []


def test_summarize_this_document_with_context():
    history = [
        {"role": "user", "content": "tell me about the leave policy in Company_Policies_25-26.pdf"},
        {"role": "assistant", "content": "Based on your document..."},
    ]
    r = answer_question("summarize this document", history)
    assert "Which document" not in r["answer"]
    assert "couldn't find" not in r["answer"]
    assert _only_sources(r)
    assert all(s == "Company_Policies_25-26.pdf" for s in _only_sources(r))


def test_answer_not_found():
    r = answer_question("What is the salary of the CEO of NASA?", [])
    assert "couldn't find" in r["answer"]
    assert r["sources"] == []


def test_sources_only_used_chunks():
    r = answer_question("show me holidays", [])
    assert _only_sources(r) == ["company_holidays.txt"]


def test_txt_source_page_is_null():
    r = answer_question("show me holidays", [])
    assert _only_sources(r) == ["company_holidays.txt"]
    assert r["sources"][0]["page"] is None


def test_pdf_source_page_is_number():
    r = answer_question("what is the company address?", [])
    assert _only_sources(r)
    for source in r["sources"]:
        if source["source"] == "Company_Policies_25-26.pdf":
            assert isinstance(source["page"], int)


def test_sources_deduplicated():
    r = answer_question("summarize Company_Policies_25-26.pdf", [])
    keys = [(s["source"], s["page"]) for s in r["sources"]]
    assert len(keys) == len(set(keys))


def test_summary_does_not_repeat_boilerplate():
    r = answer_question("summarize Company_Policies_25-26.pdf", [])
    assert r["answer"].count("Tel:") <= 1
    assert r["answer"].count("reachus@") <= 1


def test_answer_summary_does_not_leak_other_doc():
    r = answer_question("summarize company_holidays.txt", [])
    assert _only_sources(r) == ["company_holidays.txt"]


def test_conversation_followup_independence_day():
    history = [
        {"role": "user", "content": "When is Independence Day?"},
        {"role": "assistant", "content": "Based on your document..."},
    ]
    r = answer_question("What day is it?", history)
    assert "Independence Day" in r["answer"] or "Saturday" in r["answer"] or "Friday" in r["answer"]
