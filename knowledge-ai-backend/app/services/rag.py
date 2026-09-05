import logging
import re
import numpy as np
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
    FETCH_K,
    SUMMARY_FETCH_K,
    MIN_FINAL_SCORE,
    SEMANTIC_FLOOR,
    GENERAL_SEMANTIC_FLOOR,
    RAG_DEBUG,
    document_id_for
)

from app.services.embeddings import (
    embedding_service
)

from app.services.vector_store import (
    vector_store
)

logger = logging.getLogger("knowledge_ai")

MAX_CONTEXT_CHARS = 5000
MAX_HISTORY_MESSAGES = 6

# Hybrid ranking weights (transparent + configurable via env).
W_SEMANTIC = float(__import__("os").getenv("W_SEMANTIC", "1.0"))
W_KEYWORD = float(__import__("os").getenv("W_KEYWORD", "0.5"))
W_PHRASE = float(__import__("os").getenv("W_PHRASE", "0.6"))
W_HEADING = float(__import__("os").getenv("W_HEADING", "0.45"))
W_INTENT = float(__import__("os").getenv("W_INTENT", "0.45"))
W_TABLE = float(__import__("os").getenv("W_TABLE", "0.5"))
W_TIME = float(__import__("os").getenv("W_TIME", "0.5"))
W_DUP = float(__import__("os").getenv("W_DUP", "0.35"))


SYSTEM_PROMPT = """You are a document-grounded assistant.

Answer the user's question using ONLY the retrieved information from the uploaded documents.

Rules:
- Do not invent information. If the answer cannot be supported by the retrieved documents, say clearly: "I couldn't find this information in the uploaded documents."
- Do not use unrelated retrieved content.
- Do not repeat the same information twice, and do not copy redundant overlapping excerpts from the context.
- If the relevant source information is tabular and the user's question requires tabular information, answer using a Markdown table.
- If the source information is a list, use bullet points.
- For normal explanatory questions, use concise paragraphs.
- For summary questions ("summarize this document"), produce a coherent short document overview covering the main sections present in the retrieved context. Never invent sections that are not present.
- Do not reproduce the entire document. Answer only what the user asked.
- Preserve factual values such as dates, names, numbers and policy rules exactly as supported by the source.
- Do not include a "Sources:" section in your answer; sources are displayed separately."""


# ---------------------------------------------------------------------------
# Query normalization / typo tolerance
# ---------------------------------------------------------------------------

_COMMON_TYPOS = {
    "hors": "hours",
    "compny": "company",
    "polcy": "policy",
    "polcies": "policies",
    "adress": "address",
    "adresses": "addresses",
    "holidy": "holiday",
    "holidayss": "holidays",
    "emial": "email",
    "eamil": "email",
    "rutine": "routine",
    "leaves": "leave",
    "timings": "timing",
    "policies": "policy",
}

# Words that end in "s" but are NOT plurals - never strip.
_NON_PLURALS = {
    "this", "its", "as", "us", "has", "was", "is", "his", "hers",
    "ours", "yours", "theirs", "always", "bus", "status", "class",
    "process", "focus", "bonus", "thus", "rather",
}


def normalize_query(question: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace and fix typos."""
    text = question.lower()
    text = re.sub(r"[\"'`]", "", text)
    tokens = re.findall(r"[a-z0-9]+", text)

    fixed = []
    for token in tokens:
        if token in _COMMON_TYPOS:
            token = _COMMON_TYPOS[token]
        elif token in _NON_PLURALS or token.endswith("ss"):
            pass
        elif token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("es") and len(token) > 4 and token not in _NON_PLURALS:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3 and token not in _NON_PLURALS:
            token = token[:-1]
        fixed.append(token)

    return " ".join(fixed)


def _light_normalize(text: str) -> str:
    """Lowercase + collapse whitespace, keeping punctuation/digits intact."""
    return re.sub(r"\s+", " ", text).strip().lower()


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "about", "what", "which", "when", "where", "why",
    "how", "who", "whom", "of", "in", "on", "at", "to", "for", "from", "by",
    "with", "and", "but", "or", "so", "if", "then", "than", "too", "some",
    "info", "tell", "give", "show", "me", "this", "that", "my", "am",
}


def _significant_tokens(query: str):
    return [
        token for token in query.split()
        if token not in STOPWORDS and len(token) >= 3
    ]


# ---------------------------------------------------------------------------
# Intent detection (lightweight, rule based)
# ---------------------------------------------------------------------------

INTENT_TERMS = {
    "company_information": [
        "company", "company detail", "company information", "company overview",
        "about the company", "about us", "organization", "establishment",
    ],
    "contact_information": [
        "contact", "phone", "telephone", "mobile", "email", "mail address",
        "website", "web", "reach us", "reachus", "call",
    ],
    "address": [
        "address", "located", "location", "where are", "office address",
        "pincode", "pin code", "city", "registered office",
    ],
    "working_hours": [
        "working hour", "office timing", "office hour", "work timing",
        "working time", "hours of work", "business hour", "what time",
        "timing",
    ],
    "leave_policy": [
        "leave policy", "leave", "time off", "leaves", "paid leave",
        "casual leave", "absent",
    ],
    "holiday": [
        "holiday", "holidays", "festival", "dates", "calendar", "day off",
        "when is", "when are", "which day", "list of holiday",
    ],
    "work_from_home": [
        "work from home", "wfh", "remote work", "home office",
    ],
    "internship_policy": [
        "intern", "internship", "stipend", "probation",
    ],
    "confidentiality": [
        "confidential", "confidentiality", "non disclosure", "nda", "secrecy",
    ],
    "notice_period": [
        "notice period", "notice", "resignation", "resign", "relieving",
    ],
    "document_summary": [
        "summary", "summarize", "summarise", "overview of the document",
        "about this document", "what is this document",
        "describe this document", "document content", "main points",
    ],
}

# Detection order: more specific intents first.
_INTENT_PRIORITY = [
    "document_summary",
    "working_hours",
    "work_from_home",
    "contact_information",
    "address",
    "company_information",
    "leave_policy",
    "holiday",
    "internship_policy",
    "confidentiality",
    "notice_period",
]


def _norm_intent_phrase(phrase: str) -> str:
    return normalize_query(phrase)


def detect_intents(normalized_query: str) -> list:
    """Return the set of intents the query plausibly matches."""
    detected = []
    for intent in _INTENT_PRIORITY:
        for phrase in INTENT_TERMS[intent]:
            if _norm_intent_phrase(phrase) in normalized_query:
                detected.append(intent)
                break
    if not detected:
        return ["general_question"]

    # Bare "company detail(s)/information" usually means the address,
    # phone and email block too - activate those facets so the company
    # letterhead row itself becomes a strong candidate.
    if detected == ["company_information"]:
        detected.extend(["contact_information", "address"])

    return detected


# Matchers used to score a chunk against a query token.
def _token_match(needle: str, hay_tokens) -> bool:
    for candidate in hay_tokens:
        if candidate == needle:
            return True
    for candidate in hay_tokens:
        if not candidate or not needle:
            continue
        if candidate == needle + "s" or needle == candidate + "s":
            return True
        if len(needle) >= 4 and len(candidate) >= 4:
            if candidate.startswith(needle) or needle.startswith(candidate):
                return True
    return False


_PHRASE_LIST = [
    "working hour", "office timing", "office hour", "work from home",
    "leave policy", "time off", "company detail", "company information",
    "contact detail", "contact information", "notice period",
    "non disclosure", "internship policy", "day of week", "what time",
]


def _phrase_hit(normalized_chunk: str, phrase: str) -> bool:
    target = _norm_intent_phrase(phrase)
    if target in normalized_chunk:
        return True
    if target.endswith("s") and target[:-1] in normalized_chunk:
        return True
    return False


_HEADING_SPECIFIC = {
    "phone", "telephone", "email", "mail", "address", "holiday", "festival",
    "leave", "time off", "nda", "stipend", "internship", "wfh", "work from home",
    "summary", "timing", "working hour", "office timing", "resignation",
    "notice period", "punctuality", "probation", "salary", "cell phone",
}


def _heading_score(intent, raw_heading: str) -> float:
    """Credit headings only when they contain a *specific* term/phrase.

    Generic words such as "company" or "policy" in a heading must NOT
    act as a retrieval signal (that is what lifted "Non-Compliance with
    Company Policies" above the real company-info block before).
    """
    heading = _light_normalize(raw_heading or "")
    heading_words = set(heading.split())
    score = 0.0

    for phrase in INTENT_TERMS.get(intent, []):
        target = normalize_query(phrase)
        if not target:
            continue
        if " " in target:
            if _phrase_hit(heading, target):
                score += 1.0
        else:
            if target in _HEADING_SPECIFIC and target in heading_words:
                score += 1.0

    return min(1.0, score)


def _intent_extra(intent, raw_text: str) -> float:
    """Domain-specific pattern boosts on the RAW text (keeps @, +, times)."""
    light = _light_normalize(raw_text or "")
    if intent == "contact_information":
        patterns = [r"\b\+91\b", r"@", r"\btel", r"reachus", r"\bwww\.", r"email"]
        hits = sum(1 for pattern in patterns if re.search(pattern, light))
        score = hits / 3.0
        # A chunk carrying the actual contact block (phone+email+web)
        # must beat an address-only chunk for contact queries.
        if hits >= 3:
            score += 0.4
        return min(1.0, score)
    if intent == "address":
        patterns = [r"\bs7b\b", r"second floor", r"city square", r"\broad\b",
                    r"gwalior", r"madhya pradesh", r"\b\d{6}\b"]
        hits = sum(1 for pattern in patterns if re.search(pattern, light))
        return min(1.0, hits / 2.0)
    if intent == "company_information":
        # The company letterhead block is the strongest "company detail"
        # evidence even though it lacks the literal word "company".
        patterns = [r"private limited", r"pvt\.?\s*ltd", r"privated\s+limited",
                    r"ltd"]
        if any(re.search(pattern, light) for pattern in patterns):
            return 0.5
        return 0.0
    if intent == "working_hours":
        patterns = [r"\b\d{1,2}[:.]\d{2}\s*(am|pm|a m|p m)\b",
                    r"\b\d{1,2}[:.]\d{2}\b"]
        hits = sum(1 for pattern in patterns if re.search(pattern, light))
        return min(1.0, hits / 2.0)
    return 0.0


def _time_pattern_hits(raw_text: str) -> bool:
    return bool(re.search(
        r"\b\d{1,2}[:.]\d{2}\s*(am|pm|a m|p m)\b",
        _light_normalize(raw_text or "")
    ))


def _intent_score(intent, raw_text: str, raw_heading: str) -> float:
    terms = INTENT_TERMS.get(intent, [])
    if not terms:
        return 0.0

    light = _light_normalize(raw_text or "")
    hits = 0
    for phrase in terms:
        if _phrase_hit(light, phrase):
            hits += 1

    coverage = hits / max(1, len(terms))
    extra = _intent_extra(intent, raw_text)

    heading_boost = 0.0
    heading = _light_normalize(raw_heading or "")
    if heading and any(_phrase_hit(heading, phrase) for phrase in terms):
        heading_boost = 0.2

    return min(1.0, coverage + extra + heading_boost)


# A specific medium named in the query (email/phone/website/address) must
# gate retrieval: a chunk lacking that medium should rank below one that
# has it, even when generic intent signals tie.
_MEDIUM_TERMS = {
    "email": ["email", "mail"],
    "phone": ["phone", "telephone", "mobile", "contact number", "contact no", "tel"],
    "website": ["website", "web", "www"],
    "address": ["address", "registered office", "located", "location"],
}


def _medium_presence(raw_text: str, medium: str) -> bool:
    light = _light_normalize(raw_text or "")
    if medium == "address":
        # The registered-office block never says "address"; it *is* the
        # address (S7B, road, city, PIN). Match the physical patterns.
        return bool(
            re.search(
                r"[\da-z]+,\s*(second |2nd )?floor|s7b|road|gwalior|"
                r"panchayat|\b\d{6}\b",
                _light_normalize(raw_text or ""),
            )
        )
    return any(term in light for term in _MEDIUM_TERMS[medium])


def _named_medium(normalized_query: str):
    for medium, terms in _MEDIUM_TERMS.items():
        if any(_norm_intent_phrase(term) in normalized_query for term in terms):
            return medium
    return None


def _medium_gate_score(normalized_query, raw_text):
    """+0.6 when the chunk contains the specific contact medium asked for,
    -0.6 when it clearly does not (drags address-only chunks down for
    pure contact questions). Returns 0 when no medium is named.

    For "address" the presence check is pattern-based and a chunk that
    lacks the physical address block (e.g. the contact block) must not be
    pushed below the bar — compound "address + contact" asks want both."""
    medium = _named_medium(normalized_query)
    if medium is None:
        return 0.0
    present = _medium_presence(raw_text, medium)
    if medium == "address":
        return 0.6 if present else 0.0
    return 0.6 if present else -0.6


# ---------------------------------------------------------------------------
# Hybrid reranking
# ---------------------------------------------------------------------------

_TABLE_INTENTS = {"holiday", "working_hours"}

def _feels_like_table_query(normalized_query: str) -> bool:
    return any(
        phrase in normalized_query
        for phrase in ["table", "calendar", "schedule", "timeline",
                       "list of", "dates", "day of week", "when is", "when are"]
    )


# Query expansion: compound/short queries get canonical terms appended so
# the embedding pulls the genuinely relevant blocks (e.g. the company
# letterhead with address/contact) into the semantic candidate set.
INTENT_EXPANSIONS = {
    "company_information": [
        "company overview about the company contact email phone website address",
    ],
    "contact_information": [
        "phone telephone email website address contact reach us",
    ],
    "address": [
        "address location street road city pin code registered office",
    ],
    "working_hours": [
        "timings working hours office hours office timing start end morning evening",
    ],
    "leave_policy": [
        "leave policy time off leave days off absentee",
    ],
    "holiday": [
        "holiday dates festivals calendar off days when",
    ],
}


def _expand_query(normalized_query: str, intents: list) -> str:
    additions = []
    for intent in intents:
        additions.extend(INTENT_EXPANSIONS.get(intent, []))
    return " ".join([normalized_query] + additions).strip()


def rerank_results(candidates, normalized_query: str, intents: list):
    """Compute a transparent hybrid score for each retrieved chunk.

    final_score = semantic + keyword + phrase + heading + intent + table
                  + time - duplicate penalty
    """
    analyzed = []
    seen_heading_pages = {}

    for result in candidates:
        text = result.get("text", "") or ""
        heading_text = result.get("heading") or ""
        chunk_type = result.get("type") or "paragraph"
        source = result.get("source", "")
        page = result.get("page")

        semantic = float(result.get("score", 0.0))

        normalized_text = normalize_query(text)
        normalized_heading = normalize_query(heading_text)
        text_tokens = set(normalized_text.split())
        query_tokens = _significant_tokens(normalized_query)

        if query_tokens:
            matched = sum(
                1 for token in query_tokens
                if _token_match(token, text_tokens)
            )
            keyword = matched / len(query_tokens)
        else:
            keyword = 0.0

        phrase_hits = 0
        phrase_count = 0
        candidate_phrases = _PHRASE_LIST + [
            " ".join(window) for window in zip(query_tokens, query_tokens[1:])
        ]
        for phrase in candidate_phrases:
            target = _norm_intent_phrase(phrase)
            if not target:
                continue
            phrase_count += 1
            if _phrase_hit(normalized_text, target):
                phrase_hits += 1
        phrase = phrase_hits / max(1, phrase_count)

        heading_scores = [
            _heading_score(intent, heading_text)
            for intent in intents
        ]
        heading = max(heading_scores) if heading_scores else 0.0

        intent_scores = [
            _intent_score(intent, text, heading_text)
            for intent in intents
        ]
        # Strongest satisfied intent carries the query (compound queries
        # like "address + contact" must not be averaged into weakness).
        intent_score = max(intent_scores) if intent_scores else 0.0

        table_on = (chunk_type == "table" and (
            any(intent in _TABLE_INTENTS for intent in intents)
            or _feels_like_table_query(normalized_query)
        ))
        table = 1.0 if table_on else 0.0

        time = 1.0 if ("working_hours" in intents
                       and _time_pattern_hits(text)) else 0.0

        medium = _medium_gate_score(normalized_query, text)

        # Duplicate penalty: the same block repeated across the document
        # (letterhead, headers) must not dominate retrieval. Key on the
        # normalized TEXT, not (source, page, heading) — distinct chunks
        # that merely share an empty heading on one page are not duplicates.
        dup_key = (source, normalized_text)
        seen_heading_pages[dup_key] = seen_heading_pages.get(dup_key, 0) + 1
        duplicate_penalty = (seen_heading_pages[dup_key] - 1) * W_DUP

        final_score = (
            W_SEMANTIC * semantic
            + W_KEYWORD * keyword
            + W_PHRASE * phrase
            + W_HEADING * heading
            + W_INTENT * intent_score
            + W_TABLE * table
            + W_TIME * time
            + medium
            - duplicate_penalty
        )

        analyzed.append({
            **result,
            "score": final_score,
            "semantic_score": semantic,
            "keyword_score": keyword,
            "phrase_score": phrase,
            "heading_score": heading,
            "intent_score": intent_score,
            "table_score": table,
            "time_score": time,
        })

    return sorted(
        analyzed,
        key=lambda item: item["score"],
        reverse=True
    )


def _has_keyword_evidence(results, normalized_query):
    """True when at least one significant query token appears in a result."""
    query_tokens = _significant_tokens(normalized_query)
    if not query_tokens:
        return True
    for result in results:
        text_tokens = set(normalize_query(result.get("text", "") or "").split())
        for token in query_tokens:
            if _token_match(token, text_tokens):
                return True
    return False


# ---------------------------------------------------------------------------
# Retrieval entry points
# ---------------------------------------------------------------------------

def _extra_intent_candidates(intents, named_medium, fetched_texts):
    """Lexical intent fallback: pull chunks that demonstrably carry one of
    the active content intents (contact/address/timings/…) even when their
    embedding rank is beyond the semantic fetch window. This is what keeps
    the letterhead/contact block from ever being missed."""
    extras = []
    for item in vector_store.all_metadata():
        text = item.get("text") or ""
        if not text.strip():
            continue
        if _is_redundant(text, fetched_texts):
            continue
        hit = False
        for intent in intents:
            pattern = _LINE_PATTERN_BONUS.get(intent)
            if pattern and pattern.search(text):
                hit = True
                break
        if not hit and named_medium and _medium_presence(text, named_medium):
            hit = True
        if hit:
            extras.append(item)
    return extras


def retrieve(question, conversation_history=None, is_summary=False,
             summary_target=None):
    """Normalize -> expand -> embed -> vector search -> hybrid rerank.

    Returns a list of accepted chunks sorted by final score plus the
    full analyzed list (for debug).
    """
    augmented = _augment_query(question, conversation_history or [])
    normalized_query = normalize_query(augmented)
    intents = detect_intents(normalized_query)

    doc_filter = summary_target or _extract_named_document(augmented or question)

    if is_summary:
        return _retrieve_summary_structure(
            normalized_query, intents, source_filter=doc_filter
        )

    embed_query = _expand_query(normalized_query, intents)
    query_embedding = embedding_service.generate([embed_query])[0]

    fetch_k = max(FETCH_K, TOP_K * 3)
    raw_results = vector_store.search(query_embedding, fetch_k)

    if any(intent in _LINE_PATTERN_BONUS for intent in intents):
        named_medium = _named_medium(normalized_query)
        fetched_texts = [r.get("text", "") for r in raw_results]
        extras = _extra_intent_candidates(intents, named_medium, fetched_texts)
        ids_in_raw = {r.get("id") for r in raw_results}
        extras = [e for e in extras if e.get("id") not in ids_in_raw]
        if extras:
            extra_embs = embedding_service.generate(
                [e.get("text", "") for e in extras])
            for idx, extra in enumerate(extras):
                extra["score"] = cosine_similarity(
                    query_embedding, extra_embs[idx])
            raw_results = raw_results + extras

    if doc_filter:
        raw_results = [
            r for r in raw_results if r.get("source") == doc_filter
        ]

    analyzed = rerank_results(raw_results, normalized_query, intents)

    accepted = _select_answer_chunks(analyzed, normalized_query, intents)
    accepted = deduplicate_results(accepted)

    return {
        "accepted": accepted[:TOP_K],
        "analyzed": analyzed,
        "intents": intents,
        "query_embedding": query_embedding,
        "normalized_query": normalized_query,
        "doc_filter": doc_filter,
    }


def _retrieve_summary_structure(normalized_query, intents,
                                source_filter=None):
    """Summary retrieval: cover the whole document structurally.

    picks one representative chunk per detected section heading plus a
    fallback across pages, so the summary reflects the overall document
    instead of a handful of top-k fragments.
    """
    all_items = vector_store.all_metadata()

    if source_filter:
        all_items = [
            item for item in all_items
            if item.get("source") == source_filter
        ]

    by_heading = {}
    for item in all_items:
        heading = (item.get("heading") or "").strip()
        key = (item.get("source"), heading)
        length = len(item.get("text", "") or "")
        if key not in by_heading or length > len(by_heading[key].get("text", "")):
            by_heading[key] = item

    # Document has no recognisable section headings (e.g. titles are
    # inlined after bullets): fall back to one representative chunk per
    # page so the summary still spans the whole document.
    headings = {h for (_src, h) in by_heading if h}
    if not headings and len(by_heading) < 3:
        by_page = {}
        for item in all_items:
            key = (item.get("source"), item.get("page"))
            length = len(item.get("text", "") or "")
            if key not in by_page or length > len(by_page[key].get("text", "")):
                by_page[key] = item
        by_heading = by_page if len(by_page) > len(by_heading) else by_heading

    selected = []
    seen_texts = []
    for item in by_heading.values():
        text = item.get("text", "") or ""
        if not text.strip():
            continue
        if _is_redundant(text, seen_texts):
            continue
        seen_texts.append(text)
        selected.append(item)

    if len(selected) < 3:
        for item in all_items:
            text = item.get("text", "") or ""
            if not text.strip() or _is_redundant(text, seen_texts):
                continue
            seen_texts.append(text)
            selected.append(item)
            if len(selected) >= 8:
                break

    for item in selected:
        item["score"] = 0.0
        item["semantic_score"] = 0.0
        item["keyword_score"] = 0.0
        item["phrase_score"] = 0.0
        item["heading_score"] = 0.0
        item["intent_score"] = 0.0
        item["table_score"] = 0.0
        item["time_score"] = 0.0

    return {
        "accepted": selected,
        "analyzed": selected,
        "intents": intents,
        "query_embedding": None,
        "normalized_query": normalized_query,
    }


INTENT_STRONG = 0.55


def _select_answer_chunks(analyzed, normalized_query, intents):
    specific = intents != ["general_question"]
    out = []
    for result in analyzed:
        semantic = result["semantic_score"]
        final = result["score"]
        intent_score = result["intent_score"]

        if specific:
            strong_intent = intent_score >= INTENT_STRONG
            medium_gate = _medium_gate_score(
                normalized_query, result.get("text", ""))
            # Intent score capped at ~1.0 means raw-text patterns (the
            # actual Tel/+/@/www or address block) fired, so the chunk
            # demonstrably carries the requested content even if the bare
            # embedding similarity sits just under the floor.
            pattern_backed = intent_score >= 0.99

            # Hard rule: when the question names a specific medium (email /
            # phone / website / address), a chunk that lacks that medium
            # outright must clear the final-score bar on its own — a strong
            # generic intent must not rescue it.
            if medium_gate < 0 and final < MIN_FINAL_SCORE:
                continue

            # Demonstrable evidence (contains the named medium, or is a
            # pattern-backed contact/address/time block): the final score is
            # real enough even slightly under the semantic floor.
            if medium_gate > 0 or pattern_backed:
                if final >= MIN_FINAL_SCORE or strong_intent:
                    out.append(result)
                    continue
                if semantic < SEMANTIC_FLOOR:
                    continue
            else:
                if final < MIN_FINAL_SCORE and not strong_intent:
                    continue
                if semantic < SEMANTIC_FLOOR:
                    continue
        else:
            # General questions need real semantic evidence plus at least
            # one keyword hit of the question in the retrieved text.
            if semantic < GENERAL_SEMANTIC_FLOOR:
                continue
            if not _has_keyword_evidence([result], normalized_query):
                continue
            if final < SIMILARITY_THRESHOLD:
                continue
        out.append(result)
    return out


def debug_retrieval(question, conversation_history=None):
    """Return transparent scoring information (test/debug helper).

    Not reachable through the normal chat paths.
    """
    data = retrieve(question, conversation_history)
    rows = []
    for result in data["analyzed"][:8]:
        rows.append({
            "query": question,
            "normalized_query": data["normalized_query"],
            "intents": data["intents"],
            "source": result.get("source"),
            "page": result.get("page"),
            "heading": result.get("heading") or "",
            "type": result.get("type") or "",
            "semantic_score": result["semantic_score"],
            "keyword_score": result["keyword_score"],
            "phrase_score": result["phrase_score"],
            "heading_score": result["heading_score"],
            "intent_score": result["intent_score"],
            "table_score": result["table_score"],
            "final_score": result["score"],
            "snippet": (result.get("text") or "")[:180],
        })
    return {
        "query": question,
        "normalized_query": data["normalized_query"],
        "intents": data["intents"],
        "results": rows,
    }


# ---------------------------------------------------------------------------
# Context / answer building (unchanged behavior, plus section labels)
# ---------------------------------------------------------------------------

def get_openai_client():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not configured in .env.")
    return OpenAI(api_key=OPENAI_API_KEY)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _word_set(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _is_redundant(candidate: str, kept: list, threshold: float = 0.85) -> bool:
    """True when most of the candidate's content is already in kept chunks.

    Uses word containment so genuine new information (different rows,
    different facts) survives while overlap duplicates are removed.
    """
    cand_words = _word_set(candidate)
    if len(cand_words) < 4:
        return any(_normalize(candidate) == _normalize(k) for k in kept)

    best_overlap = 0.0
    for text in kept:
        kept_words = _word_set(text)
        if not kept_words:
            continue
        overlap = len(cand_words & kept_words) / len(cand_words)
        best_overlap = max(best_overlap, overlap)

    return best_overlap >= threshold


def deduplicate_results(results):
    if not results:
        return results

    ordered = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
    kept = []
    kept_texts = []

    for result in ordered:
        text = result.get("text", "")
        if not text:
            continue
        if _is_redundant(text, kept_texts):
            continue
        kept.append(result)
        kept_texts.append(text)

    return kept


def build_context(results):
    parts = []
    used_chars = 0

    for number, result in enumerate(results, start=1):
        if used_chars >= MAX_CONTEXT_CHARS:
            break

        source = result.get("source", "Unknown document")
        page = result.get("page")
        heading = result.get("heading") or ""
        location = f"{source}, Page {page}" if page else source
        text = result.get("text", "").strip()

        lines = [f"SOURCE {number}", f"Document: {location}"]
        if heading:
            lines.append(f"Section: {heading}")
        segment = "\n".join(lines) + "\n\n" + text

        if used_chars + len(segment) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - used_chars
            if remaining > 200:
                segment = segment[:remaining]
            else:
                break

        parts.append(segment)
        used_chars += len(segment)

    return "\n\n".join(parts)


def _is_table_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines()]
    return sum(1 for line in lines if line.startswith("|") or line.count("|") >= 2) >= 2


_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")


def _pipe_cells(row: str):
    body = row.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells = [cell.strip() for cell in body.split("|")]
    while cells and cells[0] == "":
        cells.pop(0)
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def _table_as_markdown(text: str, max_rows: int = 12) -> str:
    rows = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and (line.strip().startswith("|") or line.strip().count("|") >= 2)
    ]
    rows = rows[:max_rows + 8]

    table_rows = []
    for row in rows:
        cells = _pipe_cells(row)
        if all(_SEPARATOR_CELL.fullmatch(cell) for cell in cells):
            continue
        if len(cells) == 1 and _SEPARATOR_CELL.fullmatch(cells[0]):
            continue
        table_rows.append(cells)

    if not table_rows:
        return ""

    width = max(len(row) for row in table_rows)
    header = (table_rows[0] + [""] * width)[:width]

    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join(["---"] * width) + " |")
    for row in table_rows[1:]:
        row = (row + [""] * width)[:width]
        out.append("| " + " | ".join(row) + " |")

    return "\n".join(out)


def _looks_like_table_question(question: str) -> bool:
    keywords = [
        "holiday", "holidays", "festival", "date", "dates",
        "schedule", "calendar", "table", "timeline", "when is",
        "when are", "list of", "day of", "which day"
    ]
    q = question.lower()
    return any(keyword in q for keyword in keywords)


def _strip_boilerplate_lines(text: str) -> str:
    """Drop repeated letterhead / header / footer lines so a summary does
    not repeat contact blocks on every page."""
    kept = []
    for line in text.splitlines():
        light = _light_normalize(line)
        if not line.strip():
            continue
        if re.search(
            r"\btel\b|@|\+91|www\.|reachus|phone|website:", light
        ):
            continue
        if re.search(
            r"\bs7b\b|\bsecond floor\b|\b(opposite )?jila panchayat\b|"
            r"\bgwalior\b|\bmadhya pradesh\b|\b474011\b",
            light,
        ):
            continue
        if re.search(r"^\s*(page|pg)\s*\d+", light):
            continue
        kept.append(line.strip())
    return "\n".join(kept)


def _build_summary_answer(results):
    if not results:
        return None

    sources = sorted({r.get("source") for r in results if r.get("source")})
    pages = sorted(
        {r.get("page") for r in results if r.get("page") is not None}
    )

    headings = []
    for result in results:
        heading = (result.get("heading") or "").strip()
        if heading and heading not in headings:
            headings.append(heading)

    if headings:
        intro = (
            f"This document covers {len(headings)} section(s) "
            f"({', '.join(sources)})."
        )
        if pages:
            intro += f" It spans {len(pages)} page(s)."
    elif pages:
        intro = (
            f"This document covers topics across {len(pages)} page(s) "
            f"({', '.join(sources)})."
        )
    else:
        intro = f"This document covers the following topics ({', '.join(sources)})."

    parts = [intro]

    if headings:
        parts.append("Sections:")
        emitted = set()
        for result in results:
            heading = (result.get("heading") or "").strip()
            if not heading or heading in emitted:
                continue
            emitted.add(heading)
            src = result.get("source", "Document")
            pg = result.get("page")
            loc = f"{src}, page {pg}" if pg else src
            parts.append(f"- {heading} *({loc})*")

    parts.append("Key points:")
    seen_texts = []
    for result in results:
        stripped = " ".join(
            _strip_boilerplate_lines(result.get("text", "") or "").split()
        )
        if len(stripped) < 20:
            continue
        if _is_redundant(stripped, seen_texts):
            continue
        seen_texts.append(stripped)
        src = result.get("source", "Document")
        pg = result.get("page")
        loc = f"{src}, page {pg}" if pg else src
        clipped = stripped[:120]
        suffix = "…" if len(stripped) > 120 else ""
        parts.append(f"- {clipped}{suffix} *({loc})*")
        if len(parts) >= 14:
            break

    return "\n".join(parts).strip() or None


_LINE_PATTERN_BONUS = {
    "contact_information": re.compile(
        r"\+91|@|[^a-z]tel:|reachus|www\.", re.IGNORECASE),
    "address": re.compile(
        r"s7b|floor|road|gwalior|panchayat|[a-z]{3,} +\- +4?7\d{3}\b", re.IGNORECASE),
    "working_hours": re.compile(r"\d{1,2}[:.]\d{2}\s*(am|pm)", re.IGNORECASE),
    "leave_policy": re.compile(r"\bleaves?\b|\bcasual leave\b", re.IGNORECASE),
}


def _line_pattern_bonus(intents, line: str) -> float:
    """Small bump for lines that literally carry the content the query's
    intent is asking about (contact block, address block, timings, leave).
    Lets a demonstrably-correct line clear the extract gate even when fastembed
    gives it a slightly-low bare cosine."""
    if not intents:
        return 0.0
    for intent in intents:
        pattern = _LINE_PATTERN_BONUS.get(intent)
        if pattern and pattern.search(line):
            return 0.15
    return 0.0


_FORM_LINE_RE = re.compile(
    r"\bduration\s*:\s*from\b|\breason for leave\b|\brequested by\b|"
    r"\bapproved by\b|\bsignature\b|\bemployee (id|code)\b|"
    r"from\s+_{2,}|\bvalid uses\b",
    re.IGNORECASE,
)

_CLAUSE_START = re.compile(r"\s+\([a-z]\)(?=\s+[A-Z])")


def _is_form_line(line: str) -> bool:
    """Skip blank form fields ("Duration: from ___ to ___", "Requested
    by: ____") that read as junk when extracted from a form document."""
    return bool(_FORM_LINE_RE.search(_light_normalize(line or "")))


def _trim_trailing_clauses(line: str) -> str:
    """Letterhead lines sometimes run into the next clause of a document
    ("…Gwalior, Madhya (e) The employee shall not remove…"). When a line
    carries contact/address markers, cut it at the clause so the source
    shows the relevant part only."""
    if re.search(
        r"\bs7b\b|\bsirol\b|\bgwalior\b|\breachus@\b|\bwww\.\b|\+91",
        line or "",
        re.IGNORECASE,
    ):
        match = _CLAUSE_START.search(line or "")
        if match:
            return line[: match.start()]
    return line


def _is_redundant_prefix(line: str, kept: list) -> bool:
    """True when a candidate line starts with the same normalized opening
    as an already-selected line (the letterhead repeats verbatim on many
    pages). Drops the later copies so the answer does not repeat the same
    block."""
    prefix = normalize_query(" ".join(line.split()))[:28]
    if not prefix:
        return False
    for kept_line in kept:
        kept_prefix = normalize_query(" ".join(kept_line.split()))[:28]
        if kept_prefix and prefix == kept_prefix:
            return True
    return False


def extract_concise_answer(question, relevant_results, query_embedding=None,
                           intents=None):
    """Heuristic answer builder. Returns (answer_text, used_chunks) so the
    API can display sources for exactly the chunks the answer used."""
    if not relevant_results:
        return ("I couldn't find this information in the uploaded documents.", [])

    if _looks_like_table_question(question):
        for chunk in relevant_results:
            text = chunk.get("text", "")
            if _is_table_like(text):
                table = _table_as_markdown(text)
                if table:
                    src = chunk.get("source", "Document")
                    pg = chunk.get("page")
                    loc = f"{src} (page {pg})" if pg else src
                    answer = (
                        f"Based on your document:\n\n{table}\n\nSource: {loc}"
                    )
                    return answer, [chunk]

    # Chunk-order aware line selection: the top-ranked (hybrid) chunks
    # get priority so a weak chunk's line cannot outrank the section that
    # actually matches the query intent.
    if query_embedding is None:
        query_embedding = embedding_service.generate([question])[0]

    selected = []
    selected_texts = []
    used_chunks = []
    processed = 0

    def _remember(chunk):
        for used in used_chunks:
            if (used.get("source") == chunk.get("source")
                    and used.get("page") == chunk.get("page")):
                return
        used_chunks.append(chunk)

    for chunk in relevant_results:
        if len(selected) >= 3:
            break
        processed += 1
        if processed > 4:
            break

        src = chunk.get("source", "Document")
        pg = chunk.get("page")
        loc = src if not pg else f"{src}, page {pg}"
        text = chunk.get("text", "")

        raw_lines = re.split(r"[\n•\r]+|(?<=[a-zA-Z0-9])\.\s+(?=[A-Z0-9])", text)
        lines = []
        for raw in raw_lines:
            line = " ".join(raw.split())
            if line and len(line) >= 15 and line not in selected_texts:
                lines.append(line)

        if not lines:
            continue

        line_embeddings = embedding_service.generate(lines)
        scored = [
            (cosine_similarity(query_embedding, line_embeddings[idx])
             + _line_pattern_bonus(intents, line), line)
            for idx, line in enumerate(lines)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        for sim, line in scored:
            if sim < 0.30:
                break
            line = _trim_trailing_clauses(line)
            if _is_form_line(line):
                continue
            if _is_redundant_prefix(line, selected_texts):
                continue
            if _is_redundant(line, selected_texts, threshold=0.9):
                continue
            selected.append(f"• {line} *({loc})*")
            selected_texts.append(line)
            _remember(chunk)
            if len(selected) >= 3:
                break

    if not selected:
        scored_all = []
        for chunk in relevant_results[:4]:
            src = chunk.get("source", "Document")
            pg = chunk.get("page")
            loc = src if not pg else f"{src}, page {pg}"
            for raw in re.split(r"[\n•\r]+|(?<=[a-zA-Z0-9])\.\s+(?=[A-Z0-9])",
                                chunk.get("text", "")):
                line = " ".join(raw.split())
                if line and len(line) >= 15 and not _is_form_line(line):
                    scored_all.append((line, loc, chunk))
        if not scored_all:
            return ("I couldn't find this information in the uploaded documents.", [])
        lines = [item[0] for item in scored_all[:20]]
        emb = embedding_service.generate(lines)
        best = [(cosine_similarity(query_embedding, emb[i])
                 + _line_pattern_bonus(intents, lines[i]), lines[i],
                 scored_all[i][1], scored_all[i][2])
                for i in range(len(lines))]
        best.sort(key=lambda x: x[0], reverse=True)
        if best[0][0] < 0.30:
            return ("I couldn't find this information in the uploaded documents.", [])
        best_line = _trim_trailing_clauses(best[0][1])
        answer = f"Based on your document:\n\n• {best_line} *({best[0][2]})*"
        return answer, [best[0][3]]

    return "Based on your document:\n\n" + "\n".join(selected), used_chunks


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _strip_sources_section(answer: str) -> str:
    lines = answer.splitlines()
    out = []
    for line in lines:
        if re.match(r"^\s*sources?\s*:?\s*$", line, re.IGNORECASE):
            break
        out.append(line)
    return "\n".join(out).strip()


def _call_openai(messages):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=messages
    )
    return response.choices[0].message.content.strip()


def _call_openrouter(messages):
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        temperature=0,
        messages=messages
    )
    return response.choices[0].message.content.strip()


def _build_history_text(conversation_history):
    recent = conversation_history[-MAX_HISTORY_MESSAGES:]
    return "\n".join(
        f'{message.get("role", "user")}: {message.get("content", "")}'
        for message in recent
    )


def _augment_query(question: str, conversation_history) -> str:
    """Resolve short anaphoric follow-ups like "What day is it?".

    When the current question refers back to a previous turn, the
    previous user question is folded in so retrieval finds the intended
    content even without an LLM. Chat answers are not modified.
    """
    if not conversation_history:
        return question

    lower = question.lower()
    pronouns = [
        r"\bit\b", r"\bthis\b", r"\bthat\b",
        r"\bthey\b", r"\bthose\b", r"\bthem\b",
    ]
    is_anaphoric = (
        any(re.search(pattern, lower) for pattern in pronouns)
        or "what about" in lower
        or "what day" in lower
    )
    if not is_anaphoric:
        return question

    previous_user = None
    for message in conversation_history[-4:]:
        if message.get("role") == "user":
            previous_user = message.get("content", "")

    if not previous_user:
        return question

    return f"{previous_user} {question}"


# ---------------------------------------------------------------------------
# Conversational routing (no RAG for social/casual turns)
# ---------------------------------------------------------------------------

_CONVERSATIONAL_PATTERNS = [
    (re.compile(
        r"^(hi+|he+y+|hello+|hullo+|hiya+|yo+|howdy+|namaste+|sup)+$"),
     "Hello! How can I help you with the uploaded documents?"),
    (re.compile(
        r"^good (morning|afternoon|evening)( there)?$"),
     "Good morning/afternoon/evening! How can I help you with the documents?"),
    (re.compile(
        r"^(how are you|how are you doing|how r u|how are u|"
        r"howre (you|u)|hru|how is it going|how s it going|"
        r"whats up|what s up|how do you do)+$"),
     "I'm doing well, thank you! How can I help you with the uploaded documents?"),
    (re.compile(
        r"^(hi+|hey+|hello+|hiya+|yo+|howdy+|namaste+|"
        r"good (morning|afternoon|evening))[,; ]+(and |)"
        r"(how are you( doing)?|how r u|how are u|whats up|"
        r"what s up|hru|how is it going)*$"),
     "Hello! I'm doing well, thank you. How can I help you with the documents?"),
    (re.compile(
        r"^(thank(s| you| u| you so much| you very much|s a lot| you a lot)|"
        r"thx|ty|tq|ok(ay) thanks*|thanks for the help)+$"),
     "You're welcome! Let me know if you need anything else."),
    (re.compile(
        r"^(bye+|goodbye|good night|see you( later| soon| tomorrow)?|"
        r"cya|tata|talk to you later)+$"),
     "Goodbye! Feel free to come back if you have more questions."),
]


def _conversational_response(question: str):
    """Return a canned conversational reply when the question is pure
    small-talk (greeting / thanks / goodbye), otherwise None.

    The matchers are anchored on the WHOLE normalized question, so real
    document questions ("how are working hours handled today?") can never
    be swallowed by a casual-intent detector.
    """
    q = question.strip()
    if not q:
        return None
    plain = re.sub(r"[\u202f\u00a0]+", " ", q.lower())
    plain = re.sub(r"[^a-z0-9 ]+", "", plain)
    plain = re.sub(r"\s+", " ", plain).strip()
    if not plain or len(plain) > 48:
        return None
    for pattern, response in _CONVERSATIONAL_PATTERNS:
        if pattern.fullmatch(plain):
            return response
    return None


# ---------------------------------------------------------------------------
# Document-aware selection ("which document is being talked about?")
# ---------------------------------------------------------------------------

def _known_sources():
    """Every document the RAG layer knows about: indexed chunks + the
    (reconciled) database list. Order is stable (sorted)."""
    sources = {
        item.get("source")
        for item in vector_store.all_metadata()
        if item.get("source")
    }
    try:
        from app.database import list_documents
        for item in list_documents():
            sources.add(item["filename"])
    except Exception:
        pass
    return sorted(sources)


def _base_tokens(filename: str):
    base = re.sub(
        r"\.[a-z0-9]{1,5}$", "",
        filename.lower(),
    ).replace("_", " ").replace("-", " ")
    return re.findall(r"[a-z0-9]+", base)


def _extract_named_document(question: str):
    """Return the filename a question explicitly refers to.

    Matches the full filename / extension-less name first, then a
    single distinctive word of the filename ("policies", "holidays",
    "nds") that appears in exactly one document — so "show me holidays"
    focuses on company_holidays.txt and "summarize Company_Policies…"
    focuses on the policy PDF without ever hardcoding either name.
    """
    norm = _light_normalize(question or "")
    if not norm:
        return None
    sources = _known_sources()

    direct = []
    for src in sources:
        low = src.lower()
        low_spaced = low.replace("_", " ")
        if low in norm or low_spaced in norm:
            direct.append(src)
    if len(direct) == 1:
        return direct[0]

    token_counts = {}
    for src in sources:
        for token in set(_base_tokens(src)):
            token_counts[token] = token_counts.get(token, 0) + 1

    for src in sources:
        for token in _base_tokens(src):
            if (token_counts.get(token) == 1
                    and len(token) >= 5
                    and re.search(r"\b" + re.escape(token) + r"\b", norm)):
                return src
    return None


def _resolve_summary_target(question: str, conversation_history):
    """Pull the most-recently-referenced document from the conversation
    when the user says "this document" / "summarize this doc"."""
    named = _extract_named_document(question)
    if named:
        return named
    for message in reversed(conversation_history or []):
        if message.get("role") != "user":
            continue
        named = _extract_named_document(message.get("content", ""))
        if named:
            return named
    return None


def answer_question(question, conversation_history):
    conversational = _conversational_response(question)
    if conversational:
        return {
            "answer": conversational,
            "sources": []
        }

    augmented = _augment_query(question, conversation_history or [])
    is_summary = "document_summary" in detect_intents(
        normalize_query(augmented)
    )

    summary_target = None
    if is_summary:
        summary_target = _resolve_summary_target(augmented, conversation_history or [])
        all_sources = _known_sources()
        if len(all_sources) > 1 and summary_target is None:
            return {
                "answer": "Which document would you like me to summarize? "
                          "Try mentioning the filename (for example "
                          "company_holidays.txt or Company_Policies_25-26.pdf).",
                "sources": []
            }
        if summary_target is None and all_sources:
            summary_target = all_sources[0]

    retrieved = retrieve(
        question,
        conversation_history,
        is_summary=is_summary,
        summary_target=summary_target,
    )
    relevant_results = retrieved["accepted"]

    if not relevant_results:
        return {
            "answer": "I couldn't find this information in the uploaded documents.",
            "sources": []
        }

    if is_summary:
        summary = _build_summary_answer(relevant_results)
        if summary is not None:
            context = build_context(relevant_results)
            history_text = _build_history_text(conversation_history or [])

            user_prompt = f"""Conversation history (for resolving references like "it"):

{history_text or "(no previous messages)"}

Retrieved document context:

{context}

Current user question:

{question}

Write a short, coherent summary of the document based ONLY on the retrieved context above.
Use the retrieved sections; list the main topics the document covers in 5-8 bullets.
Do not invent sections or facts. If the context does not contain enough information, say so."""

            answer = None
            try:
                if OPENAI_API_KEY:
                    answer = _call_openai([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ])
            except Exception as error:
                logger.warning("Summary OpenAI call failed, trying OpenRouter: %s", error)
                answer = None

            if answer is None and OPENROUTER_API_KEY:
                try:
                    answer = _call_openrouter([
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ])
                except Exception as error:
                    logger.warning("Summary OpenRouter call failed, using heuristic: %s", error)
                    answer = None

            if answer is None:
                answer = summary
            else:
                answer = _strip_sources_section(answer)

            if answer.startswith("I couldn't find this information"):
                return {"answer": answer, "sources": []}

            sources = _sources_from_results(relevant_results)
            return {"answer": answer, "sources": sources}

    query_embedding = retrieved["query_embedding"]
    context = build_context(relevant_results)
    history_text = _build_history_text(conversation_history or [])

    user_prompt = f"""Conversation history (for resolving references like "it"):

{history_text or "(no previous messages)"}

Retrieved document context:

{context}

Current user question:

{question}

Answer ONLY using the retrieved document context.
- If the question is about dates, holidays or tabular data present in the context, answer with a Markdown table.
- If the relevant context is a list, answer with bullet points.
- For explanatory questions, answer with concise paragraphs.
- If the retrieved context does not contain the answer, say exactly:
  "I couldn't find this information in the uploaded documents."
"""

    answer = None

    try:
        if OPENAI_API_KEY:
            answer = _call_openai([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ])
    except Exception as error:
        logger.warning("OpenAI call failed, trying fallback: %s", error)
        answer = None

    if answer is None and OPENROUTER_API_KEY:
        try:
            answer = _call_openrouter([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ])
        except Exception as error:
            logger.warning("OpenRouter call failed, using heuristic extraction: %s", error)
            answer = None
    elif answer is None and not OPENROUTER_API_KEY:
        logger.warning("OpenAI failed and no OpenRouter key configured; using heuristic extraction")

    if answer is None:
        answer, used_chunks = extract_concise_answer(
            question, relevant_results, query_embedding,
            intents=retrieved["intents"])

        if answer.startswith("I couldn't find this information"):
            return {
                "answer": answer,
                "sources": []
            }

        return {
            "answer": answer,
            "sources": _sources_from_results(used_chunks or relevant_results)
        }

    answer = _strip_sources_section(answer)

    if answer.startswith("I couldn't find this information"):
        return {
            "answer": answer,
            "sources": []
        }

    return {
        "answer": answer,
        "sources": _sources_from_results(relevant_results)
    }


def _sources_from_results(results):
    sources = []
    seen = set()
    for result in results:
        src = result.get("source")
        if not src:
            continue
        key = (result.get("document_id") or src, result.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": src,
            "page": result.get("page"),
            "document_id": result.get("document_id") or document_id_for(src),
        })
    return sources