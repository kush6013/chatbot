import re
import numpy as np
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD
)

from app.services.embeddings import (
    embedding_service
)

from app.services.vector_store import (
    vector_store
)


def get_openai_client():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not configured in .env.")
    return OpenAI(api_key=OPENAI_API_KEY)


SYSTEM_PROMPT = """
You are Knowledge AI, a helpful, precise document assistant.

Your task is to answer user questions using ONLY the provided document context.

Strict Response Guidelines:
1. Provide a balanced, precise answer: NOT too detailed (no wall of text), and NOT too sparse (do not omit key facts).
2. Format answers using 2 to 4 clear, well-structured bullet points.
3. Be direct. Avoid long, repetitive introductory or concluding filler sentences.
4. Base all factual information strictly on the retrieved document context.
5. If the exact answer is not in the documents, state clearly: "I couldn't find this information in the uploaded documents."
"""


def build_context(results):
    parts = []
    for number, result in enumerate(results, start=1):
        source = result.get("source", "Unknown document")
        page = result.get("page")
        location = f"{source}, Page {page}" if page else source
        parts.append(
            f"""
SOURCE {number}
Document: {location}

Content:
{result["text"]}
"""
        )
    return "\n".join(parts)


def is_noise_line(line: str) -> bool:
    line_str = line.strip()
    if len(line_str) < 10:
        return True
    line_lower = line_str.lower()
    if re.match(r'^[\s_–\-\*]+$', line_str):
        return True
    if re.match(r'^page \d+( of \d+)?$', line_lower):
        return True
    return False


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def extract_contact_info(relevant_results):
    for chunk in relevant_results:
        text = chunk.get("text", "")
        src = chunk.get("source", "Document")
        pg = chunk.get("page")
        loc = f"{src} (Page {pg})" if pg else src

        if re.search(r'tel:|email:|reachus@|www\.|phone:|address', text, re.IGNORECASE):
            phone_m = re.search(r'(?:tel|phone|contact)[:\s]*([\+\d\s-]+)', text, re.IGNORECASE)
            email_m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
            web_m = re.search(r'www\.\S+|https?://\S+', text)
            company_m = re.search(r'([A-Z0-9\s]{3,}\s(?:LIMITED|PVT|PRIVATE|INC|LLC|CORP|CORPORATION))', text)
            addr_m = re.search(r'((?:S\d+B?,\s*)?(?:Second|First|Third|\d+th|\d+st|\d+nd)?\s*Floor[^\n•\r]+?\d{6}|[A-Z0-9\s,–\-]{10,}\b(?:Road|Street|Square|Block|Nagar|Gwalior|Madhya Pradesh)\b[^\n•\r]*)', text, re.IGNORECASE)

            bullets = []
            if company_m:
                bullets.append(f"• **Company Name**: {company_m.group(1).strip()}")
            if phone_m and len(phone_m.group(1).strip()) > 5:
                bullets.append(f"• **Phone**: {phone_m.group(1).strip()}")
            if email_m:
                bullets.append(f"• **Email**: {email_m.group(0).strip()}")
            if web_m:
                bullets.append(f"• **Website**: {web_m.group(0).strip()}")
            if addr_m and len(addr_m.group(1).strip()) > 15:
                clean_addr = re.sub(r'^(?:com|www\.com)\s*', '', addr_m.group(1).strip(), flags=re.IGNORECASE)
                bullets.append(f"• **Address**: {clean_addr}")

            if bullets:
                return f"Based on your document ({loc}):\n\n" + "\n".join(bullets)
    return None


def extract_concise_answer(question: str, relevant_results: list, query_embedding: list = None) -> str:
    q_lower = question.lower()
    is_contact_query = any(k in q_lower for k in ["address", "contact", "phone", "email", "location", "detail", "details", "office"])
    
    if is_contact_query:
        contact_res = extract_contact_info(relevant_results)
        if contact_res:
            return contact_res

    candidate_lines = []
    seen_lines = set()

    for chunk in relevant_results:
        src = chunk.get("source", "Document")
        pg = chunk.get("page")
        loc = f"Page {pg}" if pg else src
        text = chunk.get("text", "")
        
        # Split on newlines, bullets, or sentence-ending periods
        raw_lines = re.split(r'[\n•\r]+|(?<=[a-zA-Z0-9])\.\s+(?=[A-Z0-9])', text)
        for l in raw_lines:
            line = l.strip()
            if not is_noise_line(line) and line not in seen_lines and len(line) > 12:
                seen_lines.add(line)
                candidate_lines.append((line, src, loc))

    if not candidate_lines:
        top_chunk = relevant_results[0]
        src = top_chunk.get("source", "Document")
        pg = top_chunk.get("page")
        loc = f"{src} (Page {pg})" if pg else src
        snippet = top_chunk.get("text", "").strip()[:300] + "..."
        return f"Based on your document ({loc}):\n\n\"{snippet}\""

    if query_embedding is None:
        query_embedding = embedding_service.generate([question])[0]

    # Generate embeddings for candidate sentences in batch using sentence-transformer
    sentence_texts = [item[0] for item in candidate_lines[:25]]
    sentence_embeddings = embedding_service.generate(sentence_texts)

    scored = []
    for idx, (line, src, loc) in enumerate(candidate_lines[:25]):
        sim = cosine_similarity(query_embedding, sentence_embeddings[idx])
        scored.append((sim, line, src, loc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_selected = scored[:3]
    formatted = [f"• {item[1]} *({item[2]}, {item[3]})*" for item in top_selected if item[0] > 0.15]
    
    if formatted:
        return "Based on your document:\n\n" + "\n".join(formatted)

    top_line = candidate_lines[0]
    return f"Based on your document:\n\n• {top_line[0]} *({top_line[1]}, {top_line[2]})*"


def answer_question(question, conversation_history):
    normalized_question = question.strip()
    query_embedding = embedding_service.generate([normalized_question])[0]

    lower_q = normalized_question.lower()
    is_summary = any(
        term in lower_q for term in [
            "summary", "overview", "highlights", "tell me about",
            "document", "doc", "what is this"
        ]
    )

    k_to_fetch = max(TOP_K, 8) if is_summary else 6
    results = vector_store.search(query_embedding, k_to_fetch)
    threshold = 0.10 if is_summary else SIMILARITY_THRESHOLD

    relevant_results = [
        result for result in results
        if result["score"] >= threshold
    ]

    if not relevant_results:
        return {
            "answer": "I couldn't find this information in the uploaded documents.",
            "sources": []
        }

    context = build_context(relevant_results)
    history_text = ""
    for message in conversation_history[-6:]:
        history_text += f'{message["role"]}: {message["content"]}\n'

    prompt = f"""
Conversation history:

{history_text}

Retrieved document context:

{context}

Current user question:

{question}

Answer ONLY using the retrieved document context. Provide a clear, balanced answer in 2-3 concise bullet points.
"""

    answer = None

    # 1. Try OpenAI first
    if OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception:
            pass

    # 2. Try OpenRouter fallback
    if answer is None and OPENROUTER_API_KEY:
        try:
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception:
            pass

    # 3. Fallback to dynamic semantic sentence extraction if LLM API is unavailable/exhausted
    if answer is None:
        answer = extract_concise_answer(question, relevant_results, query_embedding)

    sources = []
    seen = set()
    for result in relevant_results:
        key = (result.get("source"), result.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "source": result.get("source"),
            "page": result.get("page")
        })

    return {
        "answer": answer,
        "sources": sources
    }
