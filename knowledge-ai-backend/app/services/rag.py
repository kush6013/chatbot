from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
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
You are Knowledge AI, an intelligent, document-powered company assistant.

You MUST answer questions using the provided document context.

Guidelines:
1. Handle minor user typos or missing spaces gracefully (e.g. "working hors" -> working hours, "internshipduration" -> internship duration, "doc" -> uploaded document).
2. If the user asks general questions like "can you read the doc", "what documents do you have?", or "summarize", confirm that you have read the uploaded document context and provide a helpful overview of the available document content.
3. If an exact detail is not explicitly stated in the context (e.g. exact number of months for internship duration), explain clearly what the document *does* specify regarding that topic rather than giving a rigid refusal.
4. If the user's question is completely unrelated or absent from the document context, politely inform the user: "I couldn't find this information in the uploaded documents."
5. Base all factual information strictly on the retrieved document context. Never invent policies or facts.
6. Provide clear, direct, well-structured, and helpful responses.
"""



def build_context(results):

    parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        source = result.get(
            "source",
            "Unknown document"
        )

        page = result.get(
            "page"
        )

        if page:

            location = (
                f"{source}, Page {page}"
            )

        else:

            location = source

        parts.append(
            f"""
SOURCE {number}
Document: {location}

Content:
{result["text"]}
"""
        )

    return "\n".join(parts)


def clean_query(query: str) -> str:
    cleaned = query.strip()
    replacements = {
        "working hors": "working hours",
        "work hors": "working hours",
        "office hrs": "office hours",
        "internshipduration": "internship duration",
        "probationperiod": "probation period",
        "summery": "summary",
    }
    lower_q = cleaned.lower()
    for typo, correction in replacements.items():
        if typo in lower_q:
            cleaned = lower_q.replace(typo, correction)
            break
    return cleaned


def answer_question(
    question,
    conversation_history
):

    normalized_question = clean_query(question)

    query_embedding = (
        embedding_service.generate(
            [normalized_question]
        )[0]
    )


    lower_q = question.lower()
    is_summary = any(
        term in lower_q for term in [
            "summary", "summery", "tell me about",
            "overview", "highlights", "policy", "policies",
            "document", "doc", "what is this"
        ]
    )

    k_to_fetch = max(TOP_K, 8) if is_summary else TOP_K

    results = vector_store.search(
        query_embedding,
        k_to_fetch
    )

    threshold = 0.10 if is_summary else SIMILARITY_THRESHOLD

    relevant_results = [
        result
        for result in results
        if result["score"] >= threshold
    ]

    if not relevant_results:

        return {
            "answer": (
                "I couldn't find this "
                "information in the "
                "uploaded documents."
            ),
            "sources": []
        }


    context = build_context(
        relevant_results
    )

    history_text = ""

    for message in (
        conversation_history[-6:]
    ):

        history_text += (
            f'{message["role"]}: '
            f'{message["content"]}\n'
        )

    prompt = f"""
Conversation history:

{history_text}

Retrieved document context:

{context}

Current user question:

{question}

Answer ONLY using the retrieved
document context.
"""

    client = None
    response = None

    # Try OpenAI first
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
        except Exception as error:
            print(f"OpenAI completion failed ({error}), trying OpenRouter fallback...")
 
            raise error

    if response is None:
        raise ValueError("AI completion failed: No active API key or model available.")

    answer = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


    sources = []

    seen = set()

    for result in relevant_results:

        key = (
            result.get("source"),
            result.get("page")
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": result.get(
                "source"
            ),
            "page": result.get(
                "page"
            )
        })

    return {
        "answer": answer,
        "sources": sources
    }
