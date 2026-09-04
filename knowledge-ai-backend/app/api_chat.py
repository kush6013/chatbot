from fastapi import (
    APIRouter,
    HTTPException
)

from pydantic import BaseModel

from app.database import (
    create_conversation,
    update_conversation_title,
    add_message,
    get_messages,
    get_conversations,
    delete_conversation
)

from app.services.rag import (
    answer_question
)


router = APIRouter(
    prefix="/api/chat"
)


class ChatRequest(BaseModel):

    message: str

    conversation_id: str

    language: str = "en"

    model: str = "gemma"


@router.post("")
def chat(
    request: ChatRequest
):

    question = (
        request.message.strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    if len(question) > 2000:

        raise HTTPException(
            status_code=400,
            detail=(
                "Message cannot exceed "
                "2000 characters."
            )
        )

    conversation_id = (
        request.conversation_id.strip()
    )

    if not conversation_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "conversation_id is required."
            )
        )

    history = get_messages(
        conversation_id
    )

    create_conversation(
        conversation_id,
        question[:50]
    )

    try:

        result = answer_question(
            question,
            history
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "AI processing failed: "
                f"{str(error)}"
            )
        )

    add_message(
        conversation_id,
        "user",
        question
    )

    add_message(
        conversation_id,
        "assistant",
        result["answer"]
    )

    update_conversation_title(
        conversation_id,
        question[:50]
    )

    return {
        "success": True,
        "conversation_id": conversation_id,
        "answer": result["answer"],
        "sources": result["sources"]
    }


@router.get("/history")
def chat_history():

    return {
        "conversations":
            get_conversations()
    }


@router.get("/history/{conversation_id}")
def get_conversation_history(
    conversation_id: str
):

    return {
        "messages":
            get_messages(conversation_id, limit=50)
    }



@router.delete("/{conversation_id}")
def clear_chat(
    conversation_id: str
):

    delete_conversation(
        conversation_id
    )

    return {
        "success": True,
        "message": "Chat cleared."
    }
