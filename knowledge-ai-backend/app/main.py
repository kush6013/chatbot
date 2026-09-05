from fastapi import FastAPI

from fastapi.middleware.cors import (
    CORSMiddleware
)

from app.database import (
    initialize_database
)

from app.api_documents import (
    router as documents_router
)

from app.api_chat import (
    router as chat_router
)


app = FastAPI(
    title="Knowledge AI API",
    description=(
        "Document-powered RAG chatbot API"
    ),
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]
)



initialize_database()


app.include_router(
    documents_router
)

app.include_router(
    chat_router
)


@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "service": "Knowledge AI"
    }


@app.get("/health")
def health_root():

    return {
        "status": "ok"
    }


@app.get("/")
def root():

    return {
        "service": "Knowledge AI",
        "docs": "/docs",
        "health": "/health"
    }
