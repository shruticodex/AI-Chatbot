"""FastAPI chat endpoint for the Agentic AI RAG chatbot."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.graph import run_rag_query


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["What is Agentic AI?"])


class ContextChunk(BaseModel):
    text: str
    page: int
    score: float
    source: str
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    context_chunks: list[ContextChunk]
    confidence_score: float
    grounded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description=(
        "RAG chatbot grounded in the Konverge AI Agentic AI eBook. "
        "Built with LangGraph, Pinecone, and OpenAI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = run_rag_query(request.question.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(**result)


@app.get("/")
def root() -> dict:
    return {
        "message": "Agentic AI RAG Chatbot API",
        "docs": "/docs",
        "chat_endpoint": "POST /chat",
    }
