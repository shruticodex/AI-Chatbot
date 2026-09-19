"""
LangGraph RAG pipeline:
  retrieve → generate → validate grounding
"""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.rag.retriever import PineconeRetriever, RetrievedChunk


class RAGState(TypedDict):
    question: str
    chunks: list[RetrievedChunk]
    context: str
    answer: str
    confidence: float
    grounded: bool


SYSTEM_PROMPT = """You are a helpful assistant that answers questions STRICTLY based on the provided context from the "Agentic AI" eBook by Konverge AI.

Rules:
1. Answer ONLY using information from the context below.
2. If the context does not contain enough information, say: "I cannot answer this question based on the Agentic AI eBook. The retrieved context does not contain relevant information."
3. Do NOT use outside knowledge or make up facts.
4. Be concise and accurate. Cite page numbers when available (e.g., "According to page 5...").
5. Use bullet points for lists when appropriate.

Context:
{context}
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Chunk {i} | Page {chunk.page + 1} | Score: {chunk.score:.3f}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    scores = [c.score for c in chunks]
    top = scores[0]
    avg = sum(scores) / len(scores)
    return round(min(1.0, 0.6 * top + 0.4 * avg), 4)


def _is_grounded(answer: str, chunks: list[RetrievedChunk]) -> bool:
    refusal_phrases = [
        "cannot answer",
        "does not contain",
        "not contain relevant",
        "no relevant information",
    ]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        return True
    if not chunks:
        return False
    return chunks[0].score >= 0.3


def build_retrieve_node(retriever: PineconeRetriever):
    def retrieve(state: RAGState) -> dict:
        chunks = retriever.retrieve(state["question"])
        context = _format_context(chunks)
        confidence = _compute_confidence(chunks)
        return {"chunks": chunks, "context": context, "confidence": confidence}

    return retrieve


def build_generate_node(llm: ChatNVIDIA):
    def generate(state: RAGState) -> dict:
        if not state["chunks"]:
            return {
                "answer": (
                    "I cannot answer this question based on the Agentic AI eBook. "
                    "No relevant context was retrieved from the knowledge base."
                ),
                "grounded": True,
                "confidence": 0.0,
            }

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=state["context"])),
            HumanMessage(content=state["question"]),
        ]
        response = llm.invoke(messages)
        answer_text = response.content if isinstance(response.content, str) else str(response.content)
        answer = answer_text.strip()
        grounded = _is_grounded(answer, state["chunks"])
        return {"answer": answer, "grounded": grounded}

    return generate


def build_rag_graph(retriever: PineconeRetriever | None = None):
    settings = get_settings()
    retriever = retriever or PineconeRetriever(settings)
    
    llm = ChatNVIDIA(
        model=settings.nvidia_chat_model,
        temperature=0.0,
        nvidia_api_key=settings.nvidia_api_key,
    )

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", build_retrieve_node(retriever))
    graph.add_node("generate", build_generate_node(llm))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_rag_query(question: str, retriever: PineconeRetriever | None = None) -> dict:
    app = build_rag_graph(retriever)
    result = app.invoke({"question": question})

    return {
        "answer": result["answer"],
        "context_chunks": [
            {
                "text": c.text,
                "page": c.page + 1,
                "score": round(c.score, 4),
                "source": c.source,
                "chunk_index": c.chunk_index,
            }
            for c in result["chunks"]
        ],
        "confidence_score": result["confidence"],
        "grounded": result["grounded"],
    }