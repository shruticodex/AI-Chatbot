"""Pinecone retriever for Agentic AI knowledge base."""

from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

from src.config import Settings, get_settings


@dataclass
class RetrievedChunk:
    text: str
    page: int
    score: float
    source: str
    chunk_index: int


class PineconeRetriever:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.huggingface_embedding_model
        )
        pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = pc.Index(self.settings.pinecone_index_name)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.settings.top_k
        query_vector = self.embeddings.embed_query(query)

        results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
        )

        chunks: list[RetrievedChunk] = []
        for match in results.matches or []:
            metadata = match.metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=metadata.get("text", ""),
                    page=int(metadata.get("page", 0)),
                    score=float(match.score or 0.0),
                    source=metadata.get("source", "unknown"),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                )
            )
        return chunks
