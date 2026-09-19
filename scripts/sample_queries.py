"""Run sample queries against the RAG chatbot (requires ingested index)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.graph import run_rag_query  # noqa: E402

QUERIES = [
    "What is Agentic AI?",
    "How do AI agents differ from traditional AI systems?",
    "What are the key components of an agentic AI system?",
    "What business benefits does Agentic AI provide?",
    "What is the role of LLMs in agentic systems?",
    "How does Konverge AI approach decision science?",
]


def main() -> None:
    for i, question in enumerate(QUERIES, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}: {question}")
        print("=" * 60)
        result = run_rag_query(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence_score']:.2%}")
        print(f"Grounded: {result['grounded']}")
        print(f"Retrieved {len(result['context_chunks'])} chunks")
        for j, chunk in enumerate(result["context_chunks"][:2], 1):
            print(f"  Chunk {j} (page {chunk['page']}, score {chunk['score']:.3f}):")
            print(f"    {chunk['text'][:150]}...")


if __name__ == "__main__":
    main()
