"""
PDF ingestion pipeline: load → chunk → embed → store in Pinecone.
Run once before starting the chatbot: python -m src.ingest <path-to-pdf>
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings


def _chunk_id(text: str, page: int, chunk_index: int) -> str:
    digest = hashlib.md5(f"{page}:{chunk_index}:{text[:120]}".encode()).hexdigest()
    return f"chunk-{digest[:16]}"


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status.ready:
            time.sleep(1)


def ingest_pdf(pdf_path: str | Path) -> int:
    settings = get_settings()
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loading PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.huggingface_embedding_model
    )

    pc = Pinecone(api_key=settings.pinecone_api_key)
    ensure_index(pc, settings.pinecone_index_name, settings.embedding_dimension)
    index = pc.Index(settings.pinecone_index_name)

    batch_size = 100
    total = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)

        upsert_records = []
        for i, (doc, vector) in enumerate(zip(batch, vectors)):
            page = doc.metadata.get("page", 0)
            chunk_index = start + i
            upsert_records.append(
                {
                    "id": _chunk_id(doc.page_content, page, chunk_index),
                    "values": vector,
                    "metadata": {
                        "text": doc.page_content,
                        "page": page,
                        "source": pdf_path.name,
                        "chunk_index": chunk_index,
                    },
                }
            )

        index.upsert(vectors=upsert_records)
        total += len(upsert_records)
        print(f"Upserted {total}/{len(chunks)} chunks")

    stats = index.describe_index_stats()
    print(f"Ingestion complete. Index total vectors: {stats.total_vector_count}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Agentic AI PDF into Pinecone")
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default=r"Ebook-Agentic-AI.pdf",
        help="Path to the Agentic AI eBook PDF",
    )
    args = parser.parse_args()

    try:
        ingest_pdf(args.pdf_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
