# Agentic AI RAG Chatbot

A RAG-based AI chatbot built with **Python**, **LangGraph**, **Pinecone**, and **OpenAI embeddings**. It answers questions strictly grounded in the Konverge AI eBook: [Agentic AI PDF](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  PDF eBook  │────▶│  Chunk Text  │────▶│  Embeddings │────▶│   Pinecone   │
│  (PyPDF)    │     │  (1000 tok)  │     │  (OpenAI)   │     │  Vector DB   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                    │
User Question ──────────────────────────────────────────────────────┤
                                                                    ▼
                                                          ┌─────────────────┐
                                                          │   LangGraph     │
                                                          │  RAG Pipeline   │
                                                          └────────┬────────┘
                                                                   │
                    ┌──────────────────────────────────────────────┼──────────────────────────┐
                    │                                              │                          │
                    ▼                                              ▼                          ▼
            ┌───────────────┐                              ┌──────────────┐          ┌──────────────┐
            │  1. Retrieve  │                              │ 2. Generate  │          │ 3. Grounding │
            │  Top-K chunks │─────────────────────────────▶│  LLM Answer  │─────────▶│    Check     │
            │  from Pinecone│                              │  (GPT-4o-mini)│         │              │
            └───────────────┘                              └──────────────┘          └──────────────┘
                                                                   │
                                                                   ▼
                                                          ┌─────────────────┐
                                                          │  API Response   │
                                                          │  • answer       │
                                                          │  • context      │
                                                          │  • confidence   │
                                                          └─────────────────┘
```

### LangGraph Pipeline

The RAG workflow is modeled as a **StateGraph** with two nodes:

1. **Retrieve** — Embeds the user question, queries Pinecone for top-K similar chunks, and computes a confidence score from retrieval similarity.
2. **Generate** — Passes retrieved context to the LLM with strict grounding instructions. Refuses to answer when context is insufficient.

### Grounding Strategy

- System prompt restricts answers to retrieved context only.
- Low-confidence retrievals trigger explicit "cannot answer" responses.
- Each response includes retrieved chunks with similarity scores for transparency.

## Project Structure

```
RAG Chatbot/
├── src/
│   ├── config.py          # Environment-based settings
│   ├── ingest.py          # PDF → chunk → embed → Pinecone
│   ├── rag/
│   │   ├── retriever.py   # Pinecone similarity search
│   │   └── graph.py       # LangGraph RAG pipeline
│   └── api/
│       └── main.py        # FastAPI chat endpoint
├── ui/
│   └── streamlit_app.py   # Optional Streamlit chat UI
├── scripts/
│   ├── run_api.py         # Start FastAPI server
│   └── sample_queries.py  # Run 6 sample queries
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://app.pinecone.io/) (free tier works)

### 2. Install Dependencies

```bash
cd "RAG Chatbot"
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` with your keys:

```env
OPENAI_API_KEY=sk-your-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=agentic-ai-ebook
```

### 4. Ingest the PDF

Download the eBook from [https://konverge.ai/pdf/Ebook-Agentic-AI.pdf](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf), then run:

```bash
python -m src.ingest "C:\path\to\Ebook-Agentic-AI.pdf"
```

This creates a Pinecone index (if needed), chunks the PDF, generates embeddings, and upserts vectors.

### 5. Run the Chatbot

**Option A — FastAPI (recommended for API integration):**

```bash
python scripts/run_api.py
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API docs.

**Option B — Streamlit UI:**

```bash
streamlit run ui/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

## API Usage

**POST** `/chat`

Request:

```json
{
  "question": "What is Agentic AI?"
}
```

Response:

```json
{
  "answer": "Agentic AI refers to...",
  "context_chunks": [
    {
      "text": "...",
      "page": 3,
      "score": 0.8521,
      "source": "Ebook-Agentic-AI.pdf",
      "chunk_index": 12
    }
  ],
  "confidence_score": 0.8234,
  "grounded": true
}
```

Example with curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is Agentic AI?\"}"
```

## Sample Queries

Run all sample queries:

```bash
python scripts/sample_queries.py
```

| # | Query |
|---|-------|
| 1 | What is Agentic AI? |
| 2 | How do AI agents differ from traditional AI systems? |
| 3 | What are the key components of an agentic AI system? |
| 4 | What business benefits does Agentic AI provide? |
| 5 | What is the role of LLMs in agentic systems? |
| 6 | How does Konverge AI approach decision science? |

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| Vector DB | Pinecone |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| PDF Parsing | PyPDF |
| API | FastAPI + Uvicorn |
| UI | Streamlit |

## License

MIT — built as an interview task demonstration.
