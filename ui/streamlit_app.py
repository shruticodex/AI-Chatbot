"""Streamlit chat UI for the Agentic AI RAG chatbot."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.graph import run_rag_query  # noqa: E402

SAMPLE_QUERIES = [
    "What is Agentic AI?",
    "How do AI agents differ from traditional AI systems?",
    "What are the key components of an agentic AI system?",
    "What business benefits does Agentic AI provide?",
    "What is the role of LLMs in agentic systems?",
    "How does Konverge AI approach decision science?",
]

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("Agentic AI RAG Chatbot")
st.caption("Answers grounded in the Konverge AI Agentic AI eBook · LangGraph + Pinecone")

with st.sidebar:
    st.header("Sample Queries")
    for query in SAMPLE_QUERIES:
        if st.button(query, use_container_width=True, key=f"sample_{query[:20]}"):
            st.session_state["pending_query"] = query

    st.divider()
    st.markdown(
        "**Architecture:** PDF → Chunk → Embed → Pinecone → LangGraph RAG → LLM"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            with st.expander(f"Retrieved Context ({len(meta['context_chunks'])} chunks)"):
                for i, chunk in enumerate(meta["context_chunks"], 1):
                    st.markdown(
                        f"**Chunk {i}** · Page {chunk['page']} · Score: `{chunk['score']:.3f}`"
                    )
                    st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
            st.caption(
                f"Confidence: **{meta['confidence_score']:.2%}** · "
                f"Grounded: **{'Yes' if meta['grounded'] else 'No'}**"
            )

prompt = st.session_state.pop("pending_query", None) or st.chat_input("Ask about Agentic AI...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = run_rag_query(prompt)
                st.markdown(result["answer"])
                with st.expander(f"Retrieved Context ({len(result['context_chunks'])} chunks)"):
                    for i, chunk in enumerate(result["context_chunks"], 1):
                        st.markdown(
                            f"**Chunk {i}** · Page {chunk['page']} · Score: `{chunk['score']:.3f}`"
                        )
                        st.text(
                            chunk["text"][:500]
                            + ("..." if len(chunk["text"]) > 500 else "")
                        )
                st.caption(
                    f"Confidence: **{result['confidence_score']:.2%}** · "
                    f"Grounded: **{'Yes' if result['grounded'] else 'No'}**"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "meta": result,
                    }
                )
            except Exception as exc:
                st.error(f"Error: {exc}")
