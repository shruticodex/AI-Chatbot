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

# Configure page settings
st.set_page_config(
    page_title="Agentic AI RAG | Candidate Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
    <style>
    .stChatFloatingInputContainer { padding-bottom: 20px; }
    .metric-container { background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/11696/11696092.png", width=60) # Sleek AI icon
    st.title("Agentic AI Explorer")
    st.caption("Developed as an AI Engineering Take-Home Project")
    
    st.divider()
    
    st.markdown("### 🎯 Try Sample Queries")
    for query in SAMPLE_QUERIES:
        if st.button(query, use_container_width=True, key=f"sample_{query[:15]}"):
            st.session_state["pending_query"] = query

    st.divider()
    
    st.markdown("### 🛠️ System Specs")
    st.markdown(
        """
        - **LLM:** NVIDIA NIM (`gpt-oss-20b`)
        - **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
        - **Vector DB:** Pinecone (Serverless)
        - **Orchestration:** LangGraph
        """
    )
    st.caption("Architecture: PDF → Chunk → Embed (CPU) → Pinecone → LangGraph → LLM")

# --- MAIN CHAT AREA ---
st.title("🧠 Agentic AI RAG Chatbot")
st.markdown("Interact with the **Konverge AI Agentic AI eBook**. This system features strict grounding constraints and retrieval transparency.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a welcoming initial message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am a RAG AI designed to answer questions strictly based on the *Agentic AI* eBook. Select a sample query from the sidebar or type your own question below!"
    })

# Display chat messages
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        # Display meta-data nicely for assistant responses
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            
            # Use Streamlit columns for visual metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Retrieval Confidence", value=f"{meta['confidence_score']:.1%}")
            with col2:
                grounded_status = "✅ Verified" if meta['grounded'] else "⚠️ Unverified"
                st.metric(label="Context Grounding", value=grounded_status)
            
            # Expander for raw text chunks
            with st.expander(f"🔍 View {len(meta['context_chunks'])} Retrieved Context Chunks"):
                for i, chunk in enumerate(meta["context_chunks"], 1):
                    st.markdown(f"**Chunk {i}** (Page {chunk['page']}) — *Similarity Score: {chunk['score']:.3f}*")
                    st.info(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))

# Handle user input
prompt = st.session_state.pop("pending_query", None) or st.chat_input("Ask about Agentic AI...")

if prompt:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analyzing document and generating response..."):
            try:
                result = run_rag_query(prompt)
                
                # Display the main answer
                st.markdown(result["answer"])
                
                # Display metrics visually
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Retrieval Confidence", value=f"{result['confidence_score']:.1%}")
                with col2:
                    grounded_status = "✅ Verified" if result['grounded'] else "⚠️ Unverified"
                    st.metric(label="Context Grounding", value=grounded_status)
                
                # Display chunks in an expander using st.info for better styling
                with st.expander(f"🔍 View {len(result['context_chunks'])} Retrieved Context Chunks"):
                    for i, chunk in enumerate(result["context_chunks"], 1):
                        st.markdown(f"**Chunk {i}** (Page {chunk['page']}) — *Similarity Score: {chunk['score']:.3f}*")
                        st.info(chunk["text"][:600] + ("..." if len(chunk["text"]) > 600 else ""))
                
                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "meta": result,
                })
                
            except Exception as exc:
                st.error(f"System Error: {exc}")
                st.warning("Please ensure your API keys in the `.env` file are correct and the server is running.")
