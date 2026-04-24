# app.py
# Student: Alex Yebesi | Index: 10022300128
# Purpose: Ghana-themed Streamlit chat UI with Memory-based RAG

import streamlit as st
import os
import json
import datetime
from dotenv import load_dotenv
from groq import Groq
from retrieval import load_retrieval_system, retrieve, filter_results
from main import ConversationMemory, build_memory_prompt

# ─────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ACity AI Assistant",
    page_icon="⭐",
    layout="wide"
)

# ─────────────────────────────────────────────
# GHANA THEME CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .ghana-bar {
        height: 6px;
        background: linear-gradient(to right, #c8221a 33%, #c8971e 33%, #c8971e 66%, #1a5c2e 66%);
        border-radius: 3px;
        margin-bottom: 1.2rem;
    }
    .ghana-header {
        background: #1a5c2e;
        border-radius: 12px;
        padding: 18px 22px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1rem;
    }
    .ghana-header h1 {
        color: #f5e4b0;
        font-size: 20px;
        font-weight: 500;
        margin: 0;
    }
    .ghana-header p {
        color: #9dc9a8;
        font-size: 13px;
        margin: 0;
    }
    .source-pills {
        display: flex;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .pill-election {
        background: #e8f5ed;
        color: #1a5c2e;
        border: 1px solid #9dc9a8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    .pill-budget {
        background: #f5e4b0;
        color: #7a5a0a;
        border: 1px solid #c8971e;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }
    .chunk-card {
        background: #f5e4b0;
        border-left: 4px solid #c8971e;
        border-radius: 0 8px 8px 0;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 12px;
        color: #7a5a0a;
    }
    .memory-card {
        background: #f0eafd;
        border-left: 4px solid #7F77DD;
        border-radius: 0 8px 8px 0;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 12px;
        color: #3C3489;
    }
    .ghana-footer {
        text-align: center;
        font-size: 11px;
        color: var(--color-text-tertiary);
        margin-top: 1.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid var(--color-border-tertiary);
    }
    .sidebar-label {
        font-size: 13px;
        font-weight: 500;
        color: var(--color-text-primary);
    }
    .sidebar-info {
        font-size: 12px;
        color: var(--color-text-secondary);
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD PIPELINE (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    # Load .env inside the cached function to guarantee it runs
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()

    index, chunks, model = load_retrieval_system()
    client = Groq(api_key=api_key)
    return index, chunks, model, client

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Akwaaba! 🇬🇭 I'm your ACity AI Assistant. Ask me anything about Ghana's election results or the 2025 Budget Statement! I also remember our conversation as we chat.",
            "chunks": []
        }
    ]

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=5)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 6px;'>
        <div style='font-size:36px;'>⭐</div>
        <div style='font-size:15px; font-weight:500; color:#1a5c2e;'>ACity AI Assistant</div>
        <div style='font-size:11px; color: var(--color-text-secondary);'>Powered by RAG + Groq + Memory</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="sidebar-label">⚙️ Settings</p>', unsafe_allow_html=True)

    top_k = st.slider("Chunks to retrieve", 1, 10, 5)
    threshold = st.slider("Similarity threshold", 0.1, 0.9, 0.3)
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    show_memory = st.toggle("Show memory context", value=False)

    st.divider()
    st.markdown('<p class="sidebar-label">🧠 Memory</p>', unsafe_allow_html=True)

    memory_turns = len(st.session_state.memory.turns)
    st.markdown(
        f'<div class="sidebar-info">Storing <b>{memory_turns}</b> conversation turn(s)</div>',
        unsafe_allow_html=True
    )

    if st.button("🗑️ Clear memory & chat"):
        st.session_state.memory.clear()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Memory cleared! Akwaaba again! 🇬🇭 Ask me anything.",
                "chunks": []
            }
        ]
        st.rerun()

    st.divider()
    st.markdown('<p class="sidebar-label">📚 Data Sources</p>', unsafe_allow_html=True)
    st.markdown(
        '<span class="pill-election">🗳️ Ghana Election Results</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<span class="pill-budget">💰 2025 Budget Statement</span>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("""
    <div class="sidebar-info">
        <b>Student:</b> Alex Yebesi<br>
        <b>Index:</b> 10022300128<br>
        <b>Course:</b> CS4241 - Intro to AI<br>
        <b>University:</b> Academic City
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.markdown('<div class="ghana-bar"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="ghana-header">
    <div style="font-size:28px;">⭐</div>
    <div>
        <h1>ACity AI Assistant</h1>
        <p>Akwaaba! Ask me about Ghana Elections & the 2025 Budget</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="source-pills">
    <span class="pill-election">🗳️ Ghana Election Results</span>
    <span class="pill-budget">💰 Ghana 2025 Budget Statement</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD PIPELINE
# ─────────────────────────────────────────────

with st.spinner("⭐ Loading knowledge base..."):
    index, chunks, embed_model, client = load_pipeline()

# ─────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and show_chunks and message.get("chunks"):
            with st.expander("📄 View retrieved chunks", expanded=False):
                for r in message["chunks"]:
                    score_color = "green" if r["similarity_score"] >= 0.5 else "orange"
                    st.markdown(
                        f"**Rank {r['rank']}** | "
                        f"`{r['source']}` | "
                        f"Score: :{score_color}[{r['similarity_score']}]"
                    )
                    st.markdown(
                        f'<div class="chunk-card">{r["text"][:300]}</div>',
                        unsafe_allow_html=True
                    )
                    st.divider()

# ─────────────────────────────────────────────
# HANDLE USER INPUT
# ─────────────────────────────────────────────

if query := st.chat_input("Ask about Ghana elections or the 2025 budget..."):

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({
        "role": "user", "content": query, "chunks": []
    })

    with st.chat_message("assistant"):
        with st.spinner("⭐ Searching knowledge base..."):

            # Stage 1: Retrieve
            retrieved = retrieve(query, index, chunks, embed_model, top_k=top_k)

            # Stage 2: Filter
            filtered = filter_results(retrieved, threshold=threshold)

            # Stage 3: Build memory-enhanced prompt
            prompt = build_memory_prompt(query, filtered, st.session_state.memory)

            # Stage 4: Call LLM
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.2
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"❌ Error: {str(e)}"

        # Display answer
        st.markdown(answer)

        # Show memory context if enabled
        if show_memory and st.session_state.memory.turns:
            with st.expander("🧠 Memory context used", expanded=False):
                st.markdown(
                    f'<div class="memory-card">{st.session_state.memory.get_context()}</div>',
                    unsafe_allow_html=True
                )

        # Show retrieved chunks
        if show_chunks and filtered:
            with st.expander("📄 View retrieved chunks", expanded=False):
                for r in filtered:
                    score_color = "green" if r["similarity_score"] >= 0.5 else "orange"
                    st.markdown(
                        f"**Rank {r['rank']}** | "
                        f"`{r['source']}` | "
                        f"Score: :{score_color}[{r['similarity_score']}]"
                    )
                    st.markdown(
                        f'<div class="chunk-card">{r["text"][:300]}</div>',
                        unsafe_allow_html=True
                    )
                    st.divider()

        # Save to memory
        st.session_state.memory.add_turn(
            query=query,
            response=answer,
            sources=list(set([r["source"] for r in filtered]))
        )

        # Log interaction
        os.makedirs("logs", exist_ok=True)
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "top_k": top_k,
            "chunks_retrieved": len(filtered),
            "memory_turns": len(st.session_state.memory.turns),
            "response": answer
        }
        logs = []
        if os.path.exists("logs/ui_log.json"):
            with open("logs/ui_log.json", "r") as f:
                try:
                    logs = json.load(f)
                except:
                    logs = []
        logs.append(log_entry)
        with open("logs/ui_log.json", "w") as f:
            json.dump(logs, f, indent=2)

    # Save to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": filtered
    })

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("""
<div class="ghana-footer">
    Academic City University &nbsp;|&nbsp; Alex Yebesi &nbsp;|&nbsp;
    Index: 10022300128 &nbsp;|&nbsp; CS4241 Introduction to AI &nbsp;|&nbsp; 2026
</div>
""", unsafe_allow_html=True)