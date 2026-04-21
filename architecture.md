# RAG System Architecture
## Student: Alex Yebesi | Index: 10022300128

## Overview
This RAG system has two phases:

**Phase 1 — Data Preparation (runs once):**
Election CSV + Budget PDF → Chunker → Embedder (all-MiniLM-L6-v2) → FAISS Vector Index

**Phase 2 — Query Pipeline (runs on every question):**
User Query → Query Expansion → FAISS Retrieval → Score Filter (threshold=0.3) → Prompt Engine → Groq LLM → Response

## Component Justifications

**Chunker:** CSV rows grouped in 5s (overlap=1) for election context. PDF split at 800 chars (overlap=100) to preserve paragraph meaning.

**Embedder:** all-MiniLM-L6-v2 chosen because it is free, fast, runs locally, and produces 384-dimensional vectors suitable for semantic search.

**FAISS:** Facebook AI Similarity Search chosen for fast exact L2 nearest-neighbour search. Scales well for our 1,214 chunk dataset.

**Query Expansion:** Adds domain keywords to short queries to improve retrieval recall.

**Score Filter:** Removes chunks below 0.3 similarity to reduce noise sent to the LLM.

**Prompt Engine:** Three templates (strict, standard, conversational) with token management to fit within LLM context window.

**Groq LLM:** llama-3.3-70b-versatile used via free Groq API for fast, accurate response generation.

**Logger:** Every interaction logged to JSON for experiment tracking and manual review.

**Memory (Part G):** Conversation history stored and injected into prompts for context-aware multi-turn chat.