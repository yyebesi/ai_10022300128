# pipeline.py
# Student: Alex Yebesi | Index: 10022300128
# Purpose: Full RAG pipeline - connects retrieval + prompt + LLM together

import os
import json
import datetime
from dotenv import load_dotenv
from groq import Groq
from retrieval import load_retrieval_system, retrieve, filter_results
from prompt_engine import build_prompt

# Load API key from .env file
load_dotenv()

# ─────────────────────────────────────────────
# SECTION 1: SETUP
# ─────────────────────────────────────────────

def setup_pipeline():
    """Load all components needed for the pipeline"""
    print("🚀 Setting up RAG pipeline...")
    
    # Load retrieval system
    index, chunks, model = load_retrieval_system()
    
    # Setup Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    print("✅ Pipeline ready!\n")
    return index, chunks, model, client


# ─────────────────────────────────────────────
# SECTION 2: LOGGING
# ─────────────────────────────────────────────

def log_interaction(query, retrieved, prompt, response, log_file="logs/pipeline_log.json"):
    """
    Log every interaction for experiment tracking.
    Required for Part D - logging at each stage.
    """
    os.makedirs("logs", exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "retrieved_chunks": [
            {
                "rank": r["rank"],
                "source": r["source"],
                "similarity_score": r["similarity_score"],
                "text_preview": r["text"][:100]
            }
            for r in retrieved
        ],
        "prompt_sent_to_llm": prompt,
        "llm_response": response
    }
    
    # Load existing logs
    logs = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    
    logs.append(log_entry)
    
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)


# ─────────────────────────────────────────────
# SECTION 3: CALL THE LLM
# ─────────────────────────────────────────────

def call_llm(prompt, client):
    """Send the prompt to Groq and get a response"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ LLM Error: {str(e)}"


# ─────────────────────────────────────────────
# SECTION 4: FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(query, index, chunks, model, client,
                 top_k=5, template="strict", show_details=True):
    """
    Full RAG Pipeline:
    User Query → Retrieval → Filter → Prompt → LLM → Response
    """
    
    print("\n" + "="*60)
    print(f"📥 USER QUERY: {query}")
    print("="*60)
    
    # ── STAGE 1: RETRIEVAL ──
    print("\n📡 STAGE 1: Retrieving relevant chunks...")
    retrieved = retrieve(query, index, chunks, model, top_k=top_k)
    
    if show_details:
        print(f"\n   Retrieved {len(retrieved)} chunks:")
        for r in retrieved:
            print(f"   [{r['rank']}] Score: {r['similarity_score']} | "
                  f"Source: {r['source']}")
            print(f"       Preview: {r['text'][:100]}...")
    
    # ── STAGE 2: FILTER ──
    print("\n🔧 STAGE 2: Filtering low-quality chunks...")
    filtered = filter_results(retrieved, threshold=0.3)
    print(f"   Kept {len(filtered)} of {len(retrieved)} chunks")
    
    # ── STAGE 3: PROMPT BUILDING ──
    print("\n📝 STAGE 3: Building prompt...")
    prompt = build_prompt(query, filtered, template_name=template)
    
    if show_details:
        print(f"\n   FULL PROMPT SENT TO LLM:")
        print("   " + "-"*50)
        print(prompt)
        print("   " + "-"*50)
    
    # ── STAGE 4: LLM GENERATION ──
    print("\n🤖 STAGE 4: Calling LLM...")
    response = call_llm(prompt, client)
    
    # ── STAGE 5: LOG ──
    log_interaction(query, filtered, prompt, response)
    print("   📋 Interaction logged.")
    
    # ── FINAL RESPONSE ──
    print("\n" + "="*60)
    print("💬 FINAL RESPONSE:")
    print("="*60)
    print(response)
    print("="*60)
    
    return {
        "query": query,
        "retrieved": filtered,
        "prompt": prompt,
        "response": response
    }


# ─────────────────────────────────────────────
# SECTION 5: ADVERSARIAL TESTING (Part E)
# ─────────────────────────────────────────────

def run_adversarial_tests(index, chunks, model, client):
    """
    Test the RAG system with tricky queries.
    Required for Part E.
    """
    print("\n\n" + "🔴"*30)
    print("ADVERSARIAL TESTING")
    print("🔴"*30)
    
    adversarial_queries = [
        # Query 1: Ambiguous
        "Who is the best?",
        # Query 2: Misleading
        "What was Ghana's budget in 1990?"
    ]
    
    for query in adversarial_queries:
        run_pipeline(query, index, chunks, model, client,
                    top_k=3, template="strict", show_details=False)


# ─────────────────────────────────────────────
# MAIN: RUN THE PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Setup
    index, chunks, model, client = setup_pipeline()
    
    # Test 1: Normal query
    run_pipeline(
        "Who won the 2020 presidential election in Ghana?",
        index, chunks, model, client,
        top_k=5, template="strict"
    )
    
    # Test 2: Budget query
    run_pipeline(
        "What is Ghana's total revenue in the 2025 budget?",
        index, chunks, model, client,
        top_k=5, template="standard"
    )
    
    # Test 3: Adversarial
    run_adversarial_tests(index, chunks, model, client)
    
    print("\n🎉 Full pipeline test complete!")
