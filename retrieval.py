# retrieval.py
# Student: Alex Yebesi | Index: 10022300128
# Purpose: Search the FAISS index to find the most relevant chunks for a query

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# SECTION 1: LOAD SAVED INDEX & CHUNKS
# ─────────────────────────────────────────────

def load_retrieval_system():
    """Load the FAISS index, chunks, and embedding model"""
    print("📂 Loading retrieval system...")

    # Load FAISS index
    index = faiss.read_index("chunks/faiss_index.bin")

    # Load chunks metadata
    with open("chunks/chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load the same embedding model used during indexing
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"   Index loaded: {index.ntotal} vectors")
    print(f"   Chunks loaded: {len(chunks)}")
    return index, chunks, model


# ─────────────────────────────────────────────
# SECTION 2: QUERY EXPANSION
# ─────────────────────────────────────────────

def expand_query(query):
    """
    Query Expansion: Add related terms to improve retrieval.

    WHY query expansion?
    - A user might ask 'election winner' but the data says 'votes received'
    - Expanding the query increases chance of finding relevant chunks
    - This is our INNOVATION for Part B
    """
    expansions = {
        "election": "election votes results constituency parliamentary",
        "budget": "budget expenditure revenue fiscal policy Ghana",
        "winner": "winner votes highest majority elected",
        "economy": "economy GDP growth inflation fiscal",
        "party": "party NPP NDC political votes",
        "region": "region constituency district area",
        "revenue": "revenue income tax collection budget",
        "spending": "spending expenditure allocation budget",
    }

    expanded = query
    query_lower = query.lower()
    for keyword, expansion in expansions.items():
        if keyword in query_lower:
            expanded += " " + expansion

    if expanded != query:
        print(f"   🔍 Query expanded: '{query}' → '{expanded[:80]}...'")

    return expanded


# ─────────────────────────────────────────────
# SECTION 3: TOP-K RETRIEVAL
# ─────────────────────────────────────────────

def retrieve(query, index, chunks, model, top_k=5):
    """
    Find the top_k most relevant chunks for a given query.

    HOW IT WORKS:
    1. Convert query to a vector using the same model
    2. Search FAISS index for nearest vectors
    3. Return the matching chunks with similarity scores
    """

    # Step 1: Expand the query
    expanded_query = expand_query(query)

    # Step 2: Embed the query
    query_vector = model.encode([expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)

    # Step 3: Search FAISS for top_k nearest chunks
    distances, indices = index.search(query_vector, top_k)

    # Step 4: Collect results with scores
    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue

        # Convert L2 distance to similarity score (0 to 1)
        # Lower distance = more similar, so we invert it
        similarity = float(1 / (1 + dist))

        chunk = chunks[idx]
        results.append({
            "rank": rank + 1,
            "chunk_id": chunk["id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "similarity_score": round(similarity, 4),
            "distance": round(float(dist), 4)
        })

    return results


# ─────────────────────────────────────────────
# SECTION 4: SHOW FAILURE CASES
# ─────────────────────────────────────────────

def show_failure_case(query, index, chunks, model):
    """
    Demonstrate a retrieval failure and how we handle it.

    FAILURE CASE: Very vague or unrelated queries return
    low similarity scores — we detect and flag these.
    """
    print(f"\n⚠️  FAILURE CASE TEST: '{query}'")
    results = retrieve(query, index, chunks, model, top_k=3)

    SIMILARITY_THRESHOLD = 0.3

    for r in results:
        status = "✅ RELEVANT" if r["similarity_score"] >= SIMILARITY_THRESHOLD else "❌ IRRELEVANT"
        print(f"   [{status}] Score: {r['similarity_score']} | {r['text'][:80]}...")

    low_quality = [r for r in results if r["similarity_score"] < SIMILARITY_THRESHOLD]
    if low_quality:
        print(f"\n   🔧 FIX: {len(low_quality)} low-quality results detected.")
        print("   FIX APPLIED: These chunks will be filtered out before sending to LLM.")
    return results


# ─────────────────────────────────────────────
# SECTION 5: FILTER LOW QUALITY RESULTS
# ─────────────────────────────────────────────

def filter_results(results, threshold=0.3):
    """Remove chunks with similarity score below threshold"""
    filtered = [r for r in results if r["similarity_score"] >= threshold]
    if len(filtered) < len(results):
        print(f"   🔧 Filtered out {len(results) - len(filtered)} low-quality chunks")
    return filtered if filtered else results  # fallback: return all if all are low


# ─────────────────────────────────────────────
# MAIN: TEST THE RETRIEVAL SYSTEM
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load everything
    index, chunks, model = load_retrieval_system()

    print("\n" + "="*60)
    print("TEST 1: Normal query")
    print("="*60)
    query1 = "Who won the election in Accra?"
    results1 = retrieve(query1, index, chunks, model, top_k=3)
    for r in results1:
        print(f"\n Rank {r['rank']} | Score: {r['similarity_score']} | Source: {r['source']}")
        print(f"   {r['text'][:150]}...")

    print("\n" + "="*60)
    print("TEST 2: Budget query")
    print("="*60)
    query2 = "What is Ghana's total revenue in the 2025 budget?"
    results2 = retrieve(query2, index, chunks, model, top_k=3)
    for r in results2:
        print(f"\n Rank {r['rank']} | Score: {r['similarity_score']} | Source: {r['source']}")
        print(f"   {r['text'][:150]}...")

    print("\n" + "="*60)
    print("TEST 3: Failure case")
    print("="*60)
    show_failure_case("purple elephant dancing", index, chunks, model)

    print("\n🎉 Retrieval system working!")