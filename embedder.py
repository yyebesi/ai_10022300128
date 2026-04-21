# embedder.py
# Student: [Your Name] | Index: 10022300128
# Purpose: Convert text chunks into vector embeddings and store them using FAISS

import json
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# SECTION 1: LOAD CHUNKS
# ─────────────────────────────────────────────

def load_chunks(filepath="chunks/all_chunks.json"):
    """Load the chunks we created in chunker.py"""
    print("📂 Loading chunks...")
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"   Total chunks loaded: {len(chunks)}")
    return chunks


# ─────────────────────────────────────────────
# SECTION 2: CREATE EMBEDDINGS
# ─────────────────────────────────────────────

def create_embeddings(chunks):
    """
    Convert each chunk's text into a vector (list of numbers).
    
    WHY sentence-transformers?
    - It's free, runs locally, no API key needed for embeddings
    - 'all-MiniLM-L6-v2' is fast and accurate for semantic search
    - Each text becomes a 384-dimensional vector
    """
    print("\n🔢 Creating embeddings...")
    print("   Loading model: all-MiniLM-L6-v2 (first time may take a minute)...")
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"   Embedding {len(texts)} chunks...")
    
    # Convert texts to vectors in batches of 64
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Each chunk is now a vector of {embeddings.shape[1]} numbers")
    return embeddings, model


# ─────────────────────────────────────────────
# SECTION 3: BUILD FAISS INDEX
# ─────────────────────────────────────────────

def build_faiss_index(embeddings):
    """
    Store all vectors in a FAISS index for fast similarity search.
    
    WHY FAISS?
    - Created by Facebook AI, very fast even with thousands of vectors
    - IndexFlatL2 = exact search using L2 (Euclidean) distance
    - Perfect for our dataset size (1214 chunks)
    """
    print("\n🗄️ Building FAISS index...")
    
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatL2(dimension)
    
    # Normalize embeddings for better similarity scoring
    faiss.normalize_L2(embeddings)
    
    # Add all vectors to the index
    index.add(embeddings)
    
    print(f"   FAISS index built with {index.ntotal} vectors")
    print(f"   Vector dimension: {dimension}")
    return index


# ─────────────────────────────────────────────
# SECTION 4: SAVE EVERYTHING
# ─────────────────────────────────────────────

def save_index(index, chunks, embeddings):
    """Save the FAISS index and chunks for use in retrieval"""
    os.makedirs("chunks", exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, "chunks/faiss_index.bin")
    print("\n✅ FAISS index saved to: chunks/faiss_index.bin")
    
    # Save chunks metadata
    with open("chunks/chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print("✅ Chunks metadata saved to: chunks/chunks_metadata.pkl")
    
    # Save embeddings
    np.save("chunks/embeddings.npy", embeddings)
    print("✅ Embeddings saved to: chunks/embeddings.npy")


# ─────────────────────────────────────────────
# MAIN: RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load chunks
    chunks = load_chunks()
    
    # Create embeddings
    embeddings, model = create_embeddings(chunks)
    
    # Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Save everything
    save_index(index, chunks, embeddings)
    
    print("\n🎉 Embedding complete! Ready for retrieval.")