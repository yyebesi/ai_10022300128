# chunker.py
# Student: [Your Name] | Index: 10022300128
# Purpose: Load, clean, and chunk data from CSV and PDF sources

import pandas as pd
import fitz  # PyMuPDF - for reading PDFs
import json
import os
import re

# ─────────────────────────────────────────────
# SECTION 1: CLEAN & CHUNK THE CSV FILE
# ─────────────────────────────────────────────

def load_and_clean_csv(filepath="data/Ghana_Election_Result.csv"):
    """Load the election CSV and clean it"""
    print("📂 Loading CSV file...")
    df = pd.read_csv(filepath)

    print(f"   Original shape: {df.shape}")

    # Step 1: Drop rows where ALL values are missing
    df.dropna(how="all", inplace=True)

    # Step 2: Fill remaining missing values with "Unknown"
    df.fillna("Unknown", inplace=True)

    # Step 3: Strip extra whitespace from text columns
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Step 4: Remove duplicate rows
    df.drop_duplicates(inplace=True)

    print(f"   Cleaned shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def chunk_csv(df, chunk_size=5, overlap=1):
    """
    Convert CSV rows into text chunks.
    
    WHY chunk_size=5?
    - Each row is one election record. Grouping 5 rows gives enough 
      context for comparison questions (e.g. 'who won in Accra?')
    - Too large = irrelevant info mixed in. Too small = no context.
    
    WHY overlap=1?
    - 1 row overlap ensures continuity between chunks so we don't 
      lose a record that falls on a boundary.
    """
    print(f"\n📦 Chunking CSV with size={chunk_size}, overlap={overlap}...")
    chunks = []
    rows = df.to_dict(orient="records")
    step = chunk_size - overlap

    for i in range(0, len(rows), step):
        batch = rows[i: i + chunk_size]
        # Convert each row to readable text
        text_lines = []
        for row in batch:
            line = ", ".join([f"{k}: {v}" for k, v in row.items()])
            text_lines.append(line)
        chunk_text = "\n".join(text_lines)

        chunks.append({
            "id": f"csv_chunk_{i}",
            "source": "Ghana_Election_Result.csv",
            "text": chunk_text
        })

    print(f"   Total CSV chunks created: {len(chunks)}")
    return chunks


# ─────────────────────────────────────────────
# SECTION 2: CLEAN & CHUNK THE PDF FILE
# ─────────────────────────────────────────────

def load_and_clean_pdf(filepath="data/budget.pdf"):
    """Extract and clean text from the PDF"""
    print("\n📂 Loading PDF file...")
    doc = fitz.open(filepath)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[Page {page_num + 1}]\n{text}"

    # Clean up: remove excessive whitespace and blank lines
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r'[ \t]+', ' ', full_text)

    print(f"   Total characters extracted: {len(full_text)}")
    return full_text


def chunk_pdf(text, chunk_size=800, overlap=100):
    """
    Split PDF text into overlapping character chunks.
    
    WHY chunk_size=800 characters?
    - Budget document has dense paragraphs. 800 chars ≈ 1 paragraph.
    - Keeps each chunk meaningful and not too long for the embedding model.
    
    WHY overlap=100?
    - Sentences near chunk boundaries are preserved in both chunks,
      preventing important info from being cut off mid-sentence.
    """
    print(f"\n📦 Chunking PDF with size={chunk_size}, overlap={overlap}...")
    chunks = []
    step = chunk_size - overlap
    i = 0
    chunk_index = 0

    while i < len(text):
        chunk_text = text[i: i + chunk_size]
        chunks.append({
            "id": f"pdf_chunk_{chunk_index}",
            "source": "budget.pdf",
            "text": chunk_text
        })
        i += step
        chunk_index += 1

    print(f"   Total PDF chunks created: {len(chunks)}")
    return chunks


# ─────────────────────────────────────────────
# SECTION 3: SAVE ALL CHUNKS TO FILE
# ─────────────────────────────────────────────

def save_chunks(chunks, output_path="chunks/all_chunks.json"):
    """Save all chunks to a JSON file for later use"""
    os.makedirs("chunks", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n✅ All chunks saved to: {output_path}")
    print(f"   Total chunks: {len(chunks)}")


# ─────────────────────────────────────────────
# MAIN: RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Process CSV
    df = load_and_clean_csv()
    csv_chunks = chunk_csv(df)

    # Process PDF
    pdf_text = load_and_clean_pdf()
    pdf_chunks = chunk_pdf(pdf_text)

    # Combine and save
    all_chunks = csv_chunks + pdf_chunks
    save_chunks(all_chunks)

    print("\n🎉 Chunking complete! Ready for embedding.")
