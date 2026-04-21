# prompt_engine.py
# Student: Alex Yebesi | Index: 10022300128
# Purpose: Build prompts that inject retrieved context and control hallucination

import tiktoken

# ─────────────────────────────────────────────
# SECTION 1: PROMPT TEMPLATES
# ─────────────────────────────────────────────

# Template 1: Standard RAG prompt
TEMPLATE_STANDARD = """You are an AI assistant for Academic City University.
You help students and staff answer questions about Ghana Election Results and Ghana's 2025 Budget.

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
- Answer ONLY using the context provided above
- If the context does not contain enough information, say: "I don't have enough information to answer that."
- Do NOT make up facts or figures
- Be concise and factual
- Cite which source your answer comes from (Election Data or Budget Document)

USER QUESTION: {query}

ANSWER:"""


# Template 2: Detailed RAG prompt with stricter hallucination control
TEMPLATE_STRICT = """You are a precise AI assistant for Academic City University.
Your knowledge is LIMITED to the documents provided below. Do not use outside knowledge.

--- START OF RETRIEVED DOCUMENTS ---
{context}
--- END OF RETRIEVED DOCUMENTS ---

STRICT RULES:
1. Only use facts found in the documents above
2. If unsure, say exactly: "The provided documents do not contain this information."
3. Never guess or estimate figures
4. Always mention the source: [Election Data] or [Budget Document]
5. Keep answers under 150 words

QUESTION: {query}

ANSWER (based strictly on documents above):"""


# Template 3: Conversational prompt (less strict, more friendly)
TEMPLATE_CONVERSATIONAL = """Hi! I'm ACity AI Assistant, here to help with questions about 
Ghana Elections and Ghana's 2025 Budget Statement.

Here's what I found in my knowledge base:
{context}

Based on the above information, here's my answer to your question:
"{query}"

"""


# ─────────────────────────────────────────────
# SECTION 2: CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count how many tokens a text uses"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def build_context(retrieved_chunks, max_tokens=2000):
    """
    Build context string from retrieved chunks.
    
    WHY token management?
    - LLMs have a context window limit (e.g. 4096 tokens)
    - We must fit our context + prompt + answer within that limit
    - We rank chunks by similarity and truncate if needed
    
    STRATEGY: Add chunks from highest to lowest similarity
    until we hit the token limit
    """
    context_parts = []
    total_tokens = 0

    # Chunks are already ranked by similarity (rank 1 = best)
    for chunk in retrieved_chunks:
        chunk_text = f"[Source: {chunk['source']} | Score: {chunk['similarity_score']}]\n{chunk['text']}"
        chunk_tokens = count_tokens(chunk_text)

        if total_tokens + chunk_tokens > max_tokens:
            print(f"   ⚠️ Token limit reached. Stopping at {total_tokens} tokens.")
            break

        context_parts.append(chunk_text)
        total_tokens += chunk_tokens

    context = "\n\n---\n\n".join(context_parts)
    print(f"   📊 Context built: {len(context_parts)} chunks, {total_tokens} tokens")
    return context


# ─────────────────────────────────────────────
# SECTION 3: BUILD FINAL PROMPT
# ─────────────────────────────────────────────

def build_prompt(query, retrieved_chunks, template_name="standard"):
    """
    Combine query + context into a final prompt for the LLM.
    
    template_name options: "standard", "strict", "conversational"
    """
    print(f"\n📝 Building prompt using template: {template_name}")

    # Build context from chunks
    context = build_context(retrieved_chunks)

    # Select template
    templates = {
        "standard": TEMPLATE_STANDARD,
        "strict": TEMPLATE_STRICT,
        "conversational": TEMPLATE_CONVERSATIONAL
    }
    template = templates.get(template_name, TEMPLATE_STANDARD)

    # Fill in the template
    prompt = template.format(context=context, query=query)

    # Count total tokens
    total_tokens = count_tokens(prompt)
    print(f"   📊 Final prompt tokens: {total_tokens}")
    print(f"   📊 Prompt preview (first 200 chars):\n   {prompt[:200]}...")

    return prompt


# ─────────────────────────────────────────────
# SECTION 4: EXPERIMENT - COMPARE TEMPLATES
# ─────────────────────────────────────────────

def compare_templates(query, retrieved_chunks):
    """
    Build the same query with all 3 templates and show differences.
    Used for Part C experiments.
    """
    print("\n" + "="*60)
    print(f"TEMPLATE COMPARISON FOR: '{query}'")
    print("="*60)

    results = {}
    for name in ["standard", "strict", "conversational"]:
        print(f"\n--- Template: {name.upper()} ---")
        prompt = build_prompt(query, retrieved_chunks, template_name=name)
        results[name] = prompt

    return results


# ─────────────────────────────────────────────
# MAIN: TEST PROMPT BUILDING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Create fake chunks to test prompt building
    fake_chunks = [
        {
            "rank": 1,
            "source": "Ghana_Election_Result.csv",
            "similarity_score": 0.85,
            "text": "Year: 2020, Region: Greater Accra, Candidate: Nana Akufo Addo, Party: NPP, Votes: 1253179, Votes(%): 47.36%"
        },
        {
            "rank": 2,
            "source": "budget.pdf",
            "similarity_score": 0.72,
            "text": "Total Revenue and Grants for 2025 is projected at GHS 169.8 billion representing 18.4% of GDP."
        }
    ]

    query = "Who won the 2020 election in Accra?"

    # Test all templates
    compare_templates(query, fake_chunks)

    print("\n🎉 Prompt engine working!")