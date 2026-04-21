# main.py
# Student: Alex Yebesi | Index: 10022300128
# Purpose: Memory-based RAG - stores conversation history for context-aware responses

import os
import json
import datetime

# ─────────────────────────────────────────────
# MEMORY SYSTEM
# ─────────────────────────────────────────────

class ConversationMemory:
    """
    INNOVATION - Part G: Memory-based RAG

    WHY MEMORY?
    - Standard RAG treats every question independently
    - With memory, the system remembers previous Q&A pairs
    - This allows follow-up questions like 'tell me more about that'
    - Makes the chatbot feel like a real conversation partner

    HOW IT WORKS:
    - Stores last N conversation turns in a list
    - Injects relevant history into the prompt
    - Saves memory to disk so it persists between sessions
    """

    def __init__(self, max_turns=5, memory_file="logs/memory.json"):
        self.max_turns = max_turns
        self.memory_file = memory_file
        self.turns = []
        self.load_memory()

    def add_turn(self, query, response, sources):
        """Add a new conversation turn to memory"""
        turn = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response,
            "sources": sources
        }
        self.turns.append(turn)

        # Keep only last max_turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        self.save_memory()

    def get_context(self):
        """
        Format memory as context string to inject into prompt.
        Returns last N turns as readable text.
        """
        if not self.turns:
            return ""

        lines = ["CONVERSATION HISTORY (for context):"]
        for i, turn in enumerate(self.turns):
            lines.append(f"Q{i+1}: {turn['query']}")
            lines.append(f"A{i+1}: {turn['response'][:200]}...")
            lines.append("")

        return "\n".join(lines)

    def save_memory(self):
        """Save memory to disk"""
        os.makedirs("logs", exist_ok=True)
        with open(self.memory_file, "w") as f:
            json.dump(self.turns, f, indent=2)

    def load_memory(self):
        """Load memory from disk if it exists"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    self.turns = json.load(f)
                print(f"   Memory loaded: {len(self.turns)} previous turns")
            except:
                self.turns = []
        else:
            self.turns = []

    def clear(self):
        """Clear all memory"""
        self.turns = []
        self.save_memory()
        print("   Memory cleared.")

    def summarize(self):
        """Print a summary of current memory"""
        print(f"\nMEMORY SUMMARY ({len(self.turns)} turns stored):")
        for i, turn in enumerate(self.turns):
            print(f"   Turn {i+1}: '{turn['query'][:60]}...'")


# ─────────────────────────────────────────────
# MEMORY-ENHANCED PROMPT BUILDER
# ─────────────────────────────────────────────

def build_memory_prompt(query, retrieved_chunks, memory):
    """
    Build a prompt that includes both retrieved context AND memory.

    Structure:
    1. System instructions
    2. Conversation history (from memory)
    3. Retrieved document context
    4. Current question
    """
    # Get memory context
    memory_context = memory.get_context()

    # Build retrieved context
    context_parts = []
    for chunk in retrieved_chunks:
        chunk_text = (
            f"[Source: {chunk['source']} | "
            f"Score: {chunk['similarity_score']}]\n{chunk['text']}"
        )
        context_parts.append(chunk_text)
    retrieved_context = "\n\n---\n\n".join(context_parts)

    # Build full prompt
    prompt = f"""You are an AI assistant for Academic City University.
You help with questions about Ghana Election Results and Ghana's 2025 Budget.

{memory_context}

RETRIEVED DOCUMENTS:
{retrieved_context}

STRICT RULES:
1. Use ONLY facts from the retrieved documents above
2. You MAY reference previous conversation turns for context
3. If unsure, say: "The documents do not contain this information."
4. Always cite your source: [Election Data] or [Budget Document]
5. Keep answers under 200 words

CURRENT QUESTION: {query}

ANSWER:"""

    return prompt


# ─────────────────────────────────────────────
# TEST MEMORY SYSTEM
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Memory System...\n")

    memory = ConversationMemory(max_turns=5)

    # Simulate adding turns
    memory.add_turn(
        query="Who won the 2020 election in Ghana?",
        response="Nana Akufo-Addo of NPP won with 1,253,179 votes in Greater Accra.",
        sources=["Ghana_Election_Result.csv"]
    )

    memory.add_turn(
        query="What was Ghana's total revenue in 2025?",
        response="The 2025 Budget projects total revenue and grants at GHS 169.8 billion.",
        sources=["budget.pdf"]
    )

    # Show memory context
    print("Memory context that gets injected into prompts:")
    print("-" * 50)
    print(memory.get_context())
    print("-" * 50)

    memory.summarize()
    print("\nMemory system working!")