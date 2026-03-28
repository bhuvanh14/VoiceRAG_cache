"""
rag/rag_engine.py
STEP 4b — Retrieval-Augmented Generation Engine

Retrieves relevant passages from ChromaDB then calls Groq (free)
to generate a concise spoken answer.

HOW TO RUN:
    python -m rag.rag_engine
"""

import os
from groq import Groq
from rag.knowledge_base import KnowledgeBase
from loguru import logger

SYSTEM_PROMPT = (
    "You are a concise voice assistant. "
    "Answer the user's question using ONLY the provided context passages. "
    "Keep answers under 3 sentences — they will be spoken aloud. "
    "If the context does not contain the answer, say so clearly. "
    "Never make up information."
)


class RAGEngine:
    def __init__(self, kb: KnowledgeBase = None):
        self.kb     = kb or KnowledgeBase()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"   # free on Groq
        logger.info("RAG engine ready ✅")

    def answer(self, query: str, top_k: int = 5) -> dict:
        """
        Retrieve + generate.
        Returns:
            { "query": str, "answer": str, "passages": list[dict] }
        """
        passages = self.kb.retrieve(query, top_k=top_k)

        context = "\n\n".join(
            f"[{i+1}] {p['text']}" for i, p in enumerate(passages)
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        logger.info(f"Generating answer for: '{query[:60]}'")
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        logger.info(f"Answer: '{answer[:80]}'")

        return {"query": query, "answer": answer, "passages": passages}


# ── Run this file to test RAG ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 50)
    print("STEP 4b TEST — RAG Engine")
    print("=" * 50)

    engine = RAGEngine()
    queries = [
        "What medications treat high blood pressure?",
        "How do I request time off from work?",
        "What is a non-disclosure agreement?",
    ]
    for q in queries:
        result = engine.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
    print("\n✅ RAG engine working!")