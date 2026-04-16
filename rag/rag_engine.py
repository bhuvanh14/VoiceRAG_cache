"""
rag/rag_engine.py
------------------
RAG Engine — retrieves from ChromaDB (Wikipedia-powered) and
generates concise spoken answers via Groq (Llama 70B).

HOW TO RUN:
    python -m rag.rag_engine
"""

import os
from groq import Groq
from rag.knowledge_base import KnowledgeBase
from loguru import logger

# Better prompt — uses context generously, avoids refusing on partial matches
SYSTEM_PROMPT = (
    "You are a concise and helpful voice assistant. "
    "Answer the user's question using the provided context passages. "
    "Use whatever relevant information is available in the context, even if it only partially covers the question. "
    "Keep answers under 3 sentences — they will be spoken aloud via text-to-speech. "
    "Be direct and informative. "
    "Only say the context does not contain information if the passages are completely unrelated to the question. "
    "Never make up information that is not in the context."
)


class RAGEngine:
    def __init__(self, kb: KnowledgeBase = None):
        self.kb     = kb or KnowledgeBase()
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model  = "llama-3.3-70b-versatile"
        logger.info("RAG engine ready ✅")

    def answer(self, query: str, top_k: int = 8) -> dict:
        """
        Retrieve relevant passages and generate a spoken answer.
        top_k=8 gives broader context coverage from Wikipedia articles.

        Returns:
            { "query": str, "answer": str, "passages": list[dict] }
        """
        passages = self.kb.retrieve(query, top_k=top_k)

        context = "\n\n".join(
            f"[{i+1}] (Source: {p['source']})\n{p['text']}"
            for i, p in enumerate(passages)
        )

        prompt = (
            f"Context passages:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer (spoken, under 3 sentences):"
        )

        logger.info(f"RAG generating answer for: '{query[:60]}'")
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=250,
            temperature=0.2,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        logger.info(f"RAG answer: '{answer[:80]}'")

        return {"query": query, "answer": answer, "passages": passages}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 55)
    print("RAG Engine Test — Wikipedia-powered knowledge base")
    print("=" * 55)

    engine = RAGEngine()
    queries = [
        "What medications treat high blood pressure?",
        "How do I request time off from work?",
        "What is a non-disclosure agreement?",
        "How does metformin work for diabetes?",
        "What are the side effects of beta blockers?",
    ]

    for q in queries:
        result = engine.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"   Sources: {', '.join(set(p['source'] for p in result['passages'][:3]))}")
    print("\n✅ RAG engine working with Wikipedia!")