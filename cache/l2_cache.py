"""
cache/l2_cache.py
STEP 3b — L2 FAISS Semantic Similarity Cache

Stores query embeddings + answers in a FAISS index.
On lookup: embeds the query → cosine search → returns cached answer
if best match similarity >= threshold.

This catches paraphrases that L1 misses:
  "What drugs treat hypertension?" == "Which medications are used for high blood pressure?"

HOW TO RUN:
    python -m cache.l2_cache
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

THRESHOLD  = float(os.getenv("L2_SIMILARITY_THRESHOLD", 0.85))
INDEX_FILE = "./data/l2_faiss.index"
META_FILE  = "./data/l2_meta.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
DIM        = 384


class L2Cache:
    def __init__(self, threshold: float = None):
        self.threshold = threshold or THRESHOLD
        self.embedder  = SentenceTransformer(MODEL_NAME)
        self.index     = faiss.IndexFlatIP(DIM)   # inner product = cosine on normalized vecs
        self.metadata: list[dict] = []
        self._load()
        logger.info(f"L2 FAISS ready ✅ | {self.index.ntotal} entries | threshold={self.threshold}")

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(
            [text], normalize_embeddings=True
        ).astype(np.float32)

    def get(self, query: str) -> dict | None:
        if self.index.ntotal == 0:
            return None
        q_vec           = self._embed(query)
        scores, indices = self.index.search(q_vec, k=1)
        score           = float(scores[0][0])
        idx             = int(indices[0][0])

        if score >= self.threshold:
            entry = self.metadata[idx]
            logger.debug(f"L2 HIT  sim={score:.3f} | matched='{entry['query'][:50]}'")
            return entry
        logger.debug(f"L2 MISS sim={score:.3f} | query='{query[:50]}'")
        return None

    def set(self, query: str, answer: str, metadata: dict = None):
        self.index.add(self._embed(query))
        self.metadata.append({"query": query, "answer": answer, **(metadata or {})})
        self._save()
        logger.debug(f"L2 SET  total={self.index.ntotal} | '{query[:50]}'")

    def prefetch(self, query: str, answer: str, metadata: dict = None):
        logger.info(f"L2 PREFETCH | '{query[:60]}'")
        self.set(query, answer, metadata)

    def delete_by_query(self, query: str):
        """FAISS has no native delete — rebuild without the matching entry."""
        if self.index.ntotal == 0:
            return
        q_vec           = self._embed(query)
        scores, indices = self.index.search(q_vec, k=1)
        if float(scores[0][0]) < self.threshold:
            logger.warning(f"L2 EVICT: no match for '{query[:50]}'")
            return
        target = int(indices[0][0])
        self.metadata.pop(target)
        self.index = faiss.IndexFlatIP(DIM)
        if self.metadata:
            vecs = np.vstack([self._embed(m["query"]) for m in self.metadata])
            self.index.add(vecs)
        self._save()
        logger.info(f"L2 EVICT idx={target} | '{query[:50]}'")

    def flush(self):
        self.index    = faiss.IndexFlatIP(DIM)
        self.metadata = []
        self._save()
        logger.info("L2 cache flushed.")

    def _save(self):
        os.makedirs("./data", exist_ok=True)
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self):
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"L2 loaded {self.index.ntotal} entries from disk.")


# ── Run this file to test L2 cache ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 3b TEST — L2 FAISS Semantic Cache")
    print("=" * 50)
    cache = L2Cache(threshold=0.80)
    cache.flush()

    cache.set(
        "What medications treat high blood pressure?",
        "ACE inhibitors, beta-blockers, and diuretics are commonly prescribed.",
    )
    cache.set(
        "How do I refill my prescription?",
        "Contact your pharmacy with your prescription number to arrange a refill.",
    )

    tests = [
        ("Which drugs are used for hypertension?",     True,  "Semantic paraphrase should hit"),
        ("How can I get more of my medication?",       True,  "Paraphrase of refill query"),
        ("What is the weather like today?",            False, "Completely unrelated — should miss"),
    ]
    print()
    for query, expect_hit, desc in tests:
        result = cache.get(query)
        hit    = result is not None
        ok     = "✅" if hit == expect_hit else "❌"
        print(f"{ok} {desc}")
        if hit:
            print(f"   → {result['answer'][:80]}")
    print()