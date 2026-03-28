"""
cache/l1_cache.py
STEP 3a — L1 Redis Exact-Match Cache

Caches query→answer pairs in Redis by hashed query string.
Normalized before hashing so "What is X?" and "what is x?" hit the same key.

PREREQ:  brew install redis && brew services start redis
HOW TO RUN:
    python -m cache.l1_cache
"""

import os
import json
import hashlib
import redis
from loguru import logger


class L1Cache:
    def __init__(self):
        self.client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
        )
        self.ttl = int(os.getenv("REDIS_TTL", 3600))
        self._ping()

    def _ping(self):
        try:
            self.client.ping()
            logger.info("L1 Redis connected ✅")
        except redis.ConnectionError:
            logger.error("Redis not running! Fix: brew services start redis")
            raise

    @staticmethod
    def _key(query: str) -> str:
        normalized = " ".join(query.strip().lower().split())
        return "l1:" + hashlib.sha256(normalized.encode()).hexdigest()[:20]

    def get(self, query: str) -> dict | None:
        raw = self.client.get(self._key(query))
        if raw:
            logger.debug(f"L1 HIT  | '{query[:50]}'")
            return json.loads(raw)
        logger.debug(f"L1 MISS | '{query[:50]}'")
        return None

    def set(self, query: str, answer: str, metadata: dict = None):
        value = json.dumps({
            "query":    query,
            "answer":   answer,
            "metadata": metadata or {},
        })
        self.client.setex(self._key(query), self.ttl, value)
        logger.debug(f"L1 SET  | '{query[:50]}' ttl={self.ttl}s")

    def delete(self, query: str):
        deleted = self.client.delete(self._key(query))
        if deleted:
            logger.info(f"L1 EVICT | '{query[:50]}'")

    def prefetch(self, query: str, answer: str, metadata: dict = None):
        """Store a proactively predicted answer before user asks."""
        logger.info(f"L1 PREFETCH | '{query[:60]}'")
        self.set(query, answer, metadata)

    def flush(self):
        """Clear all L1 cache entries (for testing)."""
        self.client.flushdb()
        logger.info("L1 cache flushed.")


# ── Run this file to test L1 cache ──────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 3a TEST — L1 Redis Cache")
    print("=" * 50)
    cache = L1Cache()
    cache.flush()

    cache.set("What is the capital of France?", "Paris is the capital of France.")

    # Test 1: exact hit
    r1 = cache.get("What is the capital of France?")
    print(f"\n✅ Exact hit:      {r1['answer']}" if r1 else "\n❌ Exact miss")

    # Test 2: normalized hit (different casing/spacing)
    r2 = cache.get("what is the capital of france?")
    print(f"✅ Normalized hit: {r2['answer']}" if r2 else "❌ Normalization failed")

    # Test 3: miss
    r3 = cache.get("What is the capital of Germany?")
    print(f"✅ Miss returned None: {r3 is None}" if r3 is None else "❌ Should have been None")

    print("\n✅ L1 Cache working!" if r1 and r2 and r3 is None else "\n❌ Issues found")