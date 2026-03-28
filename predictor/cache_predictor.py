"""
predictor/cache_predictor.py
------------------------------
GenAI Cache Predictor — CORE INNOVATION

Sentiment-aware prefetch scaling:
  neutral / satisfied  → 2 prefetch queries
  confused             → 3 prefetch queries (needs clarifications)
  urgent / escalating  → 5 prefetch queries (needs fast answers NOW)

This directly proves the sentiment-aware claim — you can see it in the UI.
"""

import os
import json
import re
import threading
from dataclasses import dataclass
from groq import Groq
from loguru import logger

# How many prefetch queries to request per sentiment class
PREFETCH_COUNT = {
    "neutral":    2,
    "satisfied":  2,
    "confused":   3,
    "urgent":     5,
    "escalating": 5,
}

PREDICTOR_SYSTEM = """You are a cache policy engine for a voice RAG system.
You receive the recent conversation history with sentiment labels and cache hit/miss results.
Your job is to output a JSON object with exactly two fields:
  "evict":    list of exact query strings to remove from cache (stale or unlikely to repeat)
  "prefetch": list of NEW query strings the user is LIKELY to ask next

Rules:
- Output ONLY valid JSON. No explanation, no markdown, no preamble.
- "evict" should contain at most 2 queries from the history that are unlikely to be asked again.
- The number of "prefetch" queries is given to you as N — generate exactly N prefetch queries.
- If the user is "urgent" or "escalating", prefetch follow-up queries on the SAME urgent topic fast.
- If the user is "confused", prefetch clarifying questions on the same topic.
- If the user is "satisfied", prefetch logical next-step queries.
- Keep prefetch queries concise — under 15 words each.
- Prefetch queries must be different from anything already in the history.

Example output:
{
  "evict": ["What is the weather today?"],
  "prefetch": [
    "What are the side effects of lisinopril?",
    "Can I take lisinopril with food?"
  ]
}"""


@dataclass
class PredictorDecision:
    evict:    list[str]
    prefetch: list[str]
    raw:      str
    sentiment: str
    prefetch_count: int


class CachePredictor:
    def __init__(self):
        self.client  = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model   = "llama-3.3-70b-versatile"
        self.context_window = int(os.getenv("PREDICTOR_CONTEXT_WINDOW", 5))
        logger.info("Cache predictor ready ✅")

    def predict(self, session_context: str, current_sentiment: str) -> PredictorDecision:
        """
        Call the LLM to get evict/prefetch decisions.
        Prefetch count scales with sentiment urgency.
        """
        n = PREFETCH_COUNT.get(current_sentiment, 2)

        user_message = (
            f"Session history (most recent last):\n"
            f"{session_context}\n\n"
            f"Current user sentiment: {current_sentiment}\n"
            f"Generate exactly {n} prefetch queries (more because sentiment is {current_sentiment}).\n\n"
            f"Output your cache policy decision as JSON:"
        )

        logger.info(f"Predictor: sentiment={current_sentiment} → requesting {n} prefetches")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": PREDICTOR_SYSTEM},
                    {"role": "user",   "content": user_message},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            logger.debug(f"Predictor raw output: {raw}")
            return self._parse(raw, current_sentiment, n)

        except Exception as e:
            logger.error(f"Predictor LLM call failed: {e}")
            return PredictorDecision(evict=[], prefetch=[], raw=str(e),
                                     sentiment=current_sentiment, prefetch_count=n)

    def _parse(self, raw: str, sentiment: str, n: int) -> PredictorDecision:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            logger.warning("Predictor: no JSON found in output")
            return PredictorDecision(evict=[], prefetch=[], raw=raw,
                                     sentiment=sentiment, prefetch_count=n)
        try:
            data     = json.loads(match.group())
            evict    = [str(q) for q in data.get("evict",    []) if q]
            prefetch = [str(q) for q in data.get("prefetch", []) if q]
            logger.info(f"Predictor: evict={len(evict)} prefetch={len(prefetch)} (requested {n})")
            for q in evict:    logger.info(f"  EVICT    → '{q}'")
            for q in prefetch: logger.info(f"  PREFETCH → '{q}'")
            return PredictorDecision(evict=evict, prefetch=prefetch, raw=raw,
                                     sentiment=sentiment, prefetch_count=n)
        except json.JSONDecodeError as e:
            logger.error(f"Predictor JSON parse error: {e}")
            return PredictorDecision(evict=[], prefetch=[], raw=raw,
                                     sentiment=sentiment, prefetch_count=n)

    def predict_async(self, session_context: str, current_sentiment: str, callback):
        def _run():
            decision = self.predict(session_context, current_sentiment)
            callback(decision)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    predictor = CachePredictor()

    print("\n=== TEST 1: neutral (expect 2 prefetches) ===")
    ctx = "Turn 1: [neutral] 'What medications treat blood pressure?' → miss"
    d   = predictor.predict(ctx, "neutral")
    print(f"Prefetch count: {len(d.prefetch)} (expected 2)")
    for q in d.prefetch: print(f"  + {q}")

    print("\n=== TEST 2: urgent (expect 5 prefetches) ===")
    ctx = "Turn 1: [urgent] 'I need my prescription refilled urgently!' → miss"
    d   = predictor.predict(ctx, "urgent")
    print(f"Prefetch count: {len(d.prefetch)} (expected 5)")
    for q in d.prefetch: print(f"  + {q}")

    print("\n=== TEST 3: confused (expect 3 prefetches) ===")
    ctx = "Turn 1: [confused] 'I dont understand how leave works' → miss"
    d   = predictor.predict(ctx, "confused")
    print(f"Prefetch count: {len(d.prefetch)} (expected 3)")
    for q in d.prefetch: print(f"  + {q}")