"""
predictor/session_logger.py
STEP 6 — Session Logger

Tracks every turn of a conversation: query, sentiment, cache result, latency.
This log is what gets fed to the GenAI cache predictor to make decisions.

HOW TO RUN:
    python -m predictor.session_logger
"""

import json
import time
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from loguru import logger

LOG_DIR = "./data/sessions"


@dataclass
class Turn:
    turn_id:      int
    timestamp:    str
    query:        str
    sentiment:    str
    confidence:   float
    cache_result: str       # "l1_hit" | "l2_hit" | "miss"
    latency_ms:   float
    answer:       str


@dataclass
class Session:
    session_id: str
    started_at: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(
        self,
        query:        str,
        sentiment:    str,
        confidence:   float,
        cache_result: str,
        latency_ms:   float,
        answer:       str,
    ) -> Turn:
        turn = Turn(
            turn_id      = len(self.turns) + 1,
            timestamp    = datetime.now().isoformat(),
            query        = query,
            sentiment    = sentiment,
            confidence   = confidence,
            cache_result = cache_result,
            latency_ms   = latency_ms,
            answer       = answer,
        )
        self.turns.append(turn)
        logger.debug(
            f"Turn {turn.turn_id} | {cache_result:8s} | "
            f"{sentiment:12s} | {latency_ms:.0f}ms | '{query[:40]}'"
        )
        return turn

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "turns":      [asdict(t) for t in self.turns],
        }

    def recent_turns(self, n: int = 5) -> list[Turn]:
        return self.turns[-n:]

    def hit_rate(self) -> float:
        if not self.turns:
            return 0.0
        hits = sum(1 for t in self.turns if t.cache_result != "miss")
        return hits / len(self.turns)

    def avg_latency(self) -> float:
        if not self.turns:
            return 0.0
        return sum(t.latency_ms for t in self.turns) / len(self.turns)


class SessionLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.current: Session | None = None

    def new_session(self) -> Session:
        sid = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.current = Session(session_id=sid, started_at=datetime.now().isoformat())
        logger.info(f"New session: {sid}")
        return self.current

    def log_turn(
        self,
        query:        str,
        sentiment:    str,
        confidence:   float,
        cache_result: str,
        latency_ms:   float,
        answer:       str,
    ) -> Turn:
        if self.current is None:
            self.new_session()
        return self.current.add_turn(
            query, sentiment, confidence, cache_result, latency_ms, answer
        )

    def save_session(self):
        if not self.current:
            return
        path = os.path.join(LOG_DIR, f"{self.current.session_id}.json")
        with open(path, "w") as f:
            json.dump(self.current.to_dict(), f, indent=2)
        logger.info(f"Session saved → {path}")

    def context_for_predictor(self, n: int = 5) -> str:
        """
        Formats the last N turns into a compact string for the cache predictor prompt.
        """
        if not self.current or not self.current.turns:
            return "No session history yet."

        lines = []
        for t in self.current.recent_turns(n):
            lines.append(
                f"Turn {t.turn_id}: [{t.sentiment}] '{t.query}' → {t.cache_result} ({t.latency_ms:.0f}ms)"
            )
        return "\n".join(lines)


# ── Run this file to test session logging ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 6 TEST — Session Logger")
    print("=" * 50)

    sl = SessionLogger()
    sl.new_session()

    # Simulate some turns
    sl.log_turn("What medications treat hypertension?", "neutral",    0.82, "miss",   2400.0, "ACE inhibitors...")
    sl.log_turn("Which drugs lower blood pressure?",   "confused",   0.75, "l2_hit", 180.0,  "ACE inhibitors...")
    sl.log_turn("How do I refill my prescription?",    "urgent",     0.91, "miss",   2200.0, "Contact pharmacy...")
    sl.log_turn("Can I get an emergency refill?",      "urgent",     0.88, "l1_hit", 160.0,  "Contact pharmacy...")
    sl.log_turn("What are the side effects?",          "confused",   0.70, "miss",   2350.0, "Common side effects...")

    print(f"\nHit rate:    {sl.current.hit_rate():.0%}")
    print(f"Avg latency: {sl.current.avg_latency():.0f}ms")
    print("\nContext for predictor:")
    print(sl.context_for_predictor())

    sl.save_session()
    print("\n✅ Session logger working!")