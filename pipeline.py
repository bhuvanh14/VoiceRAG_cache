"""
pipeline.py — VoiceRAG-Cache Pipeline Orchestrator

Now supports domain switching:
  pipeline.switch_domain("healthcare" | "hr" | "legal")
  This rebuilds the knowledge base and RAG engine for the selected domain.
"""

import time
import os
from dotenv import load_dotenv
load_dotenv()

from asr.whisper_asr                import WhisperASR
from sentiment.sentiment_classifier import SentimentClassifier
from cache.l1_cache                 import L1Cache
from cache.l2_cache                 import L2Cache
from rag.knowledge_base             import KnowledgeBase
from rag.rag_engine                 import RAGEngine
from tts.tts_engine                 import TTSEngine
from predictor.session_logger       import SessionLogger
from predictor.cache_predictor      import CachePredictor, PredictorDecision
from loguru import logger


class VoiceRAGPipeline:
    def __init__(self):
        logger.info("Initializing VoiceRAG-Cache pipeline...")
        self.asr        = WhisperASR()
        self.sentiment  = SentimentClassifier()
        self.l1         = L1Cache()
        self.l2         = L2Cache()
        self.kb         = KnowledgeBase()
        self.rag        = RAGEngine(kb=self.kb)
        self.tts        = TTSEngine()
        self.logger     = SessionLogger()
        self.predictor  = CachePredictor()
        self.logger.new_session()

        self.current_domain   = "healthcare"
        self.last_predictor   = {"evict": [], "prefetch": [], "raw": "",
                                  "sentiment": "", "prefetch_count": 0}
        logger.info("Pipeline ready ✅")

    # ── Domain switching (Option 3) ──────────────────────────────────────────
    def switch_domain(self, domain: str) -> dict:
        """
        Switch the knowledge base to a different domain.
        Flushes both caches and rebuilds KB + RAG for the new domain.
        Returns {"status": "ok", "domain": domain, "chunks": N}
        """
        domain = domain.lower().strip()
        valid  = ["healthcare", "hr", "legal"]
        if domain not in valid:
            return {"status": "error", "message": f"Unknown domain. Choose from: {valid}"}

        logger.info(f"Switching domain to: {domain}")
        self.current_domain = domain

        # Flush caches — new domain means old answers are irrelevant
        self.l1.flush()
        self.l2.flush()
        self.logger.new_session()
        self.last_predictor = {"evict": [], "prefetch": [], "raw": "",
                                "sentiment": "", "prefetch_count": 0}

        # Rebuild knowledge base for new domain
        self.kb = KnowledgeBase(collection_name=f"kb_{domain}")
        self.kb.index_domain(domain)
        self.rag = RAGEngine(kb=self.kb)

        chunks = self.kb.count()
        logger.info(f"Domain switched to '{domain}' — {chunks} chunks indexed.")
        return {"status": "ok", "domain": domain, "chunks": chunks}

    # ── Apply predictor decisions ────────────────────────────────────────────
    def _apply_predictor_decision(self, decision: PredictorDecision):
        self.last_predictor = {
            "evict":          decision.evict,
            "prefetch":       decision.prefetch,
            "raw":            decision.raw,
            "sentiment":      decision.sentiment,
            "prefetch_count": decision.prefetch_count,
        }
        for query in decision.evict:
            self.l1.delete(query)
            self.l2.delete_by_query(query)

        for query in decision.prefetch:
            if self.l1.get(query) is None:
                try:
                    result = self.rag.answer(query)
                    self.l1.prefetch(query, result["answer"])
                    self.l2.prefetch(query, result["answer"])
                except Exception as e:
                    logger.warning(f"Prefetch failed for '{query}': {e}")

    # ── Core query processor ─────────────────────────────────────────────────
    def process_query(self, query: str, run_predictor_sync: bool = False) -> dict:
        start = time.time()

        sent       = self.sentiment.classify(query)
        sentiment  = sent["label"]
        confidence = sent["confidence"]

        cache_result = "miss"
        answer       = None

        l1_hit = self.l1.get(query)
        if l1_hit:
            answer       = l1_hit["answer"]
            cache_result = "l1_hit"
        else:
            l2_hit = self.l2.get(query)
            if l2_hit:
                answer       = l2_hit["answer"]
                cache_result = "l2_hit"
                self.l1.set(query, answer)
            else:
                rag_result   = self.rag.answer(query)
                answer       = rag_result["answer"]
                cache_result = "miss"
                self.l1.set(query, answer)
                self.l2.set(query, answer)

        latency_ms = (time.time() - start) * 1000

        self.logger.log_turn(
            query        = query,
            sentiment    = sentiment,
            confidence   = confidence,
            cache_result = cache_result,
            latency_ms   = latency_ms,
            answer       = answer,
        )

        context = self.logger.context_for_predictor()

        if run_predictor_sync:
            decision = self.predictor.predict(context, sentiment)
            self._apply_predictor_decision(decision)
        else:
            self.predictor.predict_async(
                session_context   = context,
                current_sentiment = sentiment,
                callback          = self._apply_predictor_decision,
            )

        return {
            "query":        query,
            "answer":       answer,
            "sentiment":    sentiment,
            "cache_result": cache_result,
            "latency_ms":   latency_ms,
        }

    # ── Voice turn ───────────────────────────────────────────────────────────
    def process_voice(self) -> dict:
        query = self.asr.listen_and_transcribe()
        if not query or not query.strip():
            self.tts.speak("Sorry, I didn't catch that. Please try again.")
            return {"query": "", "answer": "", "cache_result": "miss", "latency_ms": 0}
        result = self.process_query(query)
        self.tts.speak(result["answer"])
        return result

    # ── Stats ────────────────────────────────────────────────────────────────
    def print_stats(self):
        session = self.logger.current
        if not session:
            return
        print("\n── Session Stats ────────────────────────")
        print(f"  Turns:       {len(session.turns)}")
        print(f"  Hit rate:    {session.hit_rate():.0%}")
        print(f"  Avg latency: {session.avg_latency():.0f}ms")
        hits = {"l1_hit": 0, "l2_hit": 0, "miss": 0}
        for t in session.turns:
            hits[t.cache_result] = hits.get(t.cache_result, 0) + 1
        print(f"  L1 hits: {hits['l1_hit']}  L2 hits: {hits['l2_hit']}  Misses: {hits['miss']}")
        print("─────────────────────────────────────────")

    def save(self):
        self.logger.save_session()