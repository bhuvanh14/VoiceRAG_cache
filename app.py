"""
app.py — VoiceRAG-Cache Web Server
HOW TO RUN:  python app.py  →  open http://localhost:5000

First run: Wikipedia articles are fetched for each domain (~30-60s per domain).
Subsequent runs: ChromaDB cache is used instantly.
"""

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os, tempfile

load_dotenv()

from pipeline import VoiceRAGPipeline
from eval.evaluator import (
    generate_sessions, simulate_voicerag_cache,
    simulate_lru, simulate_lfu, simulate_no_cache,
    load_real_latencies,
    DEFAULT_L1_LATENCY, DEFAULT_L2_LATENCY, DEFAULT_RAG_LATENCY,
)

app       = Flask(__name__)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = VoiceRAGPipeline()
    return _pipeline


def session_stats(pl):
    session = pl.logger.current
    if not session:
        return {"turns":0,"hit_rate":0,"avg_latency":0,"l1_hits":0,"l2_hits":0,"misses":0}
    return {
        "turns":       len(session.turns),
        "hit_rate":    round(session.hit_rate() * 100, 1),
        "avg_latency": round(session.avg_latency()),
        "l1_hits":     sum(1 for t in session.turns if t.cache_result == "l1_hit"),
        "l2_hits":     sum(1 for t in session.turns if t.cache_result == "l2_hit"),
        "misses":      sum(1 for t in session.turns if t.cache_result == "miss"),
    }


def build_response(pl, result):
    pred = pl.last_predictor
    return {
        "query":        result["query"],
        "answer":       result["answer"],
        "sentiment":    result["sentiment"],
        "cache_result": result["cache_result"],
        "latency_ms":   round(result["latency_ms"]),
        "domain":       pl.current_domain,
        "predictor": {
            "evict":          pred.get("evict", []),
            "prefetch":       pred.get("prefetch", []),
            "sentiment":      pred.get("sentiment", ""),
            "prefetch_count": pred.get("prefetch_count", 0),
        },
        **session_stats(pl),
    }


@app.route("/")
def index():
    return render_template("index.html")


# ── Text query ────────────────────────────────────────────────────────────────
@app.route("/query", methods=["POST"])
def query():
    text = (request.get_json() or {}).get("query", "").strip()
    if not text:
        return jsonify({"error": "Empty query"}), 400
    try:
        pl     = get_pipeline()
        result = pl.process_query(text, run_predictor_sync=True)
        return jsonify(build_response(pl, result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Voice query ───────────────────────────────────────────────────────────────
@app.route("/voice_query", methods=["POST"])
def voice_query():
    try:
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"error": "No audio received"}), 400

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            audio_file.save(f.name)
            tmp = f.name

        pl  = get_pipeline()
        out = pl.asr.model.transcribe(tmp, fp16=False, language="en")
        os.unlink(tmp)
        query_text = out["text"].strip()

        if not query_text:
            return jsonify({"error": "Couldn't hear anything — please try again."}), 400

        result = pl.process_query(query_text, run_predictor_sync=True)
        return jsonify(build_response(pl, result))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Domain switch ─────────────────────────────────────────────────────────────
@app.route("/switch_domain", methods=["POST"])
def switch_domain():
    """
    Switch knowledge base domain.
    First switch to a new domain fetches Wikipedia (~30-60s).
    Subsequent switches are instant (ChromaDB cached).
    """
    domain = (request.get_json() or {}).get("domain", "").strip().lower()
    if domain not in ["healthcare", "hr", "legal"]:
        return jsonify({"error": "Invalid domain. Choose: healthcare, hr, legal"}), 400
    try:
        pl     = get_pipeline()
        result = pl.switch_domain(domain)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Evaluation (uses real measured latencies) ─────────────────────────────────
@app.route("/eval", methods=["POST"])
def run_eval():
    try:
        latencies = load_real_latencies()
        if not latencies:
            latencies = {
                "l1":  DEFAULT_L1_LATENCY,
                "l2":  DEFAULT_L2_LATENCY,
                "rag": DEFAULT_RAG_LATENCY,
            }

        sessions = generate_sessions(100, 10)
        results  = [
            simulate_voicerag_cache(sessions, latencies),
            simulate_lru(sessions, latencies),
            simulate_lfu(sessions, latencies),
            simulate_no_cache(sessions, latencies),
        ]
        base = results[-1].avg_latency
        return jsonify({
            "results": [{
                "name":     r.name,
                "hit_rate": round(r.hit_rate * 100, 1),
                "latency":  round(r.avg_latency),
                "speedup":  round(base / r.avg_latency, 1) if r.avg_latency else 0,
            } for r in results],
            "latencies_used": latencies,
            "source": "real session data" if load_real_latencies() else "defaults",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Reset ─────────────────────────────────────────────────────────────────────
@app.route("/reset", methods=["POST"])
def reset():
    try:
        pl = get_pipeline()
        pl.l1.flush()
        pl.l2.flush()
        pl.logger.new_session()
        pl.last_predictor = {
            "evict":[],"prefetch":[],"raw":"","sentiment":"","prefetch_count":0
        }
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀  VoiceRAG-Cache  →  http://localhost:5000")
    print("   First run: Wikipedia articles will be fetched (~30-60s)")
    print("   Subsequent runs: instant from ChromaDB cache\n")
    app.run(debug=False, port=5000)