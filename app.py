"""
app.py — VoiceRAG-Cache Web Server
HOW TO RUN:  python app.py  →  open http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os, tempfile

load_dotenv()

from pipeline import VoiceRAGPipeline
from eval.evaluator import (
    generate_sessions, simulate_voicerag_cache,
    simulate_lru, simulate_lfu, simulate_no_cache,
)

app       = Flask(__name__)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = VoiceRAGPipeline()
        # Index default domain on startup
        _pipeline.kb.index_domain("healthcare")
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


# ── Text query ───────────────────────────────────────────────────────────────
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


# ── Domain switch (Option 3) ─────────────────────────────────────────────────
@app.route("/switch_domain", methods=["POST"])
def switch_domain():
    """
    Switch the active knowledge base domain.
    Body: { "domain": "healthcare" | "hr" | "legal" }
    Flushes caches and reindexes for the new domain.
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


# ── Evaluation ───────────────────────────────────────────────────────────────
@app.route("/eval", methods=["POST"])
def run_eval():
    try:
        sessions = generate_sessions(100, 10)
        results  = [
            simulate_voicerag_cache(sessions),
            simulate_lru(sessions),
            simulate_lfu(sessions),
            simulate_no_cache(sessions),
        ]
        base = results[-1].avg_latency
        return jsonify([{
            "name":     r.name,
            "hit_rate": round(r.hit_rate * 100, 1),
            "latency":  round(r.avg_latency),
            "speedup":  round(base / r.avg_latency, 1) if r.avg_latency else 0,
        } for r in results])
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
        pl.last_predictor = {"evict":[],"prefetch":[],"raw":"","sentiment":"","prefetch_count":0}
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🚀  VoiceRAG-Cache  →  http://localhost:5000\n")
    app.run(debug=False, port=5000)