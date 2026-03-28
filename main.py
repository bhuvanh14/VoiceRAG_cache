"""
main.py
STEP 10 — Main Entry Point

Runs VoiceRAG-Cache as an interactive voice agent.
Supports two modes:
  --voice   : full voice in / voice out (default for demo)
  --text    : type queries in terminal (easier for testing)

HOW TO RUN:
    # Text mode (easiest to start with)
    python main.py --text

    # Full voice mode
    python main.py --voice

    # Run evaluation benchmarks
    python main.py --eval
"""

import argparse
import sys
from dotenv import load_dotenv
load_dotenv()

from pipeline import VoiceRAGPipeline
from eval.evaluator import run_evaluation
from loguru import logger

BANNER = """
╔══════════════════════════════════════════════════════╗
║           VoiceRAG-Cache  v1.0                       ║
║   Sentiment-Aware GenAI Two-Tier Caching Agent       ║
║   PES University · 2024-25                           ║
╚══════════════════════════════════════════════════════╝
"""

CACHE_LABELS = {
    "l1_hit": "⚡ L1 hit",
    "l2_hit": "🔍 L2 hit",
    "miss":   "🌐 RAG   ",
}


def run_text_mode(pipeline: VoiceRAGPipeline):
    """Interactive text mode — type queries, get answers printed."""
    print("\nText mode — type your query and press Enter.")
    print("Commands:  'quit' to exit | 'stats' for session stats | 'flush' to clear cache\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "stats":
            pipeline.print_stats()
            continue
        if query.lower() == "flush":
            pipeline.l1.flush()
            pipeline.l2.flush()
            print("  Caches cleared.\n")
            continue

        result = pipeline.process_query(query)

        label = CACHE_LABELS.get(result["cache_result"], result["cache_result"])
        print(f"\n  {label} | {result['sentiment']:12s} | {result['latency_ms']:.0f}ms")
        print(f"  Agent: {result['answer']}\n")


def run_voice_mode(pipeline: VoiceRAGPipeline):
    """Full voice mode — speak, hear the answer, repeat."""
    print("\nVoice mode — speak your query. Say 'stop' or 'quit' to exit.\n")

    while True:
        try:
            print("Listening...")
            result = pipeline.process_voice()

            query = result.get("query", "")
            if not query:
                continue

            if query.lower().strip() in ("stop", "quit", "exit"):
                pipeline.tts.speak("Goodbye!")
                break

            label = CACHE_LABELS.get(result["cache_result"], result["cache_result"])
            print(f"  {label} | {result['sentiment']:12s} | {result['latency_ms']:.0f}ms")
            print(f"  Q: {query}")
            print(f"  A: {result['answer']}\n")

        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser(description="VoiceRAG-Cache")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--text",  action="store_true", help="Text input mode (default)")
    group.add_argument("--voice", action="store_true", help="Full voice in/out mode")
    group.add_argument("--eval",  action="store_true", help="Run evaluation benchmarks")
    args = parser.parse_args()

    print(BANNER)

    if args.eval:
        run_evaluation(num_sessions=200, turns=10)
        return

    print("Initializing pipeline (this takes ~10s on first run)...\n")
    pipeline = VoiceRAGPipeline()
    print("\nPipeline ready.\n")

    try:
        if args.voice:
            run_voice_mode(pipeline)
        else:
            # Default to text mode
            run_text_mode(pipeline)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        pipeline.print_stats()
        pipeline.save()
        print("\nSession saved. Goodbye!\n")


if __name__ == "__main__":
    main()