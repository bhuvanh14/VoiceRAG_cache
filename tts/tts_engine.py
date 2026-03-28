"""
tts/tts_engine.py
STEP 5 — Text to Speech

Converts text answers to spoken audio using pyttsx3 (offline, no API needed).
Works on M2 Air out of the box using macOS native voices.

HOW TO RUN:
    python -m tts.tts_engine
"""

import pyttsx3
from loguru import logger


class TTSEngine:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate",   rate)
        self.engine.setProperty("volume", volume)
        self._set_best_voice()
        logger.info("TTS engine ready ✅")

    def _set_best_voice(self):
        """Pick a natural-sounding voice. On macOS, prefers Samantha or Alex."""
        voices   = self.engine.getProperty("voices")
        preferred = ["samantha", "alex", "karen", "daniel"]
        for name in preferred:
            for voice in voices:
                if name in voice.name.lower():
                    self.engine.setProperty("voice", voice.id)
                    logger.info(f"TTS voice: {voice.name}")
                    return
        if voices:
            self.engine.setProperty("voice", voices[0].id)
            logger.info(f"TTS voice: {voices[0].name}")

    def speak(self, text: str):
        """Speak text aloud — blocks until speech is complete."""
        logger.info(f"Speaking: '{text[:60]}'")
        self.engine.say(text)
        self.engine.runAndWait()

    def list_voices(self):
        """Print all available voices (useful for debugging)."""
        for v in self.engine.getProperty("voices"):
            print(f"  id={v.id}  name={v.name}")


# ── Run this file to test TTS ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 5 TEST — Text to Speech")
    print("=" * 50)
    tts = TTSEngine()
    print("\nAvailable voices:")
    tts.list_voices()
    print("\nSpeaking test sentence...")
    tts.speak("Voice RAG Cache initialized. I am ready to answer your questions.")
    print("✅ TTS working!")