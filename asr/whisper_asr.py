"""
asr/whisper_asr.py
STEP 1 — Speech to Text

Records audio from mic and transcribes using OpenAI Whisper.
On M2 Air use model='base' for best speed/accuracy tradeoff (~300ms latency).

HOW TO RUN:
    python -m asr.whisper_asr
"""

import os
import tempfile
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
from loguru import logger


class WhisperASR:
    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv("WHISPER_MODEL", "base")
        logger.info(f"Loading Whisper [{model_name}] — first run downloads the model (~150MB for base)")
        self.model = whisper.load_model(model_name)
        self.sample_rate = 16000

    def record(self, duration_seconds: float = 5.0) -> np.ndarray:
        """Fixed duration recording."""
        logger.info(f"Recording {duration_seconds}s — speak now...")
        audio = sd.rec(
            int(duration_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        logger.info("Recording done.")
        return audio.flatten()

    def record_until_silence(
        self,
        silence_threshold: float = 0.01,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        chunk_duration: float = 0.5,
    ) -> np.ndarray:
        """
        Records until silence is detected.
        Better for conversational use — stops automatically when you stop talking.
        """
        chunks = []
        elapsed = 0.0
        silent_streak = 0
        needed_silent_chunks = 2  # 2 × 0.5s = 1s of silence to stop

        logger.info("Listening... speak now. Will stop after 1s of silence.")
        while elapsed < max_duration:
            chunk = sd.rec(
                int(chunk_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            chunk = chunk.flatten()
            chunks.append(chunk)
            elapsed += chunk_duration

            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms < silence_threshold and elapsed >= min_duration:
                silent_streak += 1
                if silent_streak >= needed_silent_chunks:
                    logger.info("Silence detected — stopping.")
                    break
            else:
                silent_streak = 0

        return np.concatenate(chunks)

    def transcribe(self, audio: np.ndarray) -> str:
        """Convert numpy audio array to text string."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, self.sample_rate, (audio * 32767).astype(np.int16))
            tmp_path = f.name
        try:
            result = self.model.transcribe(tmp_path, fp16=False, language="en")
            text = result["text"].strip()
            logger.info(f"Transcribed: '{text}'")
            return text
        finally:
            os.unlink(tmp_path)

    def listen_and_transcribe(self, duration: float = None) -> str:
        """
        One-call interface.
        duration=None uses silence detection (recommended for demos).
        duration=5.0 records for exactly 5 seconds.
        """
        audio = self.record(duration) if duration else self.record_until_silence()
        return self.transcribe(audio)


# ── Run this file to test ASR ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1 TEST — Whisper ASR")
    print("=" * 50)
    asr = WhisperASR()
    print("\nSay something into your mic...")
    text = asr.listen_and_transcribe()
    print(f"\nResult: '{text}'")
    print("\n✅ ASR working!" if text else "\n❌ No transcription — check your mic settings.")