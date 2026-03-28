"""
sentiment/sentiment_classifier.py
STEP 2b — Sentiment inference

Loads the fine-tuned RoBERTa model and classifies text into 5 classes.
Falls back to keyword rules if model hasn't been trained yet.

HOW TO RUN:
    python -m sentiment.sentiment_classifier
"""

import os
import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from loguru import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

ID2LABEL = {
    0: "neutral",
    1: "confused",
    2: "urgent",
    3: "satisfied",
    4: "escalating",
}

# Fallback keyword rules (used before model is trained)
KEYWORD_RULES = {
    "urgent":     ["urgent", "emergency", "asap", "immediately", "critical", "help me"],
    "confused":   ["don't understand", "confused", "unclear", "what does", "how does", "why"],
    "satisfied":  ["thank", "great", "perfect", "got it", "understood", "awesome", "helpful"],
    "escalating": ["frustrated", "angry", "ridiculous", "useless", "terrible", "again", "still not"],
}


class SentimentClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = (
            "mps"  if torch.backends.mps.is_available()  else
            "cuda" if torch.cuda.is_available()           else
            "cpu"
        )
        self._load()

    def _load(self):
        if os.path.isdir(MODEL_DIR):
            logger.info(f"Loading sentiment model from {MODEL_DIR}")
            self.tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
            self.model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Sentiment model ready.")
        else:
            logger.warning(
                "Sentiment model not found — using keyword fallback. "
                "Run: python -m sentiment.train_sentiment"
            )

    def classify(self, text: str) -> dict:
        """
        Returns:
            {
                "label": "urgent",        # one of 5 classes
                "confidence": 0.92,
                "scores": {"neutral": 0.03, "confused": 0.02, ...}
            }
        """
        if self.model:
            return self._model_classify(text)
        return self._keyword_classify(text)

    def _model_classify(self, text: str) -> dict:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        pred_id = int(torch.argmax(logits).item())
        return {
            "label":      ID2LABEL[pred_id],
            "confidence": round(probs[pred_id], 4),
            "scores":     {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
        }

    def _keyword_classify(self, text: str) -> dict:
        t = text.lower()
        for label, keywords in KEYWORD_RULES.items():
            if any(kw in t for kw in keywords):
                return {"label": label, "confidence": 0.65, "scores": {}}
        return {"label": "neutral", "confidence": 0.80, "scores": {}}


# ── Run this file to test sentiment ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 2 TEST — Sentiment Classifier")
    print("=" * 50)
    clf = SentimentClassifier()
    tests = [
        ("I need help urgently, this is an emergency!", "urgent"),
        ("I don't understand how this caching works", "confused"),
        ("Thanks, that explanation was perfect!", "satisfied"),
        ("This is absolutely ridiculous, I asked three times", "escalating"),
        ("What time does the clinic open?", "neutral"),
    ]
    print()
    all_pass = True
    for text, expected in tests:
        result = clf.classify(text)
        ok = "✅" if result["label"] == expected else "⚠️ "
        if result["label"] != expected:
            all_pass = False
        print(f"{ok} [{result['label']:12s} {result['confidence']:.0%}]  {text}")
    print(f"\n{'✅ All passed!' if all_pass else '⚠️  Some mismatches (normal with keyword fallback)'}")