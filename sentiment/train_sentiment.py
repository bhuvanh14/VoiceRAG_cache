"""
sentiment/train_sentiment.py
STEP 2a — Fine-tune RoBERTa on GoEmotions

Maps 28 GoEmotions labels → 5 classes:
  0 = neutral     1 = confused     2 = urgent
  3 = satisfied   4 = escalating

On M2 Air: uses MPS backend, ~45 mins for 3 epochs.
Only needs to be run ONCE. Model is saved to sentiment/model/

HOW TO RUN:
    python -m sentiment.train_sentiment
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch
from sklearn.metrics import accuracy_score, f1_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

ID2LABEL = {
    0: "neutral",
    1: "confused",
    2: "urgent",
    3: "satisfied",
    4: "escalating",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# GoEmotions 28-label → 5-class mapping
GOEMOTION_MAP = {
    "neutral":        0,
    "confusion":      1, "surprise":    1, "realization": 1, "curiosity":     1,
    "fear":           2, "nervousness": 2, "anger":       2, "disgust":       2,
    "joy":            3, "love":        3, "gratitude":   3, "admiration":    3,
    "approval":       3, "caring":      3, "amusement":   3, "excitement":    3,
    "pride":          3, "relief":      3, "optimism":    3,
    "grief":          4, "sadness":     4, "disappointment": 4,
    "embarrassment":  4, "remorse":     4, "annoyance":   4, "disapproval":   4,
}
PRIORITY = ["urgent", "escalating", "confused", "satisfied", "neutral"]


def map_labels(example, label_names):
    active = [label_names[i] for i, v in enumerate(example["labels"]) if v == 1]
    if not active:
        return {"label": 0}
    for cls in PRIORITY:
        mapped = [GOEMOTION_MAP.get(l, 0) for l in active]
        if LABEL2ID[cls] in mapped:
            return {"label": LABEL2ID[cls]}
    return {"label": 0}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1_macro": round(f1_score(labels, preds, average="macro"), 4),
    }


def main():
    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    )
    print(f"Training on: {device}")

    print("Downloading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified")
    label_names = dataset["train"].features["labels"].feature.names

    print("Mapping to 5 classes...")
    dataset = dataset.map(lambda ex: map_labels(ex, label_names))
    dataset = dataset.remove_columns(["labels"])

    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=5,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=200,
        fp16=False,                            # MPS does not support fp16
        use_mps_device=(device == "mps"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting training — this takes ~45 mins on M2 Air...")
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\n✅ Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()