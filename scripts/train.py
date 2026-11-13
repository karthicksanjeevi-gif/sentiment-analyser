from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
ARTIFACT_DIR = ROOT / "artifacts"

# Keep negation words so sentiment context is preserved
CUSTOM_STOP_WORDS = [
    w for w in ENGLISH_STOP_WORDS
    if w not in {"not", "no", "never", "neither", "nor", "none", "nothing", "nowhere", "nobody", "without"}
]


def create_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words=CUSTOM_STOP_WORDS, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=5.0, max_iter=1000, random_state=42)),
    ])


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"])
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Training data must contain text and label columns.")
    if data["label"].nunique() < 2:
        raise ValueError("Training data must contain at least two labels.")

    # Validation split to measure hold-out metrics
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"].astype(str), data["label"].astype(str),
        test_size=0.15, random_state=42, stratify=data["label"].astype(str)
    )
    val_model = create_pipeline()
    val_model.fit(X_train, y_train)
    predictions = val_model.predict(X_test)
    accuracy = round(float(accuracy_score(y_test, predictions)), 4)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    # Train production model on full dataset for maximum vocabulary and accuracy
    prod_model = create_pipeline()
    prod_model.fit(data["text"].astype(str), data["label"].astype(str))

    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "model": prod_model,
        "model_version": "1.0.0",
        "trained_at": trained_at,
        "training_rows": int(len(data)),
    }
    joblib.dump(bundle, ARTIFACT_DIR / "model.joblib")

    metrics = {
        "model_version": "1.0.0",
        "trained_at": trained_at,
        "training_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": accuracy,
        "classification_report": report,
        "labels": sorted(data["label"].astype(str).unique().tolist()),
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model saved: {ARTIFACT_DIR / 'model.joblib'}")
    print(f"Metrics saved: {ARTIFACT_DIR / 'metrics.json'}")
    print(f"Hold-out accuracy: {accuracy}")


if __name__ == "__main__":
    main()
