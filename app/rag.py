from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge"


def split_chunks(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
        else:
            current = ""
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                if current and len(current) + len(sentence) + 1 > max_chars:
                    chunks.append(current)
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                chunks.append(current)
    return chunks


class LocalRAG:
    """Local retrieval augmented answering using TF-IDF and cosine similarity."""

    def __init__(self, knowledge_dir: Path = DEFAULT_KNOWLEDGE_DIR):
        self.documents: list[dict[str, Any]] = []
        for path in sorted(Path(knowledge_dir).glob("*.md")):
            for i, chunk in enumerate(split_chunks(path.read_text(encoding="utf-8"))):
                self.documents.append({"source": path.name, "chunk": i, "text": chunk})
        if not self.documents:
            self.documents = [{
                "source": "built-in-fallback",
                "chunk": 0,
                "text": "This project has a FastAPI prediction endpoint, TF-IDF RAG retrieval, automated tests, Docker, and GitHub Actions CI/CD."
            }]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in self.documents])

    def query(self, question: str, top_k: int = 3) -> dict[str, Any]:
        vector = self.vectorizer.transform([question])
        scores = cosine_similarity(vector, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        results = []
        for i in indices:
            score = float(scores[int(i)])
            if score <= 0:
                continue
            doc = self.documents[int(i)]
            results.append({
                "source": doc["source"],
                "score": round(score, 4),
                "excerpt": doc["text"][:500],
                "text": doc["text"],
            })
        if not results:
            return {
                "answer": "I couldn't find a relevant passage in the local knowledge base. Rephrase the question or add documents under data/knowledge/.",
                "sources": [],
            }
        best = results[0]
        return {
            "answer": f"Based on {best['source']}: {best['text'][:700].strip()}",
            "sources": [
                {"source": r["source"], "score": r["score"], "excerpt": r["excerpt"]}
                for r in results
            ],
        }
