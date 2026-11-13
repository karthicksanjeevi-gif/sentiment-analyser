from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from typing import Any

import pypdf

from app.schemas import AnalyzedReview, DocumentAnalysisResponse, WordFrequency

STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "cannot", "could", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "it's", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not",
    "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won't", "would", "you", "your", "yours", "yourself",
    "yourselves", "also", "really", "get", "got", "even", "much", "many", "still", "like", "one", "two",
    "review", "reviews", "dataset", "page", "educational", "synthetic", "customer", "comments"
}


def clean_review_text(text: str) -> str:
    """Strip numbering, prefixes like 'Review 001:' or quotes, and normalize whitespace."""
    cleaned = text.strip()
    # Strip Review XXX: or Review XXX -
    cleaned = re.sub(r"^review\s+\d+[:\-\.]\s*", "", cleaned, flags=re.IGNORECASE)
    # Strip numbered list prefix like 1. or 1)
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
    # Strip bullet points
    cleaned = re.sub(r"^[\*\-•]\s*", "", cleaned)
    # Strip wrapping quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    # Strip trailing category/sentiment metadata e.g. DELIVERY / QUALITY POSITIVE
    cleaned = re.sub(r"\s+[A-Z\s/]+(?:POSITIVE|NEGATIVE)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:POSITIVE|NEGATIVE)\s+", "", cleaned, flags=re.IGNORECASE)
    # Normalize internal whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    """Extract and split clean customer reviews from a PDF document."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return []

    # Remove standard header/footer and instructions noise
    cleaned_full_text = re.sub(r"Customer Feedback Audit Report\s+Page\s+\d+\s+of\s+\d+", "", full_text, flags=re.I)
    cleaned_full_text = re.sub(r"Customer Reviews Dataset\s+Synthetic educational data", "", cleaned_full_text, flags=re.I)
    cleaned_full_text = re.sub(r"Page\s+\d+(\s+of\s+\d+)?", "", cleaned_full_text, flags=re.I)
    cleaned_full_text = re.sub(r"How to use:.*?(?=(?:Review\s+\d+|\n\n|$))", "", cleaned_full_text, flags=re.I | re.DOTALL)
    cleaned_full_text = re.sub(r"Dataset note:.*?(?=(?:Review\s+\d+|\n\n|$))", "", cleaned_full_text, flags=re.I | re.DOTALL)

    reviews: list[str] = []

    # 1. Check for labeled pattern 'Review XXX: ...'
    pattern_review = re.compile(r"Review\s+\d+[:\-\.]\s*(.*?)(?=(?:Review\s+\d+[:\-\.]|$))", re.I | re.DOTALL)
    matches = pattern_review.findall(cleaned_full_text)
    if len(matches) > 1:
        for m in matches:
            cleaned = clean_review_text(m)
            if len(cleaned) > 5:
                reviews.append(cleaned)
        if reviews:
            return reviews

    # 2. Numbered table entries (e.g. 1 Review text... POSITIVE/NEGATIVE or 1. Review text...)
    table_start = re.search(r"#\s*Feedback Text", cleaned_full_text, re.I)
    working_text = cleaned_full_text[table_start.start():] if table_start else cleaned_full_text

    lines = working_text.splitlines()
    entries: list[str] = []
    current_num = None
    current_tokens: list[str] = []

    ignore_prefixes = (
        "Feedback Log", "# Feedback Text", "Comprehensive Feedback", "TOTAL REVIEWS",
        "POSITIVE SENTIMENT", "NEGATIVE SENTIMENT", "TOP CATEGORY", "Executive Summary",
        "Category & Sentiment", "Product Quality", "Delivery & Packaging", "Customer Support",
        "App & Platform"
    )

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if any(s.lower().startswith(p.lower()) for p in ignore_prefixes):
            continue

        num_match = re.match(r"^(\d{1,4})[\.\)]?\s+([A-Za-z\"\'\`].*)", s)
        if num_match:
            if current_num is not None:
                entries.append(" ".join(current_tokens))
            current_num = int(num_match.group(1))
            current_tokens = [num_match.group(2)]
        elif current_num is not None:
            current_tokens.append(s)

    if current_num is not None and current_tokens:
        entries.append(" ".join(current_tokens))

    if len(entries) > 1:
        for entry in entries:
            cleaned = clean_review_text(entry)
            if len(cleaned) > 5:
                reviews.append(cleaned)
        if reviews:
            return reviews

    # 3. Check for double newline separated paragraphs
    paragraphs = re.split(r"\n\s*\n", cleaned_full_text)
    if len(paragraphs) > 1:
        for p in paragraphs:
            cleaned = clean_review_text(p)
            if len(cleaned) > 10:
                reviews.append(cleaned)
        if reviews:
            return reviews

    # 4. Fallback: single non-empty line splits
    for line in cleaned_full_text.splitlines():
        cleaned = clean_review_text(line)
        if len(cleaned) > 15:
            reviews.append(cleaned)

    return reviews


def parse_reviews_from_text(content: str, filename: str = "") -> list[str]:
    """Parse a list of customer review strings from CSV, JSON, or plain text."""
    filename_lower = filename.lower()
    reviews: list[str] = []

    if filename_lower.endswith(".json") or (content.strip().startswith("[") and content.strip().endswith("]")):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        cleaned = clean_review_text(item)
                        if cleaned:
                            reviews.append(cleaned)
                    elif isinstance(item, dict):
                        for key in ["review", "text", "comment", "feedback", "body", "content", "review_text"]:
                            if key in item and isinstance(item[key], str) and item[key].strip():
                                cleaned = clean_review_text(item[key])
                                if cleaned:
                                    reviews.append(cleaned)
                                break
            if reviews:
                return reviews
        except Exception:
            pass

    # Try parsing as CSV
    if filename_lower.endswith(".csv") or ("," in content and "\n" in content):
        try:
            reader = csv.reader(io.StringIO(content))
            rows = [row for row in reader if row]
            if rows:
                header = [col.strip().lower() for col in rows[0]]
                target_col_idx = -1
                for candidate in ["review", "text", "comment", "customer_review", "feedback", "body", "review_text", "reviews"]:
                    if candidate in header:
                        target_col_idx = header.index(candidate)
                        break

                if target_col_idx != -1:
                    for row in rows[1:]:
                        if len(row) > target_col_idx and row[target_col_idx].strip():
                            cleaned = clean_review_text(row[target_col_idx])
                            if cleaned:
                                reviews.append(cleaned)
                elif len(rows) > 1 and len(rows[0]) == 1:
                    start_idx = 1 if rows[0][0].lower() in {"review", "text", "comment", "feedback"} else 0
                    for row in rows[start_idx:]:
                        if row and row[0].strip():
                            cleaned = clean_review_text(row[0])
                            if cleaned:
                                reviews.append(cleaned)
        except Exception:
            pass

    if reviews:
        return reviews

    # Default fallback: plain text (numbered or one review per line)
    # Check if lines start with numbers: e.g. 1. ... 2. ...
    entries: list[str] = []
    current_tokens: list[str] = []
    has_numbered_prefix = False

    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^(\d{1,4})[\.\)]\s+(.*)", s)
        if m:
            has_numbered_prefix = True
            if current_tokens:
                entries.append(" ".join(current_tokens))
            current_tokens = [m.group(2)]
        elif has_numbered_prefix:
            current_tokens.append(s)
        else:
            cleaned = clean_review_text(s)
            if cleaned and len(cleaned) > 2:
                entries.append(cleaned)

    if has_numbered_prefix and current_tokens:
        entries.append(" ".join(current_tokens))

    for item in entries:
        cleaned = clean_review_text(item)
        if cleaned and len(cleaned) > 2:
            reviews.append(cleaned)

    return reviews


def parse_reviews_from_file(raw_bytes: bytes, filename: str = "") -> list[str]:
    """Parse reviews from uploaded bytes supporting PDF, CSV, JSON, and TXT formats."""
    if filename.lower().endswith(".pdf") or raw_bytes.startswith(b"%PDF"):
        pdf_reviews = parse_pdf_bytes(raw_bytes)
        if pdf_reviews:
            return pdf_reviews

    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("latin-1", errors="ignore")

    return parse_reviews_from_text(content, filename=filename)


def extract_top_words(texts: list[str], top_n: int = 10) -> list[WordFrequency]:
    """Find the most frequently used meaningful words in the given texts."""
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        meaningful_tokens = [t for t in tokens if t not in STOPWORDS]
        counter.update(meaningful_tokens)

    return [WordFrequency(word=word, count=count) for word, count in counter.most_common(top_n)]


def compute_confidence_buckets(confidences: list[float]) -> dict[str, int]:
    """Bucket confidence levels into High (>=0.80), Moderate (0.65-0.79), and Low (0.50-0.64)."""
    buckets = {"High (>=80%)": 0, "Moderate (65-79%)": 0, "Low (50-64%)": 0}
    for c in confidences:
        if c >= 0.80:
            buckets["High (>=80%)"] += 1
        elif c >= 0.65:
            buckets["Moderate (65-79%)"] += 1
        else:
            buckets["Low (50-64%)"] += 1
    return buckets


def analyze_reviews(model: Any, texts: list[str]) -> DocumentAnalysisResponse:
    """Analyze a collection of reviews, compute stats, charts distributions, and top words."""
    if not texts:
        return DocumentAnalysisResponse(
            total_reviews=0,
            positive_count=0,
            negative_count=0,
            positive_percentage=0.0,
            negative_percentage=0.0,
            average_confidence=0.0,
            positive_top_words=[],
            negative_top_words=[],
            positive_confidence_distribution={"High (>=80%)": 0, "Moderate (65-79%)": 0, "Low (50-64%)": 0},
            negative_confidence_distribution={"High (>=80%)": 0, "Moderate (65-79%)": 0, "Low (50-64%)": 0},
            reviews=[],
        )

    analyzed: list[AnalyzedReview] = []
    positive_texts: list[str] = []
    negative_texts: list[str] = []
    pos_confidences: list[float] = []
    neg_confidences: list[float] = []
    all_confidences: list[float] = []

    probabilities = model.predict_proba(texts)
    classes = [str(c) for c in model.classes_]

    for text, proba in zip(texts, probabilities):
        best_idx = int(proba.argmax())
        label = classes[best_idx]
        confidence = round(float(proba[best_idx]), 4)
        all_confidences.append(confidence)

        analyzed.append(AnalyzedReview(text=text, label=label, confidence=confidence))

        if label == "positive":
            positive_texts.append(text)
            pos_confidences.append(confidence)
        else:
            negative_texts.append(text)
            neg_confidences.append(confidence)

    total = len(texts)
    pos_count = len(positive_texts)
    neg_count = len(negative_texts)

    pos_pct = round((pos_count / total) * 100, 1) if total else 0.0
    neg_pct = round((neg_count / total) * 100, 1) if total else 0.0
    avg_conf = round(sum(all_confidences) / total, 4) if total else 0.0

    return DocumentAnalysisResponse(
        total_reviews=total,
        positive_count=pos_count,
        negative_count=neg_count,
        positive_percentage=pos_pct,
        negative_percentage=neg_pct,
        average_confidence=avg_conf,
        positive_top_words=extract_top_words(positive_texts, top_n=10),
        negative_top_words=extract_top_words(negative_texts, top_n=10),
        positive_confidence_distribution=compute_confidence_buckets(pos_confidences),
        negative_confidence_distribution=compute_confidence_buckets(neg_confidences),
        reviews=analyzed,
    )
