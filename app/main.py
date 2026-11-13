from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import analyze_reviews, parse_reviews_from_file, parse_reviews_from_text
from app.rag import LocalRAG
from app.schemas import (
    AnalyzeTextRequest,
    DocumentAnalysisResponse,
    PredictRequest,
    PredictResponse,
    RAGRequest,
    RAGResponse,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLE_REVIEWS_PATH = ROOT / "data" / "sample_customer_reviews.csv"

_model: Any = None
_model_version = "unavailable"
_rag: LocalRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _rag
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        _model = bundle["model"]
        _model_version = bundle.get("model_version", "1.0.0")
    _rag = LocalRAG()
    yield


app = FastAPI(
    title="MLOps + RAG Lab API",
    description="ML inference, customer review document analyzer, local RAG, tests, Docker, and CI/CD.",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_file)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model": "loaded" if _model is not None else "missing",
        "rag": "loaded" if _rag is not None else "missing",
    }


@app.get("/model/metrics")
def model_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics missing; run python scripts/train.py first.")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python scripts/train.py and restart the API.")
    probabilities = _model.predict_proba([request.text])[0]
    index = int(probabilities.argmax())
    return PredictResponse(
        label=str(_model.classes_[index]),
        confidence=round(float(probabilities[index]), 4),
        model_version=_model_version,
    )


@app.post("/rag/query", response_model=RAGResponse)
def rag_query(request: RAGRequest) -> RAGResponse:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG index is not loaded.")
    return RAGResponse(**_rag.query(request.question, request.top_k))


@app.post("/reviews/upload", response_model=DocumentAnalysisResponse)
async def upload_reviews_document(file: UploadFile = File(...)) -> DocumentAnalysisResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python scripts/train.py first.")

    raw_bytes = await file.read()
    reviews = parse_reviews_from_file(raw_bytes, filename=file.filename or "")
    if not reviews:
        raise HTTPException(status_code=400, detail="No readable review texts found in the uploaded file.")

    return analyze_reviews(_model, reviews)


@app.post("/reviews/analyze-text", response_model=DocumentAnalysisResponse)
def analyze_raw_reviews(request: AnalyzeTextRequest) -> DocumentAnalysisResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python scripts/train.py first.")

    reviews = parse_reviews_from_text(request.text, filename="raw_text.txt")
    if not reviews:
        raise HTTPException(status_code=400, detail="No valid reviews provided.")

    return analyze_reviews(_model, reviews)


@app.get("/reviews/sample", response_model=DocumentAnalysisResponse)
def get_sample_analysis() -> DocumentAnalysisResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run python scripts/train.py first.")

    if not SAMPLE_REVIEWS_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample reviews file missing.")

    content = SAMPLE_REVIEWS_PATH.read_text(encoding="utf-8")
    reviews = parse_reviews_from_text(content, filename="sample.csv")
    return analyze_reviews(_model, reviews)
