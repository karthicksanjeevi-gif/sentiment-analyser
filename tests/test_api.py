import io
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_frontend_index():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Customer Review Sentiment" in response.text
        assert "Chart.js" in response.text or "chart.umd.min.js" in response.text


def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Excellent product and helpful service"})
        assert response.status_code == 200
        body = response.json()
        assert body["label"] == "positive"
        assert body["confidence"] >= 0.5
        assert body["model_version"] == "1.0.0"


def test_rag_returns_sources():
    with TestClient(app) as client:
        response = client.post("/rag/query", json={"question": "How does the CI pipeline work?", "top_k": 3})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["sources"]
        assert any("cicd" in item["source"] for item in body["sources"])


def test_rag_unknown_question():
    with TestClient(app) as client:
        response = client.post("/rag/query", json={"question": "zqxv blorf plim", "top_k": 3})
        assert response.status_code == 200
        assert response.json()["sources"] == []


def test_model_metrics():
    with TestClient(app) as client:
        response = client.get("/model/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data
        assert "model_version" in data


def test_reviews_upload_csv():
    csv_data = "review\nGreat quality and fast delivery\nTerrible product and broke immediately\n"
    files = {"file": ("test_reviews.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    with TestClient(app) as client:
        response = client.post("/reviews/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_reviews"] == 2
        assert data["positive_count"] == 1
        assert data["negative_count"] == 1
        assert len(data["reviews"]) == 2
        assert "positive_top_words" in data
        assert "negative_top_words" in data
        assert "positive_confidence_distribution" in data
        assert "negative_confidence_distribution" in data


def test_reviews_analyze_text():
    text = "Really loved this purchase, very good quality!\nAwful experience, never buying again."
    with TestClient(app) as client:
        response = client.post("/reviews/analyze-text", json={"text": text})
        assert response.status_code == 200
        data = response.json()
        assert data["total_reviews"] == 2
        assert data["positive_count"] >= 1
        assert data["negative_count"] >= 1


def test_reviews_sample():
    with TestClient(app) as client:
        response = client.get("/reviews/sample")
        assert response.status_code == 200
        data = response.json()
        assert data["total_reviews"] >= 10
        assert data["positive_count"] > 0
        assert data["negative_count"] > 0
        assert len(data["positive_top_words"]) > 0
        assert len(data["negative_top_words"]) > 0


def test_reviews_upload_pdf():
    from pathlib import Path
    pdf_path = Path(__file__).resolve().parents[1] / "data" / "sample_customer_reviews.pdf"
    if not pdf_path.exists():
        return
    files = {"file": ("sample_customer_reviews.pdf", pdf_path.read_bytes(), "application/pdf")}
    with TestClient(app) as client:
        response = client.post("/reviews/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_reviews"] == 100
        assert data["positive_count"] == 50
        assert data["negative_count"] == 50
        assert len(data["reviews"]) == 100

