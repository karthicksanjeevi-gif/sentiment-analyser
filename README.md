# MLOps + RAG Lab

A runnable student project combining an ML training pipeline, FastAPI inference, local RAG retrieval, automated tests, Docker, and GitHub Actions CI/CD.

## Features


- **Customer Review Document Analyzer**: Upload PDF, CSV, TXT, or JSON files containing reviews to get:
  - Batch sentiment classification for every review
  - Separate interactive charts for positive and negative review confidence
  - Overall sentiment ratio (doughnut chart) and counts
  - Top most frequently used words for positive vs. negative reviews (with stop-words filtered)
  - Searchable, filterable review table with CSV export
- **Web Frontend Dashboard**: Modern interactive UI running directly at `http://127.0.0.1:8000/`
- **FastAPI Endpoints**:
  - `POST /reviews/upload`: Multipart document upload & analytics
  - `POST /reviews/analyze-text`: Analyze pasted review batch
  - `GET /reviews/sample`: Instant 20-sample analysis demo
  - `POST /predict`: Single review sentiment prediction and confidence
  - `POST /rag/query`: Local retrieval-augmented Q&A over Markdown documents
  - `GET /health` and `GET /model/metrics`
- Automated Pytest suite (10 tests covering UI, batch analyzer, PDF parsing, RAG, and model inference)

## Run locally (Windows PowerShell)
Python 3.11+ is supported.

```powershell
python -m pip install -r requirements-dev.txt
python scripts/train.py
python -m uvicorn app.main:app --reload
```

Open the web frontend in your browser:
👉 **http://127.0.0.1:8000/**

Interactive Swagger API docs are available at:
👉 **http://127.0.0.1:8000/docs**


Example `/predict` body:
```json
{"text":"The support team was helpful and solved my issue quickly."}

'``

Example `/rag/query` body:
```json
{"question":"How does the CI pipeline work?","top_k":3}
```

## Run tests
```powershell
pytest -q
```

## Run with Docker
```bash
docker compose up --build
```
Then open http://localhost:8000/docs.

## CI/CD Pipeline Integration (GitHub Actions)
The repository includes an automated CI/CD pipeline in [`.github/workflows/ci-cd.yml`](file:///.github/workflows/ci-cd.yml).

### How the CI/CD Pipeline Works
1. **Trigger**: Triggers automatically whenever you push code or open a pull request against `main`.
2. **Automated CI (Continuous Integration)**:
   - Sets up Python 3.11 with pip caching.
   - Installs all dependencies from `requirements-dev.txt`.
   - Runs model training (`python scripts/train.py`) and generates fresh model artifacts.
   - Runs automated unit tests (`pytest -q`), verifying all 10 test suites pass.
3. **Automated CD (Continuous Delivery & Docker Publishing)**:
   - Builds the production Docker image.
   - On pushes to `main`, authenticates with GitHub Container Registry (GHCR) and publishes tags (`latest` and short commit SHA).

### How to Link and Enable the CI/CD Pipeline on GitHub
1. Initialize a git repository and commit your files:
   ```powershell
   git init
   git add .
   git commit -m "feat: complete MLOps review analyzer with export and CI/CD"
   ```
2. Create a repository on GitHub (e.g. `mlops-rag-project`), link the remote, and push to `main`:
   ```powershell
   git remote add origin https://github.com/<YOUR_USERNAME>/mlops-rag-project.git
   git branch -M main
   git push -u origin main
   ```
3. Enable Package Write Permissions for GitHub Actions:
   - On GitHub, go to your repository **Settings** -> **Actions** -> **General**.
   - Under **Workflow permissions**, select **"Read and write permissions"**.
   - Click **Save**.
4. Check Pipeline Runs:
   - Click the **Actions** tab on your GitHub repository to watch the pipeline train the model, run tests, build the Docker container, and deploy the package.

## RAG note
The RAG implementation uses TF-IDF + cosine similarity and returns relevant source passages. Its answer step is extractive, not an LLM. This is deliberate so the project works locally without keys or paid services. For an advanced version, replace retrieval with embeddings/FAISS or Chroma and pass retrieved context to a local or hosted LLM.

## Structure
```text
app/                 FastAPI app, schemas, RAG engine
data/train.csv       Small synthetic training dataset
data/knowledge/      RAG source documents
scripts/train.py     Training + evaluation + artifact creation
artifacts/           Generated model and metrics
tests/               API tests
.github/workflows/   CI/CD pipeline
Dockerfile
compose.yaml
```

The included dataset is small and synthetic; its metrics are only a demonstration, not evidence of real-world model performance. Before production, add authentication, rate limiting, observability, secret management, model/data versioning, and deployment approvals.
