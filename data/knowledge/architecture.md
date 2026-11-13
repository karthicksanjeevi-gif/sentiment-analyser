# Project Architecture

This MLOps + RAG Lab contains a scikit-learn text classifier and a retrieval-augmented question-answering component. The FastAPI service exposes both capabilities over HTTP.

The ML workflow starts with data/train.csv. Running python scripts/train.py validates the dataset, creates a stratified train/test split, trains a TF-IDF plus Logistic Regression pipeline, evaluates it, and writes artifacts/model.joblib and artifacts/metrics.json. The API loads the trained artifact at startup. The /predict endpoint returns a predicted label, confidence score, and model version.

The RAG workflow reads Markdown files from data/knowledge. It splits large text into chunks, builds a TF-IDF index, retrieves relevant chunks using cosine similarity, and returns an extractive answer plus source excerpts. It does not call an external language model. Add more Markdown files to extend the local knowledge base.

The /health endpoint reports whether the model and RAG index loaded. The /model/metrics endpoint returns evaluation metrics from the latest training run. The /docs page provides interactive API documentation.
