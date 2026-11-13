# Testing and Operations

Tests verify that the API health check responds, prediction returns a valid label and confidence score, and the RAG endpoint returns relevant source passages or a no-match response. Run tests with pytest -q.

The training script writes a model artifact and metrics JSON file. The metrics endpoint exposes hold-out accuracy and a classification report. The sample dataset is synthetic and small, so its metrics are only a demonstration.

For production, track experiments and artifacts with MLflow, version datasets with DVC, add structured logs and metrics, use a model registry, monitor drift, and establish approval and rollback before deploying a new model.
