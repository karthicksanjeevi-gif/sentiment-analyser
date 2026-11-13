# CI/CD Pipeline

Continuous Integration (CI) automatically validates code changes before they are merged. This repository uses GitHub Actions. On a push or pull request, the workflow checks out source code, installs Python dependencies, trains the model, runs pytest, and builds the Docker image. If a required step fails, the workflow fails.

Continuous Delivery or Deployment (CD) packages the application as a Docker image. On pushes to the main branch, the workflow can publish the image to GitHub Container Registry (GHCR), using the repository's built-in GITHUB_TOKEN. The repository must allow GitHub Actions to write packages.

A typical workflow is: edit code, commit, push a branch, open a pull request, wait for CI, merge after passing checks, and publish a container from main. For production, add staging, approvals, rollback, secret management, and smoke tests.
