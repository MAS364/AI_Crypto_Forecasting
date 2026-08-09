# Inference-only image for the FastAPI backend.
# Uses already-saved models and results. No training is performed at runtime.

FROM python:3.12-slim

WORKDIR /app

# System dependency required by some ML packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install backend dependencies.
COPY deployment/requirements.txt ./deployment/requirements.txt
RUN pip install --no-cache-dir -r deployment/requirements.txt

# Application source.
COPY src/ ./src/

# FastAPI deployment code.
COPY deployment/app.py ./deployment/app.py
COPY deployment/csv_inference.py ./deployment/csv_inference.py

# Saved trained models.
COPY models/trained/ ./models/trained/

# Saved autoencoder artefacts.
COPY results/autoencoder/models/ ./results/autoencoder/models/

# Evaluation results used by API endpoints.
COPY results/evaluation/tables/ ./results/evaluation/tables/

# Saved modelling metrics and predictions used by comparison endpoints.
COPY results/modeling/metrics/ ./results/modeling/metrics/
COPY results/modeling/predictions/ ./results/modeling/predictions/

# Statistical evidence.
COPY results/causality_analysis/tables/ ./results/causality_analysis/tables/

# Permutation feature-importance results.
COPY results/feature_importance/ ./results/feature_importance/

# Cloud Run provides PORT automatically.
EXPOSE 8080

CMD ["sh", "-c", "uvicorn deployment.app:app --host 0.0.0.0 --port ${PORT:-8080}"]