# Inference-only image: serves predictions from already-saved models via
# the FastAPI app in deployment/app.py. Never trains anything at runtime.
#
# NOTE ON MODEL WEIGHTS: this Dockerfile bakes the trained models
# (models/trained/) into the image, which keeps deployment simple (one
# self-contained image, no external volume to configure) at the cost of a
# larger image. For a project this size that's the right tradeoff; if the
# model directory grows substantially, switching to a mounted volume
# (`docker run -v ./models:/app/models`) would be the next step rather
# than rebuilding this Dockerfile from scratch.

FROM python:3.11-slim

WORKDIR /app

# NOTE: results/shap/tables/ must exist before building (run
# 09_model_explainability_SHAP.ipynb first) -- Docker's COPY fails on a
# missing source directory. If you haven't run that notebook yet, create
# an empty results/shap/tables/ directory first so the build doesn't fail;
# the /shap-summary endpoint will just return 404 until real results exist.

# System dependencies for xgboost/tensorflow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY deployment/requirements.txt ./deployment/requirements.txt
RUN pip install --no-cache-dir -r deployment/requirements.txt

# Only what inference actually needs -- src/ (code), models/trained/
# (saved weights + manifest), results/autoencoder/models/ (saved encoders),
# plus the real saved evaluation/causality/SHAP tables the API's read-only
# endpoints (/decision-summary, /research-evidence, /shap-summary) serve.
# Deliberately does NOT copy notebooks/, data/raw/, results/eda/, etc.
COPY src/ ./src/
COPY deployment/app.py deployment/csv_inference.py ./deployment/
COPY models/trained/ ./models/trained/
COPY results/autoencoder/models/ ./results/autoencoder/models/
COPY results/evaluation/tables/ ./results/evaluation/tables/
COPY results/causality_analysis/tables/ ./results/causality_analysis/tables/
COPY results/shap/tables/ ./results/shap/tables/

EXPOSE 8000

CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
