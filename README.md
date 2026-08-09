# AI for Cryptocurrency Time Series Analysis

**Correlation, Causality, and Predictive Modelling for Bitcoin and Ethereum**

MSc Data Science dissertation, University of Surrey. The project forecasts Bitcoin and Ethereum prices using market data together with Google Trends search interest and the Crypto Fear & Greed Index, across 1-, 7-, and 14-day forecast horizons.

## Research Questions

1. Can machine learning and deep learning models outperform classical statistical models in forecasting Bitcoin and Ethereum prices?
2. Does incorporating Google Trends and the Crypto Fear & Greed Index improve forecasting performance?
3. Which forecasting model provides the most accurate and reliable predictions?
4. What relationships exist between cryptocurrency prices and the selected external variables?
5. Do the correlation and Granger causality findings align with the forecasting results?

## Reproducibility Design

The repository deliberately provides two reproducible routes over the same scientific implementation:

- **`notebooks/` — research route.** Step-by-step experimentation, methodology explanation, figures, saved results, and dissertation evidence.
- **`src/` — modular route.** Reusable implementations for preprocessing, analysis, feature engineering, autoencoding, model training, evaluation, inference, and deployment. The notebooks import these modules rather than maintaining a separate copy of the core modelling logic.

The intended notebook workflow is:

```text
01 Data Collection
 -> 02 Data Understanding
 -> 03 Data Preparation
 -> 04 EDA
 -> 05 Correlation & Granger Causality
 -> 06 Feature Engineering
 -> 06b Autoencoder
 -> 07 Model Development
 -> 08 Evaluation
 -> 09 Feature Importance Analysis
```

## Notebook Order

| # | Notebook | Purpose |
|---|---|---|
| 01 | `01_data_collection.ipynb` | Collects Bitcoin/Ethereum market data, Google Trends data, and Fear & Greed data |
| 02 | `02_data_understanding.ipynb` | Examines structure, types, coverage, and missingness |
| 03 | `03_data_preparation.ipynb` | Cleans and merges the collected datasets |
| 04 | `04_eda.ipynb` | Exploratory data analysis |
| 05 | `05_correlation_and_granger.ipynb` | Correlation and Granger-causality analysis |
| 06 | `06_feature_engineering.ipynb` | Builds four feature configurations across three horizons and two assets (24 feature CSVs) |
| 06b | `06b_autoencoder.ipynb` | Creates autoencoder-compressed representations for the feature configurations |
| 07 | `07_model_development.ipynb` | Trains/evaluates Naive, ARIMA, Random Forest, XGBoost, and LSTM models |
| 08 | `08_evaluation.ipynb` | Loads saved model outputs and performs comparative evaluation; it does not retrain models |
| 09 | `09_feature_importance_analysis.ipynb` | Analyses permutation feature importance and links model evidence to the research questions |

## Environment Setup

The tested development environment uses Python 3.12. Create a clean environment from the repository root:

```bash
git clone <repository-url>
cd AI_Crypto_Forecasting
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All executable paths are resolved relative to the repository/project root; no personal machine path should be required.

## Route A — Reproduce with Notebooks

Start Jupyter from the repository root:

```bash
jupyter lab
```

Run the notebooks in this order:

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 06b -> 07 -> 08 -> 09
```

The full model-development stage is computationally expensive because it evaluates multiple assets, horizons, feature configurations, model families, and LSTM hyperparameter configurations. The repository retains the research configuration rather than reducing the experiment for convenience.

## Route B — Reproduce with the Modular Pipeline

The modular route uses the same reusable `src/` implementations.

### Training

```bash
python -m src.pipeline train
```

This command builds missing engineered features and autoencoder representations before running model training. Full training can take a substantial amount of time.

### Evaluation

```bash
python -m src.pipeline evaluate
```

Evaluation loads saved predictions and metrics and does not retrain models.

### Inference / Prediction

```bash
python -m src.pipeline predict \
    --horizon 1 \
    --asset Bitcoin \
    --feature-set primary \
    --representation Original \
    --model "Random Forest" \
    --input path/to/new_features.csv
```

Prediction loads a saved fitted model and does not retrain it.

## Trained-Model Artefacts

`models/trained/` is the **canonical model registry for reusable inference and deployment**. It contains the deployment-ready fitted models and `manifest.json`.

Notebook 07 also retains experiment-local fitted artefacts under `results/modeling/fitted_models/` as research outputs. Notebook 07 publishes/migrates deployment-ready model artefacts into `models/trained/`; downstream inference and deployment should use `models/trained/`, not the experiment-local directory.

This distinction is intentional:

```text
results/modeling/fitted_models/  -> research/experiment artefacts
models/trained/                  -> canonical inference/deployment artefacts
```

## Provided vs Regenerated Artefacts

This submission includes generated artefacts so the reported results can be inspected without forcing an examiner to immediately repeat the complete, expensive training run. They are also reproducible from the notebook or modular routes above.

Provided in the repository/ZIP:

- **24 engineered feature CSVs** in `data/processed/features/`.
- **72 encoded split CSVs** in `results/autoencoder/encoded_features/`.
- saved autoencoder artefacts and evaluation outputs;
- saved model-development predictions/metrics;
- **deployment-ready trained models** under `models/trained/`;
- generated tables and figures used in the dissertation analysis.

When reproducing from scratch, these generated outputs may be regenerated by the corresponding notebooks/modules. Existing generated artefacts are provided as verification evidence, not as hidden prerequisites for understanding the code.

## Testing

From the repository root:

```bash
python -m pytest tests/ -q
```

`pytest.ini` already adds the repository root to Python's import path, so an additional `PYTHONPATH=.` prefix is not required.

The test suite combines lightweight unit tests with integration tests. Integration tests that require generated feature/model artefacts skip cleanly when those artefacts have not yet been produced, while core unit tests use synthetic data and remain runnable on a clean clone.

## Deployment

The deployment consists of a FastAPI backend and a Streamlit frontend.

Because the main `requirements.txt` now declares the frontend dependencies, a normal project installation is sufficient.

### Local

Terminal 1:

```bash
source .venv/bin/activate
uvicorn deployment.app:app --reload
```

Terminal 2:

```bash
source .venv/bin/activate
streamlit run deployment/frontend/Home.py
```

Open:

- Streamlit: `http://localhost:8501`
- FastAPI documentation: `http://127.0.0.1:8000/docs`

### Docker Compose

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

The current deployment exposes forecasting, model registry, evaluation/research evidence, feature-importance comparison, and ARIMA/Naive information endpoints. SHAP is not part of the current implementation; feature importance is based on the project's permutation-importance workflow.

## Main Output Locations

| Directory | Contents |
|---|---|
| `data/processed/` | Prepared merged data |
| `data/processed/features/` | 24 engineered feature CSVs used by the modelling stage |
| `results/autoencoder/` | Encoded features, autoencoder artefacts, and reconstruction metrics |
| `results/modeling/predictions/` | Model predictions |
| `results/modeling/metrics/` | Model metrics |
| `results/modeling/fitted_models/` | Experiment-local fitted-model artefacts retained for research traceability |
| `models/trained/` | Canonical fitted models used by reusable inference/deployment plus `manifest.json` |
| `results/evaluation/` | Evaluation tables, figures, and reports |
| `results/feature_importance/` | Permutation feature-importance raw/processed results and figures |
| `results/causality_analysis/` | Correlation/Granger-related tables and figures |

## Repository Hygiene

Generated local development metadata is excluded through `.gitignore`, including virtual environments, Python bytecode/cache directories, `.pytest_cache/`, `.claude/`, notebook checkpoints, logs, and `.DS_Store` files. The `.streamlit/` directory is intentionally retained because it contains project UI configuration.

## Notes on Data Collection

Notebook 01 may require internet access because it collects external market/search/sentiment data. For inspection and downstream reproducibility, the repository includes prepared/generated data artefacts used by later stages. If an external provider changes availability or historical responses, the provided prepared data offers a fixed reference for the reported dissertation experiment.

## Key Findings

- Random Forest, XGBoost, and LSTM improve on a naive persistence baseline in many configurations, while ARIMA remains a strong classical benchmark.
- Google Trends and Fear & Greed features yield relatively small and configuration-dependent forecasting changes, consistent with the supporting statistical analyses.
- Autoencoder-compressed features can improve selected configurations but do not provide a universal advantage across all assets and horizons.

Full model comparison and discussion are in `notebooks/08_evaluation.ipynb`; feature-importance evidence is in `notebooks/09_feature_importance_analysis.ipynb` and `results/feature_importance/`.
