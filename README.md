# AI-Driven Bitcoin & Ethereum Forecasting

**Correlation, Causality, and Predictive Modelling for Bitcoin and Ethereum**

MSc Data Science dissertation, University of Surrey. The project forecasts Bitcoin and Ethereum returns using market data together with Google Trends search interest and the Crypto Fear & Greed Index, comparing statistical, machine learning, and deep learning models across 1-, 7-, and 14-day forecast horizons, with permutation feature importance used to interpret the trained models.

## Project Overview

Cryptocurrency prices are volatile and only partially explained by their own history. This project evaluates whether adding two external, publicly available signals — Google Trends search interest and the Crypto Fear & Greed Index — improves short-horizon forecasting of Bitcoin and Ethereum returns, and whether machine learning and deep learning models outperform classical statistical baselines when doing so. Five model families are compared under identical evaluation conditions: a naive persistence baseline, ARIMA, Random Forest, XGBoost, and an LSTM. An autoencoder-compressed feature representation is evaluated alongside the original engineered features as a secondary comparison. Permutation feature importance is used to interpret which engineered features the tree-based models actually rely on, cross-referenced against the project's own correlation and Granger causality analysis rather than treated as an independent, unrelated result.

## Research Questions

1. Can machine learning and deep learning models outperform classical statistical models in forecasting Bitcoin and Ethereum prices?
2. Does incorporating Google Trends and the Crypto Fear & Greed Index improve forecasting performance?
3. Which forecasting model provides the most accurate and reliable predictions?
4. What relationships exist between cryptocurrency prices and the selected external variables?
5. Do the correlation and Granger causality findings align with the forecasting results?

## Key Contributions

- A controlled comparison of five model families (Naive, ARIMA, Random Forest, XGBoost, LSTM) across two assets, three forecast horizons, and four feature configurations, evaluated identically on chronologically held-out test data.
- A statistically grounded feature-selection process: which external variables enter which feature configuration is determined by the project's own stationarity, correlation, and Granger causality analysis (Notebook 05), not by assumption.
- A secondary comparison between the original engineered features and an autoencoder-compressed representation, evaluated under the same conditions as the primary comparison.
- Permutation feature importance for Random Forest and XGBoost, cross-referenced directly against the correlation and Granger causality evidence rather than reported in isolation.
- A reproducible modular pipeline (`src/`) that mirrors the notebook implementation exactly — the notebooks import from `src/`, they do not maintain a separate copy of the modelling logic.
- A working deployment (FastAPI backend, Streamlit frontend) that serves predictions from the same saved artefacts the evaluation results are drawn from.

## Methodology

```text
Data collection
  -> data understanding / preparation
  -> exploratory data analysis
  -> stationarity, correlation, and Granger causality analysis
  -> feature engineering (4 configurations x 3 horizons x 2 assets)
  -> autoencoder representation (Original vs. Autoencoder-Encoded)
  -> model development (Naive, ARIMA, Random Forest, XGBoost, LSTM)
  -> evaluation (test-set comparison, baseline comparison, Diebold-Mariano tests)
  -> permutation feature importance
  -> deployment (FastAPI + Streamlit)
```

## Data Sources

Raw data is already included in `data/raw/` — this is the frozen snapshot the dissertation results were produced from, not a live feed. The four raw files present are:

- `btc-usd-max.csv`, `eth-usd-max.csv` — Bitcoin and Ethereum market data
- `google_trends.csv` (plus windowed intermediate files) — search interest, collected via the `pytrends` library, confirmed directly in `notebooks/01_data_collection.ipynb`
- `fear_greed.csv` — the Crypto Fear & Greed Index

**A note on precision here rather than assumption**: `requirements.txt` pins `pycoingecko` and `yfinance`, and the raw price filenames follow a naming convention consistent with a CoinGecko export, but neither library is actually imported or called anywhere in the current `notebooks/` or `src/` code. The exact original collection method for the price data could not be confirmed from the code as it currently stands. This does not affect reproducibility of the results below, since the frozen CSVs are what every downstream stage actually reads.

A separate, public repository (`MAS364/AI_Crypto_Forecasting_Data`) hosts a copy of the processed dataset for the deployed application's "download dataset" feature. This dissertation repository itself is not assumed to be public.

**Exact dissertation reproduction** should use the included `data/raw/` and `data/processed/` snapshots as-is. **Re-running `01_data_collection.ipynb`** would fetch current Google Trends data and is a legitimate way to collect a *new* dataset, but will not reproduce this dissertation's exact reported numbers, since search interest and market data are both time-dependent.

## Dataset

The processed, merged dataset (`data/processed/merged_dataset.csv`) contains **1,799 rows and 12 columns**, spanning **2021-06-01 to 2026-05-06**, in daily, chronological order. It combines Bitcoin and Ethereum price/volume/market-cap data with Google Trends search interest and the Fear & Greed Index for the same period.

The modelling target is the **log return**, not the raw price: `log(price_t) - log(price_{t-1})`, established as the stationary representation of price during the correlation and Granger causality analysis (Notebook 05) and used consistently from that point onward. For horizons beyond 1 day, the target is the **cumulative** log return over the horizon — `log(price_{t+h} / price_t)` — not a single day's return sampled `h` days ahead; these are different quantities, and the implementation (`src/analysis/feature_engineering.py`) computes the cumulative form directly. A `{price}_level` reference column is carried through every feature set to reconstruct a price-level forecast from a predicted return afterward; it is explicitly excluded from every model's input matrix and is never used as a predictor.

## Feature Engineering

Feature families implemented in `src/analysis/feature_engineering.py`:

- Price and return features (lags, rolling statistics), computed on log returns
- Volatility features (rolling standard deviation of returns)
- Trading volume features
- Google Trends search-interest features (own-asset)
- Fear & Greed Index / sentiment features

All lag and rolling-window features use `.shift()` in the leakage-safe direction and are computed on already-shifted series, so a given row's feature values never include information from that same row's target period.

## Experimental Design

- **Chronological splitting**: 70% train / 15% validation / 15% test, in date order, with no shuffling. An internal assertion in `src/models/data_prep.py` actively raises if the split function is ever called on a shuffled dataframe.
- **Leakage prevention**: predictor and target construction are implemented in separate functions specifically so a leakage check can verify the target was never used to build a predictor; scalers (for the LSTM and autoencoder) are fit on training data only.
- **Forecast horizons**: 1, 7, and 14 days.
- **Feature configurations**: four per asset per horizon (see below).
- **Representations**: Original engineered features, and a secondary Autoencoder-Encoded representation (8 latent dimensions).
- **Model selection**: Random Forest and XGBoost are tuned via a validation-based grid search; the LSTM uses early stopping on validation loss; the test set is never used for hyperparameter selection or early stopping in any model.

## Models

| Model | Role | Saved fitted artefact |
|---|---|---|
| Naive (Persistence) | Baseline — predicts the next return equals the last observed return | No fitted object; metadata only |
| ARIMA | Classical statistical baseline, refit on a growing history window at each forecasting step | No picklable fitted object; metadata only |
| Random Forest | Tree-based machine learning model | Yes (`.joblib`) |
| XGBoost | Gradient-boosted machine learning model | Yes (`.joblib`) |
| LSTM | Deep learning model on windowed sequences | Yes (`.keras` + scaler) |

Naive and ARIMA are univariate — they never see the engineered feature matrix at all, so their results are identical across all four feature configurations at a given asset/horizon. This is deliberate (confirmed directly in `notebooks/07_model_development.ipynb`): recomputing an identical univariate result four times would be redundant, not a different experiment.

## Feature Configurations

| Internal key | Human-readable name |
|---|---|
| `primary` | Market Only |
| `candidate` | Market + Google Trends |
| `ablation` | Market + Fear & Greed Index |
| `combined` | Market + Google Trends + Fear & Greed Index |

This mapping (`src/models/feature_naming.py`) is used consistently across the notebooks, the modular pipeline, and the deployment layer.

## Representation Learning

Two representations of the same underlying features are compared:

- **Original** — the named, interpretable engineered features (price/return, volatility, volume, Google Trends, Fear & Greed), evaluated for all five models.
- **Autoencoder-Encoded** — those same features compressed into 8 unnamed latent dimensions by a trained autoencoder (`06b_autoencoder.ipynb`), evaluated only for Random Forest, XGBoost, and LSTM. Naive and ARIMA were never run through the autoencoder branch, since both are univariate and have no feature matrix to encode.

Permutation feature importance is only meaningful for the Original representation — the encoded dimensions have no individual real-world interpretation to attribute importance to, so this is not applied to the Autoencoder-Encoded results.

## Evaluation Metrics

Every model is scored through a single shared `evaluate()` function (`src/evaluation/metrics.py`), so no model receives a bespoke evaluation path:

- **RMSE, MAE, R²** — computed on returns (the modelling target)
- **Directional Accuracy / Hit Rate** — the percentage of test-set days where the predicted return's sign matched the actual return's sign; both names refer to the same computed quantity
- **Price-space RMSE/MAE/MAPE** — computed after reconstructing price from the predicted return, for interpretability alongside the return-space metrics

Magnitude accuracy (RMSE/MAE/R²) and directional accuracy are reported together deliberately: a model can have a smaller average error while still getting the direction wrong more often than not, which matters if a prediction is meant to inform a real decision rather than only minimise squared error.

## Key Results

The best-performing model and configuration per asset/horizon, on the held-out test set, from `results/evaluation/tables/decision_summary.csv`:

| Asset | Horizon | Best model | Feature configuration | Representation | RMSE (returns) | Hit Rate (%) |
|---|---|---|---|---|---|---|
| Bitcoin | 1 day | ARIMA | N/A (univariate) | Original | 0.0245 | 48.9 |
| Bitcoin | 7 days | LSTM | Market + Google Trends | Autoencoder-Encoded | 0.0576 | 58.1 |
| Bitcoin | 14 days | Random Forest | Market + Google Trends | Original | 0.0835 | 52.7 |
| Ethereum | 1 day | Random Forest | Market Only | Autoencoder-Encoded | 0.0365 | 48.9 |
| Ethereum | 7 days | Random Forest | Market + Google Trends + Fear & Greed Index | Autoencoder-Encoded | 0.0842 | 54.0 |
| Ethereum | 14 days | LSTM | Market + Google Trends | Autoencoder-Encoded | 0.1190 | 50.4 |

No single model wins across every asset/horizon combination. R² is negative in every row above — every model performs worse on this test period than simply predicting the historical mean return, a genuine finding reported as-is rather than minimised. The full experiment grid (5 models x 2 assets x 3 horizons x 4 feature configurations x 2 representations, minus the univariate exclusions) is in `results/modeling/metrics/model_development_metrics.csv`, and Diebold-Mariano significance tests against both baselines are in `results/evaluation/tables/dm_test_results.csv`.

## Statistical Evidence

Correlation and Granger causality analysis (`05_correlation_and_granger.ipynb`) tested whether Google Trends and the Fear & Greed Index show a statistically significant relationship with price, using FDR-corrected significance thresholds. Neither variable Granger-caused either asset's price at any tested lag. Google Trends showed a weak but significant correlation with price for at least one asset; Fear & Greed showed no significant correlation with either asset. These are association and predictive-value findings, not evidence of causation, and are described using that language throughout — Granger causality is a specific statistical test for whether past values of one series carry incremental predictive information about another under a given lag structure, not proof of a real causal mechanism.

## Explainability / Feature Importance

Permutation feature importance (`09_feature_importance_analysis.ipynb`, `src/feature_importance/`) is computed for Random Forest and XGBoost, on the Original representation, across all 48 real (asset, horizon, feature-configuration, model) combinations that have a saved model for this representation. It works by shuffling one feature at a time in the real test set and measuring how much the already-fitted model's RMSE degrades — it never retrains anything, and it never touches the model's internal structure.

Cross-model agreement between Random Forest and XGBoost is real but modest: mean Spearman rank correlation of **0.247** across the 24 (asset, horizon, feature-configuration) pairs where both models have a saved importance result (`results/feature_importance/processed/rank_agreement_summary.csv`). This is interpreted as the two models capturing similar aggregate signal without ranking individual features the same way, consistent with how differently the two algorithms are regularised — not as a discrepancy to explain away. Findings are cross-referenced directly against the correlation and Granger causality results (Notebook 05) rather than reported independently; no causal or economic interpretation is claimed for any feature's importance score.

## Repository Structure

```text
data/            Raw and processed data (frozen snapshot + generated feature sets)
notebooks/       01-09 + 06b, the research/dissertation workflow
src/             Reusable implementation the notebooks import from
models/trained/  Deployment-ready fitted models + manifest.json
results/         Generated tables, figures, predictions, and metrics
deployment/      FastAPI backend + Streamlit frontend
tests/           pytest suite
```

## Reproducibility

### 1. Clone / obtain the repository

```bash
git clone <repository-url>
cd AI_Crypto_Forecasting
```

### 2. Create environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Frozen data snapshot

`data/raw/` and `data/processed/merged_dataset.csv` are already included. **Do not re-run `01_data_collection.ipynb` for exact dissertation reproduction** — it would fetch current, different Google Trends data. Start from Notebook 02 onward if reproducing the reported results exactly; only re-run Notebook 01 if deliberately collecting a new dataset.

### 5. Reproduce preprocessing / features

```text
02_data_understanding.ipynb -> 03_data_preparation.ipynb -> 04_eda.ipynb
  -> 05_correlation_and_granger.ipynb -> 06_feature_engineering.ipynb -> 06b_autoencoder.ipynb
```

### 6. Reproduce model development

```text
07_model_development.ipynb
```
or, via the modular route:
```bash
python -m src.pipeline train
```
This is the computationally expensive stage (multiple assets, horizons, feature configurations, and LSTM hyperparameter grids); the repository retains the full research configuration rather than a reduced version.

### 7. Reproduce evaluation

```text
08_evaluation.ipynb
```
or:
```bash
python -m src.pipeline evaluate
```
Evaluation only loads already-saved predictions and metrics; it does not retrain anything.

### 8. Reproduce statistical / feature-importance analysis

```text
09_feature_importance_analysis.ipynb
```
or:
```bash
python run_feature_importance_grid.py
```

### 9. Run tests

```bash
python -m pytest tests/ -q
```

## Saved Artefacts

| Location | Contents |
|---|---|
| `models/trained/` | Deployment-ready fitted models (Random Forest, XGBoost, LSTM) used for Original-representation inference, plus `manifest.json`. This is what `src/evaluation/loaders.py` reads from. |
| `results/modeling/fitted_models/` | Notebook 07 experiment-local fitted artefacts retained for research traceability. The relevant Autoencoder-Encoded fitted models in this directory are also required by encoded runtime inference and are included in the backend Docker image; not every file here is necessarily a deployment artefact. |
| `results/autoencoder/` | Encoded feature CSVs and autoencoder model artefacts |
| `results/modeling/predictions/`, `results/modeling/metrics/` | Every model's predictions and metrics, all splits |
| `results/evaluation/` | Decision summary, master comparison, Diebold-Mariano results |
| `results/feature_importance/` | Permutation importance raw and processed results |
| `results/causality_analysis/` | Correlation and Granger causality tables and figures |

## Testing

```bash
python -m pytest tests/ -q
```

`pytest.ini` already sets the repository root on the import path, so no `PYTHONPATH=.` prefix is required. As audited, the suite reports **130 passed, 0 failed** on this repository state. Tests cover metrics, data preparation, feature engineering (including a regression test for a previously-identified target-alignment bug), the autoencoder, saved model loading, the unified `src.pipeline` entry point, and both deployment surfaces (the FastAPI app and CSV inference). Tests that depend on generated artefacts (trained models, saved predictions) skip cleanly rather than failing when those artefacts are not yet present; this is not a claim of 100% coverage.

## Local Application

Terminal 1 — backend:
```bash
source .venv/bin/activate
uvicorn deployment.app:app --reload
```

Terminal 2 — frontend:
```bash
source .venv/bin/activate
streamlit run deployment/frontend/Home.py
```

Open `http://localhost:8501` (Streamlit) and `http://127.0.0.1:8000/docs` (FastAPI). The frontend locates the backend via the `CRYPTO_API_URL` environment variable, defaulting to `http://127.0.0.1:8000` for local development.

## Docker

Backend (verified against `deployment/Dockerfile`):
```bash
docker build -f deployment/Dockerfile -t crypto-forecast-api .
docker run -p 8080:8080 crypto-forecast-api
```

Frontend (verified against `deployment/Dockerfile.streamlit`): the container listens on port 8080 and requires a reachable FastAPI backend URL supplied through the `CRYPTO_API_URL` environment variable. `http://localhost:8080` is only correct if the backend is reachable from the frontend container under that address (e.g. `--network host`, or an equivalent Docker network setup) — inside a standard Docker network, `localhost` refers to the frontend container itself, not the backend.
```bash
docker build -f deployment/Dockerfile.streamlit -t crypto-forecast-frontend .
docker run -p 8501:8080 -e CRYPTO_API_URL=<reachable backend URL> crypto-forecast-frontend
```

**`docker-compose.yml`**: corrected and verified end-to-end. `docker compose up --build` starts both containers together, with the backend healthcheck (`http://localhost:8080/health`, checked from inside the backend container) confirmed returning HTTP 200 before the frontend starts. Once running:

- Frontend: `http://localhost:8501`
- FastAPI documentation: `http://localhost:8000/docs`

Internally, the frontend reaches the backend over the Docker Compose network at `http://api:8080`.

## Deployment Architecture

```text
Browser
  |
  v
Streamlit frontend  (deployment/frontend/, talks to the API over HTTP only, never imports src/ directly)
  | HTTP
  v
FastAPI inference API  (deployment/app.py, imports src/ directly)
  |
  v
Runtime model artefacts
  |- models/trained/                       (Original representation)
  `- results/modeling/fitted_models/       (Autoencoder-Encoded representation)
  |
  v
Saved evaluation/research artefacts (results/)
```

The backend is inference-only — it loads already-saved models and results and never trains anything at runtime. If Google Cloud Run is used for the live deployment, no project IDs, service URLs, or credentials are included in this repository; deployment configuration is handled outside version control.

## API

Confirmed directly against `deployment/app.py`:

```text
GET  /                              service info / valid options
GET  /health                        health check
POST /predict                       structured JSON inference
POST /predict/csv                   raw CSV upload inference
GET  /csv-requirements               required/optional columns for CSV upload
GET  /models                        trained model manifest
GET  /decision-summary               best model per asset/horizon
GET  /research-evidence              correlation/Granger causality evidence
GET  /feature-importance-comparison  cross-model permutation importance
GET  /feature-importance-agreement-summary
GET  /feature-group-importance
GET  /feature-importance-for-model
GET  /arima-info
GET  /naive-info
GET  /last-training-date
GET  /full-metrics                   full saved metrics table
GET  /predictions                    full saved predictions table
```

`/predict` (structured JSON inference, loading a saved fitted model) supports Random Forest, XGBoost, and LSTM only; ARIMA and Naive are rejected here, since neither has a single reusable saved fitted model to load. `/predict/csv` additionally supports ARIMA: it fits a fresh ARIMA model directly on the uploaded historical price series, using the order already selected during model development, rather than loading a saved object. Naive is not exposed through either endpoint, since it has no trainable parameters and no saved model to load for this workflow.

## Live Demonstration

The frontend is deployed on Google Cloud Run.

**Live demo:** https://ai-crypto-frontend-448332130847.europe-west2.run.app

## Limitations

- The evaluation covers a single historical period, two assets, and the feature engineering already fixed in Notebooks 06/06b/07; it does not establish how these models would perform in a different market regime.
- Cryptocurrency return series are close to non-stationary/near-random-walk in practice, which limits how much signal any model — including the more complex ones — can realistically extract; the negative R² values throughout the results reflect this directly.
- Google Trends search interest is independently normalised per collection window; Bitcoin and Ethereum search-interest levels are not directly comparable absolute quantities, and the external-feature experiments should be read as within-series forecasting tests rather than direct measures of relative public interest.
- LSTM training involves stochastic elements (weight initialisation, and TensorFlow's own execution nondeterminism on some platforms); exact numerical reproduction of LSTM results down to the last decimal place is not guaranteed even with fixed seeds.
- XGBoost is trained with `n_jobs=4`; its multi-threaded histogram construction can, in principle, produce very small non-bit-identical results across machines with different core counts, even with a fixed random seed.
- Correlation and Granger causality findings indicate association and predictive value, not causation; this distinction is maintained throughout rather than implied away.

## Future Work

- Extending the Granger causality analysis to test cross-asset relationships between Bitcoin and Ethereum returns directly, which the current analysis does not cover.
- Evaluating the autoencoder representation's forecasting value against a like-for-like feature-count reduction of the original features, to separate "compression" from "dimensionality reduction" as explanations for its mixed results.
- Testing whether the AIC-preferred lag structure (as opposed to the BIC-preferred lag currently used) changes which external variables would be retained in feature selection.

## Technology Stack

Python 3.12, pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels, FastAPI, Streamlit, pytest, Docker.

## Academic Context

This repository accompanies an MSc Data Science dissertation submitted at the University of Surrey.

