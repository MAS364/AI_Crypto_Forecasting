# AI Cryptocurrency Time-Series Analysis: Correlation, Causality, and Predictive Modelling for Bitcoin and Ethereum

MSc Data Science dissertation, University of Surrey. The project forecasts Bitcoin and Ethereum returns using market data together with Google Trends search interest and the Crypto Fear & Greed Index, comparing statistical, machine learning, and deep learning models across 1-, 7-, and 14-day forecast horizons, with permutation feature importance used to interpret the trained models.

## Project Overview

Cryptocurrency prices are volatile and only partially explained by their own history. This project evaluates whether adding two external, publicly available signals — Google Trends search interest and the Crypto Fear & Greed Index — improves short-horizon forecasting of Bitcoin and Ethereum returns, and whether machine learning and deep learning models outperform classical statistical baselines when doing so. Five model families are compared under identical evaluation conditions: a naive persistence baseline, ARIMA, Random Forest, XGBoost, and an LSTM. An autoencoder-compressed feature representation is evaluated alongside the original engineered features as a secondary comparison. Permutation feature importance is used to interpret which engineered features the tree-based models actually rely on, cross-referenced against the project's own correlation and Granger causality analysis rather than treated as an independent, unrelated result.

## Research Questions

1. Can machine learning and deep learning models outperform classical statistical models in forecasting Bitcoin and Ethereum prices?

2. Does incorporating selected external signals (Google Trends and the Crypto Fear & Greed Index) improve forecasting performance compared with using historical cryptocurrency market data alone?

3. Which forecasting model provides the most accurate and reliable predictions of Bitcoin and Ethereum prices?

4. What relationships exist between cryptocurrency prices and selected external signals, specifically Google Trends and the Crypto Fear & Greed Index?

5. Do Google Trends and the Crypto Fear & Greed Index exhibit predictive relationships with cryptocurrency prices according to Granger causality analysis?

## Key Contributions

- A controlled comparison of five model families (Naive, ARIMA, Random Forest, XGBoost, LSTM) across two assets, three forecast horizons, and four feature configurations, evaluated identically on chronologically held-out test data.
- Evidence-informed feature engineering: the stationarity, correlation and Granger analysis in Notebook 05 informed transformations and lag selection. All four feature configurations were retained and evaluated, regardless of statistical significance.
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

- `btc-usd-max.csv`, `eth-usd-max.csv` — Bitcoin and Ethereum historical market data downloaded manually from the CoinGecko website
- `google_trends.csv` (plus windowed intermediate files) — search interest, collected via the `pytrends` library, confirmed directly in `notebooks/01_data_collection.ipynb`
- `fear_greed.csv` — the Crypto Fear & Greed Index

**A note on precision here rather than assumption**: 
Bitcoin and Ethereum historical market data were downloaded manually as CSV files from the CoinGecko website because the required programmatic API access was not available under the free plan. The frozen files used in this study are included in `data/raw/`. Although `pycoingecko` remains in `requirements.txt`, it was not used to obtain the final dissertation dataset.

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
- **Autoencoder-Encoded** — those same features compressed into 8 unnamed latent dimensions (`encoded_0`..`encoded_7`) by a trained autoencoder (`06b_autoencoder.ipynb`), evaluated only for Random Forest, XGBoost, and LSTM. Naive and ARIMA were never run through the autoencoder branch, since both are univariate and have no feature matrix to encode.

Permutation feature importance is computed for **both** representations (Random Forest and XGBoost only — see [Explainability / Feature Importance](#explainability--feature-importance) below for the full breakdown, including how latent-dimension importance is interpreted and cross-referenced against the original features it was compressed from).

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

No single model wins across every asset/horizon combination. R² is negative in every row above, indicating performance below a constant-mean benchmark on the evaluated test sample. This benchmark is different from the naïve persistence comparator. The full experiment grid (5 models x 2 assets x 3 horizons x 4 feature configurations x 2 representations, minus the univariate exclusions) is in `results/modeling/metrics/model_development_metrics.csv`, and Diebold-Mariano significance tests against both baselines are in `results/evaluation/tables/dm_test_results.csv`.

Among the 72 Original-representation ML/DL comparisons against the naïve benchmark, 64 were significant at the unadjusted 5% level. Including the six unique ARIMA comparisons gives 70 significant results among 78 unique fitted-model comparisons. Against ARIMA, 13 of 72 ML/DL comparisons were significant: 12 favoured ARIMA and one favoured XGBoost. The DM-test p-values were not adjusted for multiple comparisons; Benjamini–Hochberg correction was applied to the correlation and Granger-test families, not to the DM tests.

## Statistical Evidence

A preliminary, raw-level look at these relationships appears first during exploratory analysis (`04_eda.ipynb`): a static correlation heatmap and scatter plots computed directly on price levels (e.g. Bitcoin price vs. Fear & Greed Index, r ≈ 0.29). These are exploratory only — prices are non-stationary, so a raw-level correlation coefficient can be inflated by shared trend rather than a genuine relationship, and Notebook 04 says so directly at each such plot.

The formal test is `05_correlation_and_granger.ipynb`, and it is run on transformed, stationarity-confirmed series, not raw prices: Bitcoin and Ethereum prices are converted to **log returns**, and every other variable (volume, Google Trends search interest, Fear & Greed Index) is converted to a **first difference** — `results/causality_analysis/tables/correlation_transformations.csv` records exactly which transformation was applied to which variable, and both ADF and KPSS tests confirm the transformed series are stationary (see [Limitations](#limitations) for how this is worded precisely). Pearson and Spearman correlation, and Granger causality in the direction external factor → cryptocurrency return, are then tested with Benjamini–Hochberg FDR correction applied to the full set of p-values in each stage (`results/causality_analysis/tables/correlation_pairwise_results.csv`, `granger_causality_results.csv`). Neither external signal showed a statistically significant Granger-predictive relationship with returns in the primary lag-2 tests or the lag-5 robustness tests. Google Trends showed a very weak but statistically significant correlation with returns for at least one asset after FDR correction; Fear & Greed showed no significant correlation with either asset's returns. These are association and predictive-value findings, not evidence of causation, and are described using that language throughout — Granger causality is a specific statistical test for whether past values of one series carry incremental predictive information about another under a given lag structure, not proof of a real causal mechanism. Feature importance (below) is a separate, later-stage finding about what the trained forecasting models actually relied on, and is deliberately cross-referenced against this section rather than treated as independent corroboration of causality.

## Explainability / Feature Importance

SHAP was the original choice for this project, but `TreeExplainer` caused repeated, reproducible kernel crashes in the available environment (documented directly in `src/feature_importance/permutation.py`). Permutation Feature Importance (PFI) was adopted instead: it is model-agnostic (it only ever calls `.predict()` on shuffled copies of the real, already-held-out test set — it never inspects a model's internal structure and never retrains anything), which avoided the instability entirely. The trade-off is standard for PFI generally, not specific to this project: it measures predictive reliance, not a causal effect, and its attribution can be distorted when predictors are strongly correlated with each other, since the model can partly recover a shuffled feature's information from a correlated one.

**Original representation.** Computed for Random Forest and XGBoost (`09_feature_importance_analysis.ipynb`, `src/feature_importance/`, `run_feature_importance_grid.py`) across all 48 real (asset, horizon, feature-configuration, model) combinations that have a saved model for this representation. Cross-model agreement is real but modest: Mean RF–XGBoost Spearman rank agreement was approximately 0.247 across 23 valid Original-representation comparisons. One of the 24 configurations produced an undefined correlation and was excluded from the valid mean count (`results/feature_importance/processed/rank_agreement_summary.csv`). Findings are cross-referenced directly against the correlation and Granger causality results (Notebook 05) rather than reported independently; no causal or economic interpretation is claimed for any feature's importance score.

**Autoencoder-Encoded representation.** The same technique, same 48-combination grid shape, run separately against Random Forest and XGBoost trained on the 8 latent `encoded_0`..`encoded_7` dimensions instead of the named engineered features (`notebooks/10_feature_importance_autoencoder_analysis.ipynb`, `run_feature_importance_grid_autoencoder.py`, saved to the separate `results/feature_importance_autoencoder/` tree so nothing under `results/feature_importance/` is touched). A latent dimension's importance score describes how much the model's accuracy depends on that specific compressed dimension — it says nothing directly about any one original real-world variable, since a dimension is a learned mixture of several original signals, not a stand-in for one of them. Mean RF–XGBoost Spearman rank agreement was approximately **0.604 across 24 valid Autoencoder-Encoded comparisons** (`rank_agreement_summary_autoencoder.csv`). This indicates greater agreement between the two models' encoded-dimension rankings than in the Original representation; it does not demonstrate that the encoded representation is more interpretable or produces better forecasts.

LSTM permutation importance was not implemented because its windowed sequence input requires a separate sequence-aware permutation design specifying whether features, timesteps or both are permuted. The submitted permutation-importance analysis therefore covers Random Forest and XGBoost only.

**Latent-to-original correlation.** To make the encoded dimensions more interpretable, Pearson and Spearman correlations were calculated between each of the eight latent dimensions and every original engineered feature. This supports an *approximate* reading of what a given latent dimension leans on most. It is descriptive only: a latent dimension correlating with one or more original features is not proof that it "represents" that variable specifically — an 8-dimensional bottleneck is trained to reconstruct the whole feature vector at once, so moderate correlation with several original features simultaneously is the expected pattern, not a one-to-one mapping — and no causal claim is made in either direction.

**Original vs. Autoencoder-Encoded comparison.** Raw permutation-importance magnitudes are not directly comparable across the two branches (a ~12–17-feature configuration vs. an 8-dimension one), so `run_combined_feature_importance_report.py` produces two clearly separate things rather than one blended number: a plain long-format union of both branches' raw results for anyone who wants to filter the numbers themselves (`all_representations_long.csv`), and a genuinely comparable summary (`representation_comparison_summary.csv`) built from test RMSE (the one figure that is on the same scale regardless of representation) plus within-configuration normalised importance. Across the 48 shared configurations, Autoencoder-Encoded gave the lower test RMSE in 29 and Original in 19 — a real but not one-sided result, consistent with the mixed picture already reported in [Key Results](#key-results) above.

Notebook 10 itself covers the permutation-importance grid, the full original+encoded feature dictionary, the RF-vs-XGBoost ranking comparison, a plain combined ranking table, the Feature Importance Comparison tab diagnosis (below), and the RF-vs-XGBoost disagreement figures. The latent-to-original correlation analysis and the defensible RMSE-based representation comparison are currently only produced by the standalone scripts (or `python -m src.pipeline feature-importance-autoencoder`, which runs all four in sequence) — see [Reproducibility](#reproducibility) below for the exact commands. All outputs, from either route, are written under `results/feature_importance_autoencoder/`.

## Repository Structure

```text
data/            Raw and processed data (frozen snapshot + generated feature sets)
notebooks/       01-10 + 06b, the research/dissertation workflow
src/             Reusable implementation the notebooks import from
models/trained/  Deployment-ready fitted models + manifest.json
results/         Generated tables, figures, predictions, and metrics
deployment/      FastAPI backend + Streamlit frontend
tests/           pytest suite
```

## Reproducibility

### 1. Extract the submitted project

Extract the submitted ZIP file, then open a terminal in the extracted project directory before continuing.

### 2. Create the Python 3.12 environment

Confirm that Python 3.12 is installed:

```bash
python3.12 --version
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Use the frozen data snapshot

The required files in `data/raw/` and `data/processed/merged_dataset.csv` are included. Do not rerun `01_data_collection.ipynb` when reproducing the dissertation results because it would retrieve current Google Trends data. Rerun Notebook 01 only when deliberately creating a new dataset.

### 4. Reproduce the complete workflow

The submitted results, predictions and fitted artefacts can be inspected without retraining. Full reproduction requires running the notebooks in this order:

```text
02_data_understanding.ipynb
→ 03_data_preparation.ipynb
→ 04_eda.ipynb
→ 05_correlation_and_granger.ipynb
→ 06_feature_engineering.ipynb
→ 06b_autoencoder.ipynb
→ 07_model_development.ipynb
→ 08_evaluation.ipynb
→ 09_feature_importance_analysis.ipynb
→ 10_feature_importance_autoencoder_analysis.ipynb
```

Notebook 07 retrains the complete model grid and may take considerable time. Notebook 08 loads the saved predictions and metrics and does not retrain models.

The verified modular commands for model development and evaluation are:

```bash
python -m src.pipeline train
python -m src.pipeline evaluate
```

### 5. Reproduce feature-importance analysis

Original representation:

```bash
python run_feature_importance_grid.py
```

Autoencoder-Encoded permutation importance, latent-to-original correlations and the cross-representation comparison:

```bash
python -m src.pipeline feature-importance-autoencoder
```

This command runs the following scripts in order:

```bash
python run_feature_importance_grid_autoencoder.py
python run_latent_original_correlation_analysis.py
python run_combined_feature_importance_report.py
python run_feature_importance_figures_autoencoder.py
```

These commands load existing fitted models and saved data; they do not retrain the model grid.

### 6. Run the tests

```bash
python -m pytest tests/ -q
```

The tests cover preprocessing, feature engineering, evaluation, saved-model loading, deployment and feature-importance analysis. The exact result may depend on the installed environment and the availability of generated artefacts.

## Saved Artefacts

| Location | Contents |
|---|---|
| `models/trained/` | Deployment-ready fitted models (Random Forest, XGBoost, LSTM) used for Original-representation inference, plus `manifest.json`. This is what `src/evaluation/loaders.py` reads from. |
| `results/modeling/fitted_models/` | Notebook 07 experiment-local fitted artefacts retained for research traceability. The relevant Autoencoder-Encoded fitted models in this directory are also required by encoded runtime inference and are included in the backend Docker image; not every file here is necessarily a deployment artefact. |
| `results/autoencoder/` | Encoded feature CSVs and autoencoder model artefacts |
| `results/modeling/predictions/`, `results/modeling/metrics/` | Every model's predictions and metrics, all splits |
| `results/evaluation/` | Decision summary, master comparison, Diebold-Mariano results |
| `results/feature_importance/` | Permutation importance raw and processed results (Original representation) |
| `results/feature_importance_autoencoder/` | Permutation importance for the Autoencoder-Encoded representation, the latent-to-original correlation analysis, the cross-representation comparison, and their figures |
| `results/causality_analysis/` | Correlation and Granger causality tables and figures |

## Testing

```bash
python -m pytest tests/ -q
```

`pytest.ini` places the repository root on the import path, so no `PYTHONPATH=.` prefix is required. The suite covers metrics, preprocessing, feature engineering, model loading, deployment and Autoencoder-Encoded feature-importance analysis. Run the command above in the submitted environment to obtain the current result; tests that require unavailable generated artefacts may skip cleanly.

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

The simplest way to start the backend and frontend together is:

```bash
docker compose up --build
```

The Compose configuration waits for the backend health check before starting the frontend. Once both services are running:

- Streamlit frontend: `http://localhost:8501`
- FastAPI documentation: `http://localhost:8000/docs`

Within the Compose network, the frontend reaches the backend at `http://api:8080`.

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
GET  /feature-importance-comparison  cross-model permutation importance (Original)
GET  /feature-importance-agreement-summary
GET  /feature-group-importance
GET  /feature-importance-for-model
GET  /feature-importance-comparison-autoencoder   cross-model permutation importance (Autoencoder-Encoded)
GET  /feature-importance-agreement-summary-autoencoder
GET  /feature-group-importance-autoencoder
GET  /latent-original-correlations                latent-vs-original-feature correlation (optional asset/horizon/feature_set filters)
GET  /latent-original-top-associations
GET  /representation-comparison-summary            Original vs Autoencoder-Encoded, test RMSE + normalised importance
GET  /arima-info
GET  /naive-info
GET  /last-training-date
GET  /full-metrics                   full saved metrics table
GET  /predictions                    full saved predictions table
```

`/predict` (structured JSON inference, loading a saved fitted model) supports Random Forest, XGBoost, and LSTM only; ARIMA and Naive are rejected here, since neither has a single reusable saved fitted model to load. `/predict/csv` additionally supports ARIMA: it fits a fresh ARIMA model directly on the uploaded historical price series, using the order already selected during model development, rather than loading a saved object. Naive is not exposed through either endpoint, since it has no trainable parameters and no saved model to load for this workflow.

### CSV upload inference behaviour

Confirmed directly against `deployment/csv_inference.py`:

- **No live external API calls.** The deployed backend's own dependency list (`deployment/requirements.txt`) deliberately excludes `pytrends`, `yfinance`, and `pycoingecko` — a prediction, whether from `/predict` or `/predict/csv`, is produced entirely from already-fitted models and the data supplied in the request; it never fetches current Google Trends data or any other live source. External-API rate limits are consequently not a factor for either endpoint.
- **Validation is itemised, not first-error-only.** A `CSVValidationError` collects every problem found — missing required columns, unparseable or missing dates, duplicate dates, dates not in ascending order, non-numeric or out-of-range values (e.g. Search Interest / Fear & Greed Index outside 0–100, negative price/volume/market cap), an unrecognised `Market_Sentiment` category, or fewer rows than the selected model needs — and returns them together as a single HTTP 422 response, so a user fixing their upload doesn't have to re-submit once per error.
- **No imputation.** Missing or invalid values are reported as validation errors; they are never filled in, interpolated, or silently dropped.
- **Minimum history**: 30 rows for Random Forest, XGBoost, and LSTM (the longest lookback used by feature engineering is a 30-day rolling window); 11 rows for ARIMA, which only needs a usable return series. An LSTM-specific check additionally confirms the exact sequence window it needs is free of missing values, and explains that a gap is often a deliberately-preserved data anomaly rather than a malformed upload — suggesting Random Forest or XGBoost as an alternative for that same upload, since both tolerate missing values internally where LSTM cannot.
- **Unsupported selections are reported clearly, not silently substituted**: Naive is rejected with an explicit message (no trainable parameters, no fitted model to load); an unrecognised feature-configuration override is rejected with the specific missing column(s) it needs.

## Live Demonstration

The frontend is deployed on Google Cloud Run.

**Live demo:** https://ai-crypto-frontend-448332130847.europe-west2.run.app

## Limitations

- The evaluation covers a single historical period, two assets, and the feature engineering already fixed in Notebooks 06/06b/07; it does not establish how these models would perform in a different market regime.
- The chronological split (70% train / 15% validation / 15% test, in date order) preserves temporal order and helps prevent leakage, but final evaluation still rests on one held-out historical period. Results may be sensitive to the specific market regime that period happened to represent, and the reported comparisons between models, feature configurations, and representations should not automatically be generalised to every future regime.
- The transformed series used for correlation, Granger causality, and modelling (log returns for prices, first differences for the other variables) were tested with ADF and KPSS and treated as stationary based on that evidence — see [Statistical Evidence](#statistical-evidence) above. Cryptocurrency returns nevertheless remain noisy and only weakly predictable in practice; the negative R² values throughout the results reflect that weak predictability, not a stationarity problem, and near-random-walk / efficient-market-like behaviour here refers specifically to that weak predictability rather than to non-stationarity.
- Google Trends search interest is independently normalised per collection window; Bitcoin and Ethereum search-interest levels are not directly comparable absolute quantities, and the external-feature experiments should be read as within-series forecasting tests rather than direct measures of relative public interest.
- LSTM training involves stochastic elements (weight initialisation, and TensorFlow's own execution nondeterminism on some platforms); exact numerical reproduction of LSTM results down to the last decimal place is not guaranteed even with fixed seeds.
- XGBoost is trained with `n_jobs=4`; its multi-threaded histogram construction can, in principle, produce very small non-bit-identical results across machines with different core counts, even with a fixed random seed.
- Correlation and Granger causality findings indicate association and predictive value, not causation; this distinction is maintained throughout rather than implied away.
- Permutation feature importance measures predictive reliance on a feature, not a causal effect, and its attribution can be distorted when predictors are correlated with each other; see [Explainability / Feature Importance](#explainability--feature-importance) above.

## Future Work

- Expanding-window or walk-forward validation, evaluated across multiple market regimes rather than the single held-out period used here, directly addressing the chronological-split limitation above.
- Examining the stability of model rankings and of external-signal usefulness (Google Trends, Fear & Greed) across those regimes, rather than assuming the single-period ranking reported here would hold generally. This future work has not been carried out in this repository — the single chronological split above is what the current results are based on.
- Extending the Granger causality analysis to test cross-asset relationships between Bitcoin and Ethereum returns directly, which the current analysis does not cover.
- Evaluating the autoencoder representation's forecasting value against a like-for-like feature-count reduction of the original features, to separate "compression" from "dimensionality reduction" as explanations for its mixed results.
- Testing whether the AIC-preferred lag structure (as opposed to the BIC-preferred lag currently used) changes which external variables would be retained in feature selection.
- Extending `notebooks/10_feature_importance_autoencoder_analysis.ipynb` itself to include the latent-to-original correlation analysis and the RMSE-based representation comparison, so the full autoencoder feature-importance workflow runs through one notebook rather than the notebook plus four separate scripts.

## Technology Stack

Python 3.12, pandas, NumPy, scikit-learn, XGBoost, TensorFlow/Keras, statsmodels, FastAPI, Streamlit, pytest, Docker.

## Academic Context

This repository accompanies the MSc Data Science dissertation *AI Cryptocurrency Time-Series Analysis: Correlation, Causality, and Predictive Modelling for Bitcoin and Ethereum*, submitted at the University of Surrey.

## Disclaimer

This project was developed for academic research. Its forecasts do not constitute financial or investment advice.
