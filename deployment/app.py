"""Production inference API -- wraps src.pipeline.run_predict.

Loads saved models only. Never trains, tunes, or refits anything --
matches the same discipline as run_predict() and everything it depends on
(src/evaluation/loaders.py, src/models/autoencoder.py's predict_encode()).

Run locally:
    uvicorn deployment.app:app --reload

Then:
    curl -X POST http://127.0.0.1:8000/predict \\
        -H "Content-Type: application/json" \\
        -d @example_request.json

See GET /health for a quick check that the app started correctly, and
GET / for the available model combinations before predicting.
"""

import sys
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_predict, FEATURE_SET_PARAMS
from src.models.pipeline import ASSETS, FORECAST_HORIZONS
from src import config
from src.evaluation import loaders

FEATURE_IMPORTANCE_DIR = config.RESULTS_DIR / "feature_importance"
from csv_inference import run_csv_inference, CSVValidationError, MIN_ROWS_FOR_FEATURES

app = FastAPI(
    title="Crypto Forecasting Inference API",
    description="Loads saved Random Forest / XGBoost / LSTM models and predicts. Never retrains.",
    version="1.1.0",
)


@app.on_event("startup")
def _warm_up_tensorflow():
    """Importing TensorFlow for the first time costs 10-30+ seconds --
    unrelated to model size or data, just the import itself. Without
    this, that cost lands on whichever live request happens to be the
    first to touch LSTM or Autoencoder-Encoded (see src.pipeline's
    predict_encode() and LSTM branch), which is a bad time to discover a
    slow request -- e.g. mid-demo. Paying it once here, at boot, means
    every real request stays fast. Failures are logged, not raised --
    a slow first request is a much smaller problem than an API that
    won't start because of a warm-up hiccup.

    Also does one real model load (Bitcoin, horizon 1, Market Only LSTM
    -- a path directly verified to exist on disk) since loading a saved
    Keras model and building its inference graph has its own overhead
    beyond just the import, and only a real load exercises that."""
    import time
    import logging
    logger = logging.getLogger("uvicorn.error")
    try:
        t0 = time.time()
        from tensorflow import keras
        logger.info(f"TensorFlow import complete in {time.time() - t0:.1f}s")

        t1 = time.time()
        warm_model_path = config.MODELS_DIR / "trained" / "horizon_1day" / "bitcoin" / "market_only" / "lstm.keras"
        if warm_model_path.exists():
            keras.models.load_model(warm_model_path)
            logger.info(f"Model warm-up load complete in {time.time() - t1:.1f}s")
    except Exception as e:
        logger.warning(f"TensorFlow warm-up failed (non-fatal, will retry on first real request): {e}")


class PredictRequest(BaseModel):
    horizon: int = Field(..., description="Forecast horizon in days.", examples=[1])
    asset: str = Field(..., description="Bitcoin or Ethereum.", examples=["Bitcoin"])
    feature_set: str = Field(..., description="One of: primary, candidate, ablation, combined.", examples=["primary"])
    representation: str = Field(..., description="Original or Autoencoder-Encoded.", examples=["Original"])
    model: str = Field(..., description="Random Forest, XGBoost, or LSTM. (Naive/ARIMA have no saved model to load.)", examples=["Random Forest"])
    rows: list[dict] = Field(..., description="Feature rows -- same columns Notebook 07 trained on, including the price-level column and Date.")


class PredictResponse(BaseModel):
    horizon: int
    asset: str
    model: str
    feature_set: str
    representation: str
    input_price_level: float
    predicted_return: float
    predicted_price: float


@app.get("/")
def root():
    return {
        "message": "Crypto Forecasting Inference API. POST /predict to run inference. See /health for status.",
        "valid_assets": list(ASSETS.keys()),
        "valid_horizons": FORECAST_HORIZONS,
        "valid_feature_sets": list(FEATURE_SET_PARAMS.keys()),
        "valid_representations": ["Original", "Autoencoder-Encoded"],
        "valid_models": ["Random Forest", "XGBoost", "LSTM"],
        "note": "This structured-JSON /predict endpoint takes pre-engineered feature rows, so Naive and ARIMA aren't supported here. ARIMA IS supported on /predict/csv (raw price history) and /csv-requirements has details. Naive has no fitted object anywhere.",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.asset not in ASSETS:
        raise HTTPException(400, f"Unknown asset '{request.asset}'. Valid: {list(ASSETS.keys())}")
    if request.horizon not in FORECAST_HORIZONS:
        raise HTTPException(400, f"Unknown horizon '{request.horizon}'. Valid: {FORECAST_HORIZONS}")
    if request.feature_set not in FEATURE_SET_PARAMS:
        raise HTTPException(400, f"Unknown feature_set '{request.feature_set}'. Valid: {list(FEATURE_SET_PARAMS.keys())}")
    if request.model in ("Naive", "ARIMA"):
        raise HTTPException(400, f"{request.model} has no saved fitted model to load for inference. Use Random Forest, XGBoost, or LSTM.")

    try:
        input_df = pd.DataFrame(request.rows)
        if "Date" in input_df.columns:
            input_df["Date"] = pd.to_datetime(input_df["Date"])
    except Exception as e:
        raise HTTPException(400, f"Could not parse 'rows' into a dataframe: {e}")

    try:
        result = run_predict(
            horizon=request.horizon, asset=request.asset, feature_set_key=request.feature_set,
            model_name=request.model, representation=request.representation, input_df=input_df,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, f"No saved model found for this combination: {e}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    return result


@app.post("/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    asset: str = Form(...),
    model: str = Form(...),
    horizon: int = Form(...),
    representation: str = Form("Original"),
    feature_set: Optional[str] = Form(None),
):
    """CSV upload inference -- CSVLoader -> Validator -> Preprocessor ->
    FeatureEngineer -> Predictor, via deployment.csv_inference. Never
    retrains. Returns a clear, itemised list of every validation problem
    if the upload doesn't pass, not just the first one found.

    feature_set: optional explicit override (primary/candidate/ablation/
    combined). If omitted, auto-detects the richest configuration the
    upload's columns support, same as before this parameter existed.
    Ignored for ARIMA (always "primary" -- see csv_inference.py)."""
    file_bytes = await file.read()
    try:
        result = run_csv_inference(
            file_bytes=file_bytes, asset=asset, model_name=model,
            horizon=horizon, representation=representation,
            requested_feature_set=feature_set,
        )
    except CSVValidationError as e:
        raise HTTPException(422, detail={"validation_errors": e.problems})
    except FileNotFoundError as e:
        raise HTTPException(404, f"No saved model found for this combination: {e}")
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    # Last line of defense: NaN/inf anywhere in the result crashes
    # FastAPI's JSON encoder with an unhelpful empty response (NaN isn't
    # valid JSON) -- run_predict() already validates for this before
    # returning, but this catches anything that slips through any other
    # path (e.g. future model types) with a real error instead.
    import math
    bad_fields = [k for k, v in result.items() if isinstance(v, float) and not math.isfinite(v)]
    if bad_fields:
        raise HTTPException(500, f"Prediction produced non-finite value(s) in: {bad_fields} -- check input data for missing values or extreme outliers.")

    return result


@app.get("/csv-requirements")
def csv_requirements(asset: str = "Bitcoin"):
    """What a CSV upload needs -- read by the Streamlit frontend to show
    the user the exact required/optional columns before they upload."""
    return {
        "required_columns": ["Date", f"{asset}_Price", f"{asset}_Market_Cap", f"{asset}_Volume"],
        "optional_columns": [f"{asset}_Search_Interest", "Fear_Greed_Index", "Market_Sentiment"],
        "min_rows": MIN_ROWS_FOR_FEATURES,
        "note": "Fear_Greed_Index and Market_Sentiment must be provided together, or not at all.",
        "supported_models": ["Random Forest", "XGBoost", "LSTM", "ARIMA"],
        "unsupported_models": {"Naive": "no trainable parameters, no fitted object of any kind"},
        "arima_note": (
            "ARIMA has no saved fitted model like the other three -- instead, it fits a "
            "fresh statistical model on your uploaded data using the order already "
            "selected during model development. One real fit per request, not model "
            "training or tuning."
        ),
    }



@app.get("/models")
def list_models():
    """Model registry -- reads models/trained/manifest.json, the real
    index of every saved model this project has. No hardcoded paths."""
    try:
        manifest = loaders.load_manifest()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"count": len(manifest), "models": manifest}


@app.get("/decision-summary")
def decision_summary():
    """The real, saved best-model-per-asset-per-horizon table from
    08_evaluation.ipynb -- not recomputed."""
    path = config.RESULTS_DIR / "evaluation" / "tables" / "decision_summary.csv"
    if not path.exists():
        raise HTTPException(404, f"{path} not found -- run 08_evaluation.ipynb first.")
    df = pd.read_csv(path)
    return df.to_dict("records")


@app.get("/research-evidence")
def research_evidence():
    """Real, saved correlation and Granger causality results from
    05_correlation_and_granger.ipynb -- not recomputed."""
    tables_dir = config.RESULTS_DIR / "causality_analysis" / "tables"
    corr_path = tables_dir / "correlation_pairwise_results.csv"
    granger_path = tables_dir / "granger_causality_results.csv"
    if not corr_path.exists() or not granger_path.exists():
        raise HTTPException(404, f"Correlation/Granger tables not found under {tables_dir} -- run 05_correlation_and_granger.ipynb first.")
    return {
        "correlation": pd.read_csv(corr_path).to_dict("records"),
        "granger_causality": pd.read_csv(granger_path).to_dict("records"),
    }


@app.get("/feature-importance-comparison")
def feature_importance_comparison():
    """The full Random Forest vs XGBoost permutation feature importance
    grid -- every (asset, horizon, feature configuration) this project
    has a trained model for, 48 real experiments, both models on every
    one. Replaces the earlier SHAP-based cross-model comparison, which
    was removed from this project (a real, reproducible kernel
    instability was traced to TreeExplainer in this environment;
    permutation importance is model-agnostic and only ever calls
    .predict(), which never showed that instability). Real, saved data
    from run_feature_importance_grid.py -- not computed live.
    """
    path = FEATURE_IMPORTANCE_DIR / "processed" / "cross_model_comparison.csv"
    if not path.exists():
        raise HTTPException(
            404,
            f"{path} not found -- run run_feature_importance_grid.py first to generate "
            "the permutation importance grid.",
        )
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


@app.get("/feature-importance-agreement-summary")
def feature_importance_agreement_summary():
    """Per-configuration Random Forest / XGBoost rank agreement --
    Spearman rank correlation and top-5/top-10 overlap across all 24
    real configurations."""
    path = FEATURE_IMPORTANCE_DIR / "processed" / "rank_agreement_summary.csv"
    if not path.exists():
        raise HTTPException(
            404,
            f"{path} not found -- run run_feature_importance_grid.py first to generate "
            "the permutation importance grid.",
        )
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


@app.get("/feature-group-importance")
def feature_group_importance():
    """Feature-group-level permutation importance (Price & Return,
    Volatility, Trading Volume, Google Trends, Fear & Greed / Sentiment)
    across every real configuration and both models."""
    path = FEATURE_IMPORTANCE_DIR / "processed" / "feature_group_importance.csv"
    if not path.exists():
        raise HTTPException(
            404,
            f"{path} not found -- run run_feature_importance_grid.py first to generate "
            "the permutation importance grid.",
        )
    df = pd.read_csv(path)
    return json.loads(df.to_json(orient="records"))


@app.get("/feature-importance-for-model")
def feature_importance_for_model(asset: str, horizon: int, model: str, feature_set: str = "primary"):
    """One (asset, horizon, feature_set, model) combination's real
    permutation importance -- used by the Forecast page's technical
    report for whichever model actually made this prediction. No
    representation parameter: the permutation importance grid only
    covers the Original (named, engineered-feature) representation, not
    Autoencoder-Encoded, since the point is engineered-feature
    interpretability. Returns an empty list, not an error, if this
    specific combination isn't in the grid (e.g. the deployed model for
    this asset/horizon uses the encoded representation) -- the caller
    already treats this as optional."""
    path = FEATURE_IMPORTANCE_DIR / "raw" / "permutation_importance_long.csv"
    if not path.exists():
        raise HTTPException(
            404,
            f"{path} not found -- run run_feature_importance_grid.py first.",
        )
    df = pd.read_csv(path)
    match = df[
        (df["asset"] == asset) & (df["horizon"] == horizon)
        & (df["feature_set_key"] == feature_set) & (df["model_name"] == model)
    ].sort_values("rank")
    return json.loads(match.to_json(orient="records"))


# ---------------------------------------------------------------------------
# Autoencoder-Encoded feature importance + latent-original correlation
# ---------------------------------------------------------------------------
# Mirrors the six Original-branch endpoints above exactly (same read-only
# pd.read_csv-and-return pattern, same 404-with-a-runnable-command style),
# just pointed at results/feature_importance_autoencoder/ instead of
# results/feature_importance/. Added because the earlier version of this
# analysis computed these CSVs but never exposed them through the API or
# the Streamlit tab -- see FEATURE_IMPORTANCE_TAB_DIAGNOSIS.md.

AE_FEATURE_IMPORTANCE_DIR = config.RESULTS_DIR / "feature_importance_autoencoder"


def _read_ae_csv(relative_path: str, hint_command: str):
    path = AE_FEATURE_IMPORTANCE_DIR / relative_path
    if not path.exists():
        raise HTTPException(404, f"{path} not found -- run {hint_command} first.")
    return pd.read_csv(path)


@app.get("/feature-importance-comparison-autoencoder")
def feature_importance_comparison_autoencoder():
    """The Autoencoder-Encoded-branch equivalent of
    /feature-importance-comparison -- RF vs XGBoost permutation
    importance for the 8 latent dimensions, across all 24 real
    (asset, horizon, feature configuration) combinations."""
    df = _read_ae_csv(
        "processed/cross_model_comparison_autoencoder.csv", "run_feature_importance_grid_autoencoder.py",
    )
    return json.loads(df.to_json(orient="records"))


@app.get("/feature-importance-agreement-summary-autoencoder")
def feature_importance_agreement_summary_autoencoder():
    """The Autoencoder-Encoded-branch equivalent of
    /feature-importance-agreement-summary."""
    df = _read_ae_csv(
        "processed/rank_agreement_summary_autoencoder.csv", "run_feature_importance_grid_autoencoder.py",
    )
    return json.loads(df.to_json(orient="records"))


@app.get("/feature-group-importance-autoencoder")
def feature_group_importance_autoencoder():
    """The Autoencoder-Encoded-branch equivalent of
    /feature-group-importance. All 8 latent dimensions fall under the
    'Other' group by design -- see run_feature_importance_grid_autoencoder.py's
    module docstring for why that's expected, not a bug."""
    df = _read_ae_csv(
        "processed/feature_group_importance_autoencoder.csv", "run_feature_importance_grid_autoencoder.py",
    )
    return json.loads(df.to_json(orient="records"))


@app.get("/latent-original-correlations")
def latent_original_correlations(asset: str = None, horizon: int = None, feature_set: str = None):
    """Descriptive Pearson + Spearman correlation between every latent
    dimension and every real original engineered feature -- see
    src/feature_importance/latent_correlation.py's module docstring for
    the interpretation caveats (descriptive only, not a one-to-one
    mapping, not causal). Optional filters narrow the (potentially
    large) long-format table to one configuration."""
    df = _read_ae_csv(
        "processed/latent_original_correlations_long.csv", "run_latent_original_correlation_analysis.py",
    )
    if asset:
        df = df[df["asset"] == asset]
    if horizon:
        df = df[df["horizon"] == horizon]
    if feature_set:
        df = df[df["feature_set_key"] == feature_set]
    return json.loads(df.to_json(orient="records"))


@app.get("/latent-original-top-associations")
def latent_original_top_associations(asset: str = None, horizon: int = None, feature_set: str = None):
    """Top-5 original-feature associations per latent dimension per
    configuration -- the readable summary of /latent-original-correlations."""
    df = _read_ae_csv(
        "processed/latent_original_top_associations.csv", "run_latent_original_correlation_analysis.py",
    )
    if asset:
        df = df[df["asset"] == asset]
    if horizon:
        df = df[df["horizon"] == horizon]
    if feature_set:
        df = df[df["feature_set_key"] == feature_set]
    return json.loads(df.to_json(orient="records"))


@app.get("/representation-comparison-summary")
def representation_comparison_summary():
    """The defensible Original-vs-Autoencoder-Encoded comparison -- test
    RMSE, within-configuration normalised top-feature importance, and
    importance concentration, side by side per (asset, horizon, feature
    configuration, model). See
    src/feature_importance/representation_comparison.py's module
    docstring for why raw permutation-importance magnitudes are not
    directly compared here."""
    df = _read_ae_csv(
        "processed/representation_comparison_summary.csv", "run_combined_feature_importance_report.py",
    )
    return json.loads(df.to_json(orient="records"))


@app.get("/arima-info")
def arima_info(asset: str = "Bitcoin", horizon: int = 1, feature_set: str = "primary"):
    """ARIMA's saved metadata -- order, training/validation/test periods,
    and the explicit, honest limitation note already recorded in the
    metadata itself (no fitted model file exists for ARIMA by design --
    see run_predict's docstring for why)."""
    try:
        meta = loaders.load_model_metadata(horizon, asset, feature_set, "ARIMA")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return meta


@app.get("/naive-info")
def naive_info(asset: str = "Bitcoin", horizon: int = 1, feature_set: str = "primary"):
    """Naive persistence's saved metadata -- the rule itself, training/
    validation/test periods, and the honest limitation note (no fitted
    object exists -- it's a parameterless rule, not a trained model)."""
    try:
        meta = loaders.load_model_metadata(horizon, asset, feature_set, "Naive")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return meta


@app.get("/last-training-date")
def last_training_date():
    """Real file-modification timestamp of the most recently written
    model file -- the manifest itself doesn't record a training
    timestamp, so this is read directly from disk rather than invented."""
    from datetime import datetime
    manifest = loaders.load_manifest()
    mtimes = []
    for entry in manifest:
        # Naive and ARIMA have no fitted model file by design (see
        # loaders.load_trained_model()'s docstring) -- their manifest
        # entries correctly have model_path=None, not a real bug in the
        # manifest. Skipping them here, rather than crashing on the first
        # one encountered, which is what was actually happening before.
        model_path = entry.get("model_path")
        if not model_path:
            continue
        path = config.PROJECT_ROOT / model_path
        if path.exists():
            mtimes.append(path.stat().st_mtime)
    if not mtimes:
        raise HTTPException(404, "No model files found on disk.")
    latest = max(mtimes)
    return {"last_training_date": datetime.fromtimestamp(latest).strftime("%Y-%m-%d")}


@app.get("/full-metrics")
def full_metrics():
    """The complete, real, saved metrics table (384 rows: every model x
    asset x horizon x feature configuration x representation x split
    combination from 07_model_development.ipynb) -- not recomputed, not
    filtered server-side. The frontend filters this for the interactive
    Model comparison page."""
    metrics = loaders.load_metrics()
    return metrics.to_dict("records")


@app.get("/predictions")
def predictions(asset: str, horizon: int, model: str, feature_set: str = "primary",
                 representation: str = "Original", split: str = "test"):
    """Real, saved test-set predictions for one combination -- actual vs.
    predicted price, for the actual-vs-predicted scatter and improvement-%
    calculations. Not recomputed."""
    display_names = {"primary": "Market Only", "candidate": "Market + Google Trends",
                      "ablation": "Market + Fear & Greed Index", "combined": "Market + Google Trends + Fear & Greed Index"}
    df = loaders.load_predictions()
    subset = df[
        (df["asset"] == asset) & (df["horizon"] == horizon) & (df["model"] == model)
        & (df["feature_set"] == display_names.get(feature_set, feature_set))
        & (df["representation"] == representation) & (df["split"] == split)
    ]
    if subset.empty:
        raise HTTPException(404, "No predictions found for this combination.")
    return subset.to_dict("records")
