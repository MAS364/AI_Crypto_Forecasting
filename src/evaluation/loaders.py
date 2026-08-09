"""Shared loading functions for evaluation -- reads already-saved artefacts
from disk (predictions, metrics, feature importance, trained models, the
autoencoder manifest) without retraining or recomputing anything.

This is what Notebook 08 (Evaluation) should import rather than
reimplementing file paths and loading logic inline. Every function here
is read-only against files already produced by 06_feature_engineering,
06b_autoencoder_feature_extraction, and 07_modeling (or their src/
equivalents) -- none of them train, fit, or predict anything.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src import config

MODELING_DIR = config.RESULTS_DIR / "modeling"
AUTOENCODER_DIR = config.RESULTS_DIR / "autoencoder"
MODELS_TRAINED_DIR = config.MODELS_DIR / "trained"


def load_predictions() -> pd.DataFrame:
    """Loads all_model_predictions.csv -- every model's predictions, every
    horizon, every feature configuration, both representations, in one
    dataframe. Raises FileNotFoundError with the exact path if Notebook 07
    (or its src equivalent) has not been run yet."""
    path = MODELING_DIR / "predictions" / "all_model_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 07_modeling.ipynb (or src.models.pipeline) first.")
    return pd.read_csv(path, parse_dates=["event_date"])


def load_metrics() -> pd.DataFrame:
    """Loads model_development_metrics.csv."""
    path = MODELING_DIR / "metrics" / "model_development_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 07_modeling.ipynb (or src.models.pipeline) first.")
    return pd.read_csv(path)


def load_feature_importance() -> pd.DataFrame:
    """Loads feature_importance.csv (Random Forest and XGBoost only, both representations)."""
    path = MODELING_DIR / "metrics" / "feature_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 07_modeling.ipynb (or src.models.pipeline) first.")
    return pd.read_csv(path)


def load_master_comparison() -> pd.DataFrame:
    """Loads the pre-built master_comparison_all_dimensions.csv, if Notebook
    07 has already saved it -- avoids rebuilding the same table twice.
    Falls back to building it from load_metrics() if the saved file isn't
    present (e.g. when running against a src.models.pipeline-only run)."""
    path = MODELING_DIR / "metrics" / "master_comparison_all_dimensions.csv"
    if path.exists():
        return pd.read_csv(path)
    from src.evaluation.comparison import build_master_comparison
    return build_master_comparison(load_metrics())


def load_metric_reconciliation() -> pd.DataFrame:
    """Loads the already-computed metric_reconciliation.csv (independent
    recomputation vs. stored metrics) -- Notebook 08 should report this
    check's result, not repeat it."""
    path = MODELING_DIR / "metrics" / "metric_reconciliation.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 07_modeling.ipynb's reconciliation cell first.")
    return pd.read_csv(path)


def load_autoencoder_manifest() -> pd.DataFrame:
    """Loads the autoencoder training manifest (reconstruction loss, epochs
    trained, feature counts) for all 24 (horizon, asset, feature_set)
    combinations."""
    path = AUTOENCODER_DIR / "metrics" / "autoencoder_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 06b_autoencoder_feature_extraction.ipynb first.")
    return pd.read_csv(path)


# Saved model directories on disk use human-readable names, not the
# internal feature-set keys used elsewhere in this project (verified
# directly against a real saved model directory, not assumed) --
# primary/candidate/ablation/combined map to these DIR_NAMES.
DIR_NAMES = {
    "primary": "market_only",
    "candidate": "market_google_trends",
    "ablation": "market_fear_greed",
    "combined": "market_all_external",
}


@lru_cache(maxsize=None)
def load_trained_model(horizon: int, asset: str, feature_set_key: str, model_name: str):
    """Loads a previously-saved, fitted model (Random Forest or XGBoost via
    joblib; LSTM via keras) for a specific (horizon, asset, feature
    configuration, model) combination -- never retrains. Returns the raw
    fitted model object; caller is responsible for feeding it correctly
    shaped input (matching src.models.data_prep.get_X_y's output).

    `feature_set_key` accepts either the internal key (primary/candidate/
    ablation/combined) or the on-disk directory name (market_only/etc.) --
    the internal key is translated automatically via DIR_NAMES.

    Cached in memory (@lru_cache) -- without this, every API request
    reloaded the file from disk from scratch, which for LSTM means
    rebuilding the entire Keras computation graph on every single
    prediction (a real, repeated cost, unlike joblib.load() for RF/
    XGBoost which is cheap enough that reloading barely mattered). Since
    this function only ever reads a fitted model and never mutates it,
    sharing one cached instance across requests is safe -- inference is
    read-only.
    """
    dir_name = DIR_NAMES.get(feature_set_key, feature_set_key)
    target_dir = MODELS_TRAINED_DIR / f"horizon_{horizon}day" / asset.lower() / dir_name
    model_slug = model_name.lower().replace(" ", "_")

    if model_name in ("Random Forest", "XGBoost"):
        path = target_dir / f"{model_slug}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        return joblib.load(path)

    if model_name == "LSTM":
        from tensorflow import keras
        path = target_dir / f"{model_slug}.keras"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        return keras.models.load_model(path)

    raise ValueError(
        f"load_trained_model() has no saved fitted object for '{model_name}' -- "
        "ARIMA has no picklable fitted object (only its order, in metadata), "
        "and Naive has no parameters to save at all."
    )


@lru_cache(maxsize=None)
def load_model_metadata(horizon: int, asset: str, feature_set_key: str, model_name: str) -> dict:
    """Loads the JSON metadata for any of the five models, including ARIMA
    and Naive (which have metadata but no fitted-model file). Accepts
    either the internal key or the on-disk directory name, same as
    load_trained_model()."""
    dir_name = DIR_NAMES.get(feature_set_key, feature_set_key)
    target_dir = MODELS_TRAINED_DIR / f"horizon_{horizon}day" / asset.lower() / dir_name
    slug_map = {
        "Random Forest": "random_forest", "XGBoost": "xgboost", "LSTM": "lstm",
        "ARIMA": "arima", "Naive": "naive_persistence",
    }
    if model_name not in slug_map:
        raise ValueError(f"Unknown model_name: {model_name}")
    path = target_dir / f"{slug_map[model_name]}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.load(open(path))


def load_manifest() -> list:
    """Loads the top-level models/trained/manifest.json -- the index of
    every saved model artefact across all horizons, assets, and feature
    configurations."""
    path = MODELS_TRAINED_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.load(open(path))
