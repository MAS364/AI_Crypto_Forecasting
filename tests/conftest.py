"""Shared pytest fixtures/helpers for repository-level tests."""
from pathlib import Path
import pytest
from src import config

import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIR_NAMES = {"primary":"market_only","candidate":"market_google_trends","ablation":"market_fear_greed","combined":"market_all_external"}

def feature_file_available(asset="bitcoin", feature_set="market_only"):
    return (config.FEATURES_DIR / f"{asset}_{feature_set}_features.csv").exists()

def merged_dataset_available():
    return config.MERGED_FILE.exists()

def saved_predictions_available():
    return (config.RESULTS_DIR / "modeling" / "predictions" / "all_model_predictions.csv").exists()

def trained_model_path(horizon, asset, feature_set_key, model_name):
    dir_name = DIR_NAMES.get(feature_set_key, feature_set_key)
    slug = model_name.lower().replace(" ", "_")
    suffix = ".keras" if model_name == "LSTM" else ".joblib"
    return config.MODELS_DIR / "trained" / f"horizon_{horizon}day" / asset.lower() / dir_name / f"{slug}{suffix}"

@pytest.fixture(scope="session")
def real_feature_set_available():
    return feature_file_available()

@pytest.fixture(scope="session")
def merged_data_available():
    return merged_dataset_available()

@pytest.fixture(scope="session")
def saved_predictions_are_available():
    return saved_predictions_available()

@pytest.fixture(scope="session")
def require_trained_model():
    def _require(horizon, asset, feature_set_key, model_name):
        path = trained_model_path(horizon, asset, feature_set_key, model_name)
        if not path.exists():
            pytest.skip(f"Required trained model not found: {path}")
        return path
    return _require
