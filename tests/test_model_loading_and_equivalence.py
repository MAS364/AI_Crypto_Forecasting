"""Tests for saved model loading, prediction schema, and notebook/src
equivalence -- using real saved artefacts from this project where
available, skipping gracefully if they haven't been generated yet.

Run: pytest tests/test_model_loading_and_equivalence.py -v
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.models.data_prep import split_data, get_X_y
from src.models.tree_models import run_random_forest
from src.models.baseline import run_naive
from src.models.arima_model import run_arima

TRAINED_MODELS_DIR = config.MODELS_DIR / "trained"
FEATURES_DIR = config.FEATURES_DIR


def _real_feature_set_available():
    return (FEATURES_DIR / "bitcoin_market_only_features.csv").exists()


class TestSavedModelLoading:
    """Loading previously-saved models must never retrain -- these tests
    confirm a loaded model is immediately usable for prediction, with no
    .fit() call anywhere in the loading path."""

    def _find_a_saved_rf_model(self):
        candidates = list(TRAINED_MODELS_DIR.glob("**/random_forest.joblib"))
        if not candidates:
            pytest.skip("No saved Random Forest models found -- run Notebook 07 first.")
        return candidates[0]

    def test_saved_random_forest_loads_without_error(self):
        model_path = self._find_a_saved_rf_model()
        model = joblib.load(model_path)
        assert hasattr(model, "predict"), "Loaded object is not a fitted sklearn-style model"

    def test_loaded_model_predicts_without_retraining(self):
        """A loaded model must accept a prediction call immediately --
        sklearn raises NotFittedError if .fit() was never called, so this
        test would fail loudly if loading somehow returned an unfitted model."""
        model_path = self._find_a_saved_rf_model()
        model = joblib.load(model_path)

        n_features = model.n_features_in_
        dummy_X = np.random.randn(3, n_features)
        predictions = model.predict(dummy_X)  # would raise NotFittedError if unfitted
        assert predictions.shape == (3,)

    def test_saved_metadata_json_is_valid_and_has_expected_fields(self):
        candidates = list(TRAINED_MODELS_DIR.glob("**/random_forest_metadata.json"))
        if not candidates:
            pytest.skip("No saved Random Forest metadata found -- run Notebook 07 first.")
        import json
        metadata = json.load(open(candidates[0]))
        for field in ["cryptocurrency", "feature_set_key", "target", "feature_names"]:
            assert field in metadata, f"Missing expected metadata field: {field}"


class TestPredictionSchema:
    """Every model's run_X() function must return the same {"predictions":
    ..., "metrics": ...} schema, with predictions containing the same
    columns, regardless of which model produced them -- this is what lets
    Notebook 07 (and any future Notebook 08) treat all five models
    uniformly."""

    @pytest.fixture(scope="class")
    def real_splits(self):
        if not _real_feature_set_available():
            pytest.skip("Real feature set not found -- run Notebook 06 first.")
        df = pd.read_csv(FEATURES_DIR / "bitcoin_market_only_features.csv", parse_dates=["Date"])
        return split_data(df)

    def test_naive_prediction_schema(self, real_splits):
        train, val, test = real_splits
        result = run_naive(train, val, test, "Bitcoin_Price", horizon=1)
        self._assert_valid_schema(result)

    def test_random_forest_prediction_schema(self, real_splits):
        train, val, test = real_splits
        result = run_random_forest(train, val, test, "Bitcoin_Price", horizon=1)
        self._assert_valid_schema(result)
        assert "importance" in result
        assert set(result["importance"].columns) >= {"feature", "importance"}

    def test_arima_prediction_schema(self, real_splits):
        train, val, test = real_splits
        result = run_arima(train, val, test, "Bitcoin_Price", horizon=1)
        self._assert_valid_schema(result)
        assert "order" in result

    @staticmethod
    def _assert_valid_schema(result):
        assert "predictions" in result and "metrics" in result
        for split_name in ["validation", "test"]:
            assert split_name in result["predictions"]
            pred_df = result["predictions"][split_name]
            assert {"event_date", "actual", "predicted"}.issubset(pred_df.columns)

            metrics = result["metrics"][split_name]
            required = {"RMSE_returns", "MAE_returns", "R2_returns", "DirAcc_%", "Hit_Rate_%"}
            assert required.issubset(metrics.keys())
            assert metrics["Hit_Rate_%"] == metrics["DirAcc_%"]


class TestNotebookSrcEquivalence:
    """The core architectural claim this project makes: Notebook 07 and
    src/models/ compute identically, because Notebook 07 imports these
    exact functions rather than maintaining a separate copy. These tests
    verify the deterministic models produce the same known values found
    and recorded earlier in this project's development."""

    @pytest.fixture(scope="class")
    def real_splits(self):
        if not _real_feature_set_available():
            pytest.skip("Real feature set not found -- run Notebook 06 first.")
        df = pd.read_csv(FEATURES_DIR / "bitcoin_market_only_features.csv", parse_dates=["Date"])
        return split_data(df)

    def test_arima_bitcoin_1day_matches_known_value(self, real_splits):
        """ARIMA is fully deterministic (no randomness in AIC-based order
        selection) -- this exact value was independently verified multiple
        times during this project's development."""
        train, val, test = real_splits
        result = run_arima(train, val, test, "Bitcoin_Price", horizon=1)
        assert result["metrics"]["test"]["RMSE_returns"] == pytest.approx(    0.024480423539931285, abs=1e-5)

    def test_random_forest_is_reproducible_across_two_runs(self, real_splits):
        """Same seed, same data -> identical result. This is the
        reproducibility guarantee this project relies on."""
        train, val, test = real_splits
        result1 = run_random_forest(train, val, test, "Bitcoin_Price", horizon=1,
                                     grid=[{"n_estimators": 50, "max_depth": 2, "min_samples_split": 5,
                                            "min_samples_leaf": 2, "max_features": "sqrt"}])
        result2 = run_random_forest(train, val, test, "Bitcoin_Price", horizon=1,
                                     grid=[{"n_estimators": 50, "max_depth": 2, "min_samples_split": 5,
                                            "min_samples_leaf": 2, "max_features": "sqrt"}])
        assert result1["metrics"]["test"]["RMSE_returns"] == result2["metrics"]["test"]["RMSE_returns"]
