"""Tests for deployment/app.py -- the FastAPI inference service. Uses
FastAPI's TestClient (no real server needed). Skips gracefully where real
saved models aren't present.

Run: pytest tests/test_deployment_app.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from deployment.app import app
from src import config

client = TestClient(app)


def _real_feature_set_available():
    return (config.FEATURES_DIR / "bitcoin_market_only_features.csv").exists()


def _sample_rows():
    df = pd.read_csv(config.FEATURES_DIR / "bitcoin_market_only_features.csv", parse_dates=["Date"])
    sample = df.tail(5).copy()
    sample["Date"] = sample["Date"].astype(str)
    return sample.to_dict("records")


class TestRootAndHealth:
    def test_root_returns_200_with_expected_structure(self):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert set(["valid_assets", "valid_horizons", "valid_feature_sets", "valid_representations", "valid_models"]).issubset(body.keys())

    def test_root_lists_only_models_with_saved_fitted_objects(self):
        """Naive and ARIMA must never appear as predictable models here --
        they have no saved fitted object to load."""
        r = client.get("/")
        valid_models = r.json()["valid_models"]
        assert "Naive" not in valid_models
        assert "ARIMA" not in valid_models
        assert set(valid_models) == {"Random Forest", "XGBoost", "LSTM"}

    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestPredictEndpoint:
    def test_valid_request_returns_200_with_real_model(self):
        if not _real_feature_set_available():
            pytest.skip("Real feature set not found -- run Notebook 06 first.")
        payload = {
            "horizon": 1, "asset": "Bitcoin", "feature_set": "primary",
            "representation": "Original", "model": "Random Forest",
            "rows": _sample_rows(),
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["asset"] == "Bitcoin"
        assert isinstance(body["predicted_price"], float)
        assert body["predicted_price"] > 0

    def test_valid_request_matches_direct_run_predict_call(self):
        """The API layer must not alter the underlying prediction -- same
        inputs through the endpoint and through run_predict() directly
        should agree exactly."""
        if not _real_feature_set_available():
            pytest.skip("Real feature set not found.")
        from src.pipeline import run_predict
        df = pd.read_csv(config.FEATURES_DIR / "bitcoin_market_only_features.csv", parse_dates=["Date"])
        sample = df.tail(5)

        direct_result = run_predict(horizon=1, asset="Bitcoin", feature_set_key="primary",
                                     model_name="Random Forest", representation="Original", input_df=sample)

        payload = {
            "horizon": 1, "asset": "Bitcoin", "feature_set": "primary",
            "representation": "Original", "model": "Random Forest",
            "rows": _sample_rows(),
        }
        api_result = client.post("/predict", json=payload).json()

        assert api_result["predicted_return"] == pytest.approx(direct_result["predicted_return"], rel=1e-9)

    def test_arima_rejected_with_400(self):
        payload = {"horizon": 1, "asset": "Bitcoin", "feature_set": "primary",
                   "representation": "Original", "model": "ARIMA", "rows": []}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400
        assert "ARIMA" in r.json()["detail"]

    def test_unknown_asset_rejected_with_400(self):
        payload = {"horizon": 1, "asset": "Dogecoin", "feature_set": "primary",
                   "representation": "Original", "model": "Random Forest", "rows": []}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400
        assert "Dogecoin" in r.json()["detail"]

    def test_unknown_horizon_rejected_with_400(self):
        payload = {"horizon": 99, "asset": "Bitcoin", "feature_set": "primary",
                   "representation": "Original", "model": "Random Forest", "rows": []}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400
        assert "99" in r.json()["detail"]

    def test_unknown_feature_set_rejected_with_400(self):
        payload = {"horizon": 1, "asset": "Bitcoin", "feature_set": "not_a_real_config",
                   "representation": "Original", "model": "Random Forest", "rows": []}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400

    def test_missing_required_field_rejected_with_422(self):
        """Pydantic's own validation -- missing a required field should
        fail before run_predict() is ever called."""
        payload = {"asset": "Bitcoin", "feature_set": "primary", "representation": "Original", "model": "Random Forest", "rows": []}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_malformed_rows_rejected_with_400(self):
        if not _real_feature_set_available():
            pytest.skip("Real feature set not found.")
        payload = {
            "horizon": 1, "asset": "Bitcoin", "feature_set": "primary",
            "representation": "Original", "model": "Random Forest",
            "rows": [{"nonsense_column": "abc"}],
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 400
