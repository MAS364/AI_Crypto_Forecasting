"""Tests for deployment/csv_inference.py and the new endpoints in
deployment/app.py. Skips gracefully where real saved artefacts (models,
merged dataset) aren't present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deployment"))

import pandas as pd
import pytest

from src import config
from csv_inference import (
    load_csv, validate_csv, run_csv_inference, CSVValidationError,
    _determine_feature_set_key, MIN_ROWS_FOR_FEATURES,
)


def _merged_dataset_available():
    return config.MERGED_FILE.exists()


@pytest.fixture(scope="module")
def real_sample_csv_bytes():
    if not _merged_dataset_available():
        pytest.skip("merged_dataset.csv not found -- run 03_data_preparation.ipynb first.")
    df = pd.read_csv(config.MERGED_FILE, parse_dates=["Date"])
    sample = df.tail(45)[["Date", "Bitcoin_Price", "Bitcoin_Market_Cap", "Bitcoin_Volume",
                           "Bitcoin_Search_Interest", "Fear_Greed_Index", "Market_Sentiment"]]
    return sample.to_csv(index=False).encode()


class TestLoadCSV:
    def test_loads_valid_csv(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes)
        assert len(df) == 45

    def test_rejects_malformed_bytes(self):
        with pytest.raises(CSVValidationError):
            load_csv(b"this is not a csv \x00\x01\x02")

    def test_rejects_empty_csv(self):
        with pytest.raises(CSVValidationError, match="no rows"):
            load_csv(b"Date,Bitcoin_Price\n")


class TestValidateCSV:
    def test_valid_data_passes(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes)
        validate_csv(df, "Bitcoin", 1, "Random Forest")  # should not raise

    def test_too_few_rows_rejected(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes).head(10)
        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv(df, "Bitcoin", 1, "Random Forest")
        assert any("30" in p for p in exc_info.value.problems)

    def test_missing_required_column_rejected(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes).drop(columns=["Bitcoin_Volume"])
        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv(df, "Bitcoin", 1, "Random Forest")
        assert any("Bitcoin_Volume" in p for p in exc_info.value.problems)

    def test_negative_price_rejected(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes)
        df.loc[0, "Bitcoin_Price"] = -100
        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv(df, "Bitcoin", 1, "Random Forest")
        assert any("negative" in p for p in exc_info.value.problems)

    def test_duplicate_dates_rejected(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes)
        df.loc[1, "Date"] = df.loc[0, "Date"]
        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv(df, "Bitcoin", 1, "Random Forest")
        assert any("duplicate" in p.lower() for p in exc_info.value.problems)

    def test_arima_accepted_and_produces_real_prediction(self, real_sample_csv_bytes):
        """ARIMA was made to work in CSV upload (fits fresh with its
        already-selected order, never re-tuned) -- this replaces the old
        test that expected rejection, which became stale and started
        failing once that feature was added without updating this test."""
        from csv_inference import run_arima_inference
        df = load_csv(real_sample_csv_bytes)
        validate_csv(df, "Bitcoin", 1, "ARIMA")  # should NOT raise
        result = run_arima_inference(df, "Bitcoin", 1)
        assert result["predicted_price"] > 0
        assert "arima_order" in result

    def test_naive_rejected_with_clear_reason(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes)
        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv(df, "Bitcoin", 1, "Naive")
        assert any("Naive" in p for p in exc_info.value.problems)

    def test_fear_greed_without_sentiment_rejected(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes).drop(columns=["Market_Sentiment"])
        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv(df, "Bitcoin", 1, "Random Forest")
        assert any("Market_Sentiment" in p for p in exc_info.value.problems)

    def test_multiple_problems_all_reported_together(self, real_sample_csv_bytes):
        """A user fixing their CSV shouldn't have to re-upload once per error."""
        df = load_csv(real_sample_csv_bytes).head(10).drop(columns=["Market_Sentiment"])
        with pytest.raises(CSVValidationError) as exc_info:
            validate_csv(df, "Bitcoin", 1, "Random Forest")
        assert len(exc_info.value.problems) >= 2


class TestDetermineFeatureSetKey:
    def test_all_optional_columns_present_gives_combined(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes)
        assert _determine_feature_set_key(df, "Bitcoin") == "combined"

    def test_no_optional_columns_gives_primary(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes)[["Date", "Bitcoin_Price", "Bitcoin_Market_Cap", "Bitcoin_Volume"]]
        assert _determine_feature_set_key(df, "Bitcoin") == "primary"

    def test_only_search_interest_gives_candidate(self, real_sample_csv_bytes):
        df = load_csv(real_sample_csv_bytes).drop(columns=["Fear_Greed_Index", "Market_Sentiment"])
        assert _determine_feature_set_key(df, "Bitcoin") == "candidate"


class TestRunCSVInference:
    def test_full_pipeline_produces_real_prediction(self, real_sample_csv_bytes):
        result = run_csv_inference(real_sample_csv_bytes, asset="Bitcoin", model_name="Random Forest", horizon=1)
        assert result["predicted_price"] > 0
        assert result["feature_set_key"] == "combined"
        assert result["n_rows_uploaded"] == 45

    def test_reconstruction_formula_holds(self, real_sample_csv_bytes):
        import numpy as np
        result = run_csv_inference(real_sample_csv_bytes, asset="Bitcoin", model_name="Random Forest", horizon=1)
        expected = result["input_price_level"] * np.exp(result["predicted_return"])
        assert result["predicted_price"] == pytest.approx(expected, rel=1e-9)


class TestAPIEndpoints:
    """Tests the new FastAPI endpoints via TestClient -- no real server needed."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        import importlib
        app_module = importlib.import_module("app")
        return TestClient(app_module.app)

    def test_csv_requirements_endpoint(self, client):
        r = client.get("/csv-requirements", params={"asset": "Bitcoin"})
        assert r.status_code == 200
        assert r.json()["min_rows"] == MIN_ROWS_FOR_FEATURES

    def test_models_endpoint(self, client):
        r = client.get("/models")
        if r.status_code == 404:
            pytest.skip("No trained model manifest found.")
        assert r.status_code == 200
        assert r.json()["count"] > 0

    def test_predict_csv_endpoint_end_to_end(self, client, real_sample_csv_bytes):
        r = client.post(
            "/predict/csv",
            files={"file": ("test.csv", real_sample_csv_bytes, "text/csv")},
            data={"asset": "Bitcoin", "model": "Random Forest", "horizon": 1, "representation": "Original"},
        )
        assert r.status_code == 200
        assert r.json()["predicted_price"] > 0

    def test_predict_csv_returns_422_with_itemised_errors(self, client):
        small_csv = b"Date,Bitcoin_Price,Bitcoin_Market_Cap,Bitcoin_Volume\n2026-01-01,100,1000,10\n"
        r = client.post(
            "/predict/csv",
            files={"file": ("small.csv", small_csv, "text/csv")},
            data={"asset": "Bitcoin", "model": "Random Forest", "horizon": 1},
        )
        assert r.status_code == 422
        assert "validation_errors" in r.json()["detail"]
