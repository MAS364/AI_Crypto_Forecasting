"""Lightweight unit tests for src.preprocessing using synthetic data only."""
import pandas as pd
import pytest

from src.preprocessing.cleaning import convert_dates, filter_study_period
from src.preprocessing.encoding import encode_sentiment, decode_sentiment
from src.preprocessing.integration import merge_datasets
from src.preprocessing.pipeline import load_processed_data
from src.preprocessing.validation import check_missing_values, count_duplicate_dates


def test_convert_dates_parses_and_renames_without_mutating_input():
    raw = pd.DataFrame({"timestamp": ["2024-01-02", "2024-01-01"], "value": [2, 1]})
    cleaned = convert_dates(raw, "timestamp", drop_timezone=False)
    assert list(cleaned.columns) == ["Date", "value"]
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
    assert "timestamp" in raw.columns


def test_filter_study_period_is_inclusive():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=5), "x": range(5)})
    out = filter_study_period(df, "2024-01-02", "2024-01-04")
    assert out["Date"].tolist() == list(pd.date_range("2024-01-02", "2024-01-04"))


def test_filter_study_period_rejects_invalid_range():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=2)})
    with pytest.raises(ValueError, match="after study end"):
        filter_study_period(df, "2024-01-03", "2024-01-01")


def test_encode_and_decode_sentiment_round_trip():
    mapping = {"Fear": 0, "Neutral": 1, "Greed": 2}
    df = pd.DataFrame({"Market_Sentiment": ["Fear", "Neutral", "Greed"]})
    encoded = encode_sentiment(df, mapping=mapping)
    decoded = decode_sentiment(encoded["Market_Sentiment_Encoded"], mapping=mapping)
    assert decoded.tolist() == df["Market_Sentiment"].tolist()


def test_encode_sentiment_rejects_unknown_category():
    df = pd.DataFrame({"Market_Sentiment": ["Unknown"]})
    with pytest.raises(ValueError, match="Unmapped sentiment"):
        encode_sentiment(df, mapping={"Fear": 0})


def test_merge_datasets_inner_joins_and_sorts():
    dates = pd.to_datetime(["2024-01-02", "2024-01-01"])
    datasets = {
        "bitcoin": pd.DataFrame({"Date": dates, "btc": [2, 1]}),
        "ethereum": pd.DataFrame({"Date": dates, "eth": [20, 10]}),
        "google_trends": pd.DataFrame({"Date": dates, "trend": [50, 40]}),
        "fear_greed": pd.DataFrame({"Date": dates, "fg": [60, 55]}),
    }
    merged = merge_datasets(datasets)
    assert merged["Date"].is_monotonic_increasing
    assert list(merged.columns) == ["Date", "btc", "eth", "trend", "fg"]
    assert len(merged) == 2


def test_missing_values_are_rejected():
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01"]), "x": [None]})
    with pytest.raises(ValueError, match="missing values"):
        check_missing_values(df)


def test_duplicate_date_count():
    df = pd.DataFrame({"Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])})
    assert count_duplicate_dates(df) == 1


def test_load_processed_data_reads_dates(tmp_path):
    path = tmp_path / "merged.csv"
    pd.DataFrame({"Date": ["2024-01-01"], "value": [1]}).to_csv(path, index=False)
    loaded = load_processed_data(path)
    assert pd.api.types.is_datetime64_any_dtype(loaded["Date"])


def test_load_processed_data_missing_file_is_clear(tmp_path):
    with pytest.raises(FileNotFoundError, match="Processed dataset not found"):
        load_processed_data(tmp_path / "missing.csv")
