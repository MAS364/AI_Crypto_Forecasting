"""Tests for src/models/data_prep.py -- chronological split, split
non-overlap, and predictor/target schema.

Run: pytest tests/test_data_prep.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.models.data_prep import split_data, get_X_y


def _make_synthetic_feature_set(n_rows=1000, horizon=1):
    """A minimal synthetic feature set with the same column structure as a
    real one (Date, price-level reference, target, and some predictors) --
    used here so split/schema tests don't depend on real data being present."""
    dates = pd.date_range("2021-01-01", periods=n_rows, freq="D")
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "Date": dates,
        "Test_Asset_Price_level": rng.uniform(20000, 60000, n_rows),
        f"Test_Asset_Price_log_return_target_t_plus_{horizon}": rng.randn(n_rows) * 0.02,
        "Test_Asset_Price_log_return_lag_1": rng.randn(n_rows) * 0.02,
        "Test_Asset_Volume": rng.uniform(1e9, 5e10, n_rows),
    })
    return df


class TestChronologicalSplit:
    def test_split_preserves_all_rows(self):
        df = _make_synthetic_feature_set(1000)
        train, val, test = split_data(df)
        assert len(train) + len(val) + len(test) == len(df)

    def test_split_proportions_are_approximately_70_15_15(self):
        df = _make_synthetic_feature_set(1000)
        train, val, test = split_data(df)
        assert len(train) / len(df) == pytest.approx(0.70, abs=0.01)
        assert len(val) / len(df) == pytest.approx(0.15, abs=0.01)
        assert len(test) / len(df) == pytest.approx(0.15, abs=0.02)

    def test_split_is_chronological_not_shuffled(self):
        df = _make_synthetic_feature_set(1000)
        train, val, test = split_data(df)
        # train should be entirely the earliest dates, in original order
        assert train["Date"].is_monotonic_increasing
        assert val["Date"].is_monotonic_increasing
        assert test["Date"].is_monotonic_increasing

    def test_custom_fractions_are_respected(self):
        df = _make_synthetic_feature_set(1000)
        train, val, test = split_data(df, train_frac=0.5, val_frac=0.25)
        assert len(train) / len(df) == pytest.approx(0.5, abs=0.01)
        assert len(val) / len(df) == pytest.approx(0.25, abs=0.01)


class TestSplitNonOverlap:
    """The function itself asserts non-overlap internally -- these tests
    additionally verify the property from the outside, and confirm the
    internal assertion actually catches a real overlap if one is
    artificially introduced."""

    def test_no_date_overlap_between_any_two_partitions(self):
        df = _make_synthetic_feature_set(1000)
        train, val, test = split_data(df)
        assert train["Date"].max() < val["Date"].min()
        assert val["Date"].max() < test["Date"].min()
        assert train["Date"].max() < test["Date"].min()

    def test_no_shared_rows_between_partitions(self):
        df = _make_synthetic_feature_set(1000)
        train, val, test = split_data(df)
        train_dates = set(train["Date"])
        val_dates = set(val["Date"])
        test_dates = set(test["Date"])
        assert train_dates.isdisjoint(val_dates)
        assert val_dates.isdisjoint(test_dates)
        assert train_dates.isdisjoint(test_dates)

    def test_internal_assertion_catches_artificial_overlap(self):
        """If someone shuffled the dataframe before calling split_data (a
        realistic way this could break), the internal assertion must raise,
        not silently produce leaky splits."""
        df = _make_synthetic_feature_set(1000)
        shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
        with pytest.raises(AssertionError):
            split_data(shuffled)


class TestGetXY:
    def test_target_column_excluded_from_predictors(self):
        df = _make_synthetic_feature_set(500, horizon=1)
        X, y = get_X_y(df, "Test_Asset_Price", horizon=1)
        assert "Test_Asset_Price_log_return_target_t_plus_1" not in X.columns

    def test_level_column_excluded_from_predictors(self):
        """The price-level reference column must never reach a model -- it
        exists only for reconstructing price forecasts afterward."""
        df = _make_synthetic_feature_set(500, horizon=1)
        X, y = get_X_y(df, "Test_Asset_Price", horizon=1)
        assert "Test_Asset_Price_level" not in X.columns

    def test_date_column_excluded_from_predictors(self):
        df = _make_synthetic_feature_set(500, horizon=1)
        X, y = get_X_y(df, "Test_Asset_Price", horizon=1)
        assert "Date" not in X.columns

    def test_y_matches_target_column_exactly(self):
        df = _make_synthetic_feature_set(500, horizon=1)
        X, y = get_X_y(df, "Test_Asset_Price", horizon=1)
        pd.testing.assert_series_equal(y, df["Test_Asset_Price_log_return_target_t_plus_1"], check_names=False)

    def test_different_horizons_select_different_target_columns(self):
        df7 = _make_synthetic_feature_set(500, horizon=7)
        X, y = get_X_y(df7, "Test_Asset_Price", horizon=7)
        assert y.name == "Test_Asset_Price_log_return_target_t_plus_7"
