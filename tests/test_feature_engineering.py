"""Tests for src/analysis/feature_engineering.py -- cumulative-return target
creation, leakage prevention, and horizon-awareness -- using real project
data, not synthetic fixtures, since the target definition is the single
most safety-critical piece of this whole project (a wrong target here
silently invalidates every downstream result).

Run: pytest tests/test_feature_engineering.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import feature_engineering as fe
from src import config

MERGED_DATA_PATH = config.PROCESSED_DATA_DIR / "merged_dataset.csv"


@pytest.fixture(scope="module")
def merged_df():
    if not MERGED_DATA_PATH.exists():
        pytest.skip(f"{MERGED_DATA_PATH} not found -- run Notebook 03 first.")
    return pd.read_csv(MERGED_DATA_PATH, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


class TestCumulativeReturnTarget:
    """The single most important correctness property in this project: the
    target must be the CUMULATIVE return over `horizon` days, not a single
    day's return that happens to fall `horizon` days ahead. These are
    different quantities, and confusing them was a real bug caught and
    fixed earlier in this project (see align_forecast_target's docstring)."""

    @pytest.mark.parametrize("horizon", [1, 7, 14])
    def test_target_matches_manual_price_ratio(self, merged_df, horizon):
        target_block = fe.align_forecast_target(merged_df, price_column="Bitcoin_Price", horizon=horizon)
        target_col = f"Bitcoin_Price_log_return_target_t_plus_{horizon}"

        row = 100
        manual = np.log(merged_df["Bitcoin_Price"].iloc[row + horizon] / merged_df["Bitcoin_Price"].iloc[row])
        from_function = target_block[target_col].iloc[row]

        assert from_function == pytest.approx(manual, abs=1e-10)

    def test_horizon_7_is_not_the_single_day_shift_trap(self, merged_df):
        """Explicitly guards against the exact bug found and fixed earlier:
        log_return.shift(-7) gives a single day's return 7 days out, which
        is NOT the same as the cumulative 7-day return."""
        target_block = fe.align_forecast_target(merged_df, price_column="Bitcoin_Price", horizon=7)
        target_col = "Bitcoin_Price_log_return_target_t_plus_7"

        log_return = np.log(merged_df["Bitcoin_Price"]).diff()
        wrong_trap_version = log_return.shift(-7)

        # The correct (cumulative) and incorrect (single-day-shift) versions
        # must differ -- if they were ever equal, the bug would have silently
        # reappeared.
        correct = target_block[target_col]
        aligned = pd.concat([correct, wrong_trap_version], axis=1).dropna()
        assert not np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1])

    def test_target_at_horizon_1_matches_simple_diff(self, merged_df):
        """At horizon=1, the cumulative target should reduce exactly to the
        original one-day return definition."""
        target_block = fe.align_forecast_target(merged_df, price_column="Bitcoin_Price", horizon=1)
        target_col = "Bitcoin_Price_log_return_target_t_plus_1"

        simple_next_day_return = np.log(merged_df["Bitcoin_Price"]).diff().shift(-1)
        aligned = pd.concat([target_block[target_col], simple_next_day_return], axis=1).dropna()
        np.testing.assert_allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], atol=1e-10)


class TestLeakagePrevention:
    @pytest.mark.parametrize("horizon", [1, 7, 14])
    def test_run_leakage_checks_all_pass(self, merged_df, horizon):
        feature_set = fe.build_feature_set(merged_df, asset="Bitcoin", feature_set_name="primary", horizon=horizon)
        checks = fe.run_leakage_checks(merged_df, feature_set, asset="Bitcoin", horizon=horizon)
        failed = checks[~checks["passed"]]
        assert failed.empty, f"Leakage checks failed:\n{failed}"

    def test_chronological_order_is_preserved(self, merged_df):
        feature_set = fe.build_feature_set(merged_df, asset="Bitcoin", feature_set_name="primary", horizon=1)
        assert feature_set["Date"].is_monotonic_increasing

    def test_no_duplicate_dates(self, merged_df):
        feature_set = fe.build_feature_set(merged_df, asset="Bitcoin", feature_set_name="primary", horizon=1)
        assert not feature_set["Date"].duplicated().any()


class TestTrailingReturnFeature:
    """The trailing-N-day-return feature (used by the Naive baseline at
    horizon > 1) must be backward-looking only -- built entirely from
    information available at time t, never from future data."""

    def test_trailing_feature_only_present_for_horizon_greater_than_1(self, merged_df):
        fs1 = fe.build_feature_set(merged_df, asset="Bitcoin", feature_set_name="primary", horizon=1)
        fs7 = fe.build_feature_set(merged_df, asset="Bitcoin", feature_set_name="primary", horizon=7)
        assert "Bitcoin_Price_trailing_7day_return" not in fs1.columns
        assert "Bitcoin_Price_trailing_7day_return" in fs7.columns

    def test_trailing_feature_uses_only_past_data(self, merged_df):
        fs7 = fe.build_feature_set(merged_df, asset="Bitcoin", feature_set_name="primary", horizon=7)
        row = 200
        # trailing_7day_return at row t should equal log(price_t / price_t-7) --
        # uses ONLY data at or before t, never t+1 or later.
        expected = np.log(merged_df["Bitcoin_Price"].iloc[row] / merged_df["Bitcoin_Price"].iloc[row - 7])
        actual = fs7["Bitcoin_Price_trailing_7day_return"].iloc[row]
        assert actual == pytest.approx(expected, abs=1e-10)
