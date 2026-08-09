"""Tests for src/evaluation/comparison.py and src/evaluation/loaders.py --
the shared modules Notebook 08 will use. These skip gracefully if
07_modeling.ipynb (or its src equivalent) hasn't been run yet, since they
depend on real saved output existing on disk.

Run: pytest tests/test_evaluation_shared.py -v
"""
import pytest

from src import config
from src.evaluation import comparison, loaders

METRICS_PATH = config.RESULTS_DIR / "modeling" / "metrics" / "model_development_metrics.csv"
PREDICTIONS_PATH = config.RESULTS_DIR / "modeling" / "predictions" / "all_model_predictions.csv"


def _require_saved_results():
    if not METRICS_PATH.exists():
        pytest.skip(f"{METRICS_PATH} not found -- run 07_modeling.ipynb first.")


class TestLoaders:
    def test_load_metrics_returns_expected_columns(self):
        _require_saved_results()
        df = loaders.load_metrics()
        required = {"horizon", "asset", "feature_set", "model", "representation", "split",
                     "RMSE_returns", "R2_returns", "Hit_Rate_%"}
        assert required.issubset(df.columns)

    def test_load_predictions_matches_qc_expected_schema(self):
        _require_saved_results()
        df = loaders.load_predictions()
        expected = {
            "horizon", "event_date", "asset", "model", "split", "feature_set", "representation",
            "price_level", "actual_return", "predicted_return", "return_residual",
            "actual_price", "predicted_price", "price_residual",
        }
        assert set(df.columns) == expected

    def test_load_missing_file_raises_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loaders, "MODELING_DIR", tmp_path / "does_not_exist")
        with pytest.raises(FileNotFoundError, match="run 07_modeling"):
            loaders.load_metrics()

    def test_load_trained_model_accepts_internal_key_and_dir_name_identically(self):
        """Regression test for a real bug found during this project's
        development: saved model directories use display names
        (market_only) not internal keys (primary) -- both must resolve to
        the same file."""
        try:
            model_via_key = loaders.load_trained_model(1, "Bitcoin", "primary", "Random Forest")
        except FileNotFoundError:
            pytest.skip("No saved Bitcoin/horizon_1day/primary Random Forest model found.")
        model_via_dirname = loaders.load_trained_model(1, "Bitcoin", "market_only", "Random Forest")
        assert type(model_via_key) == type(model_via_dirname)
        assert model_via_key.n_features_in_ == model_via_dirname.n_features_in_

    def test_load_trained_model_rejects_unsupported_model_names(self):
        with pytest.raises(ValueError, match="ARIMA"):
            loaders.load_trained_model(1, "Bitcoin", "primary", "ARIMA")


class TestComparison:
    def test_master_comparison_has_one_row_per_combination(self):
        _require_saved_results()
        metrics_df = loaders.load_metrics()
        master = comparison.build_master_comparison(metrics_df)
        # every row should be a unique (horizon, asset, feature_set, model, representation) combination
        key_cols = ["horizon", "asset", "feature_set", "model", "representation"]
        assert not master.duplicated(subset=key_cols).any()

    def test_master_comparison_only_contains_test_split(self):
        _require_saved_results()
        metrics_df = loaders.load_metrics()
        master = comparison.build_master_comparison(metrics_df)
        # split isn't a column in the output, but every row's source must have been split=="test" --
        # verified indirectly by checking row count matches a manual filter
        expected_count = len(metrics_df[metrics_df["split"] == "test"])
        assert len(master) == expected_count

    def test_validation_summary_only_contains_validation_split(self):
        _require_saved_results()
        metrics_df = loaders.load_metrics()
        val_summary = comparison.build_validation_summary(metrics_df)
        expected_count = len(metrics_df[metrics_df["split"] == "validation"])
        assert len(val_summary) == expected_count

    def test_run_qc_checks_passes_on_real_saved_predictions(self):
        _require_saved_results()
        all_preds = loaders.load_predictions()
        # QC checks that don't need the splits dict or log_df can still be
        # spot-checked directly here
        assert all_preds["predicted_return"].isnull().sum() == 0
        assert all_preds["predicted_price"].isnull().sum() == 0
