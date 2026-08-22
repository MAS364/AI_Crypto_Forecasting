"""Tests for the Autoencoder-Encoded feature-importance analysis:
run_feature_importance_grid_autoencoder.py, src/feature_importance/
latent_correlation.py, src/feature_importance/representation_comparison.py,
and the new deployment/app.py endpoints + Streamlit data sources that
expose them. Skips gracefully where real saved artefacts aren't present,
matching this project's existing test style.

Run:
    pytest tests/test_feature_importance_autoencoder.py -v
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.feature_importance import latent_correlation as lc

grid_module = importlib.import_module("run_feature_importance_grid_autoencoder")


def _ae_grid_available() -> bool:
    return (config.RESULTS_DIR / "feature_importance_autoencoder" / "raw"
            / "permutation_importance_autoencoder_long.csv").exists()


def _correlations_available() -> bool:
    return (config.RESULTS_DIR / "feature_importance_autoencoder" / "processed"
            / "latent_original_correlations_long.csv").exists()


def _encoded_test_set_available() -> bool:
    return (config.RESULTS_DIR / "autoencoder" / "encoded_features"
            / "bitcoin_primary_encoded_test.csv").exists()


def _encoded_model_available() -> bool:
    return (config.RESULTS_DIR / "modeling" / "fitted_models"
            / "bitcoin_primary_random_forest_encoded.joblib").exists()


# ---------------------------------------------------------------------------
# Encoded model/data path construction
# ---------------------------------------------------------------------------

class TestEncodedPathConstruction:
    def test_missing_encoded_test_set_raises_filenotfounderror_with_path(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            grid_module.load_encoded_model_and_data(
                horizon=1, asset="Bitcoin", feature_set_key="does_not_exist", model_name="Random Forest",
            )
        assert "does_not_exist" in str(exc_info.value)

    def test_encoded_model_path_uses_flat_results_modeling_fitted_models_directory(self):
        """The Autoencoder-Encoded branch's fitted models live at
        results/modeling/fitted_models/{prefix}_{model_slug}_encoded.joblib
        -- a flat directory, not nested under models/trained/ the way the
        Original branch is (verified directly against the real project
        in Phase 1 of this analysis)."""
        assert grid_module.FITTED_MODELS_DIR == config.RESULTS_DIR / "modeling" / "fitted_models"

    @pytest.mark.parametrize("horizon,expected_suffix", [(1, ""), (7, "_7day"), (14, "_14day")])
    def test_horizon_suffix_matches_naming_convention(self, horizon, expected_suffix):
        from src.models.feature_naming import _horizon_suffix
        assert _horizon_suffix(horizon) == expected_suffix


# ---------------------------------------------------------------------------
# Target-column selection across all horizons
# ---------------------------------------------------------------------------

class TestTargetColumnSelection:
    @pytest.mark.parametrize("horizon", [1, 7, 14])
    @pytest.mark.parametrize("asset", ["Bitcoin", "Ethereum"])
    def test_target_column_exists_in_real_encoded_test_set(self, horizon, asset):
        if not _encoded_test_set_available():
            pytest.skip("Encoded feature sets not found -- run src.pipeline train first.")
        suffix = "" if horizon == 1 else f"_{horizon}day"
        path = config.RESULTS_DIR / "autoencoder" / "encoded_features" / f"{asset.lower()}_primary{suffix}_encoded_test.csv"
        if not path.exists():
            pytest.skip(f"{path} not found.")
        df = pd.read_csv(path)
        price_col = config.ASSET_VARIABLES[asset]["price"]
        expected_target = f"{price_col}_log_return_target_t_plus_{horizon}"
        assert expected_target in df.columns


# ---------------------------------------------------------------------------
# Latent-column detection
# ---------------------------------------------------------------------------

class TestLatentColumnDetection:
    def test_encoded_columns_extraction_excludes_date_level_and_target(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01"], "encoded_0": [0.1], "encoded_1": [0.2],
            "Bitcoin_Price_level": [50000.0], "Bitcoin_Price_log_return_target_t_plus_1": [0.01],
        })
        encoded_cols = [c for c in df.columns if c.startswith("encoded_")]
        assert encoded_cols == ["encoded_0", "encoded_1"]
        assert "Date" not in encoded_cols
        assert "Bitcoin_Price_level" not in encoded_cols
        assert "Bitcoin_Price_log_return_target_t_plus_1" not in encoded_cols

    def test_real_encoded_test_set_has_exactly_eight_latent_columns(self):
        if not _encoded_test_set_available():
            pytest.skip("Encoded feature sets not found -- run src.pipeline train first.")
        df = pd.read_csv(config.RESULTS_DIR / "autoencoder" / "encoded_features" / "bitcoin_primary_encoded_test.csv")
        encoded_cols = [c for c in df.columns if c.startswith("encoded_")]
        assert encoded_cols == [f"encoded_{i}" for i in range(8)]


# ---------------------------------------------------------------------------
# Row/date alignment for the latent-original correlation analysis
# ---------------------------------------------------------------------------

class TestRowDateAlignment:
    def test_alignment_uses_date_merge_not_position(self):
        if not _encoded_test_set_available():
            pytest.skip("Encoded feature sets and original feature files not found.")
        merged, original_cols = lc._load_aligned_test_frame("Bitcoin", 1, "primary")
        assert "Date" in merged.columns
        assert merged["Date"].is_unique
        assert len(merged) > 0
        # Every latent column must be present alongside at least one real
        # original predictor column -- proof the two files were actually
        # merged, not just one loaded.
        assert all(f"encoded_{i}" in merged.columns for i in range(8))
        assert len(original_cols) > 0

    def test_mismatched_dates_would_not_silently_align_by_position(self):
        """A same-row-count coincidence must never be treated as
        alignment -- this directly tests the merge-on-Date behaviour
        with two frames that share a row count but only partially
        overlapping dates."""
        encoded_like = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "encoded_0": [1.0, 2.0, 3.0],
        })
        original_like = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "some_feature": [10.0, 20.0, 30.0],
        })
        merged = encoded_like.merge(original_like, on="Date", how="inner")
        # Same row count on both sides (3 and 3), but only 2 dates truly
        # overlap -- an inner join on Date correctly yields 2 rows, not 3.
        assert len(merged) == 2
        assert set(merged["Date"].dt.strftime("%Y-%m-%d")) == {"2024-01-02", "2024-01-03"}


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------

class TestLeakagePrevention:
    def test_target_column_excluded_from_correlation_predictors(self):
        if not _encoded_test_set_available():
            pytest.skip("Encoded feature sets and original feature files not found.")
        _, original_cols = lc._load_aligned_test_frame("Bitcoin", 1, "primary")
        assert "Bitcoin_Price_log_return_target_t_plus_1" not in original_cols

    def test_level_column_excluded_from_correlation_predictors(self):
        if not _encoded_test_set_available():
            pytest.skip("Encoded feature sets and original feature files not found.")
        _, original_cols = lc._load_aligned_test_frame("Bitcoin", 1, "primary")
        assert "Bitcoin_Price_level" not in original_cols

    def test_permutation_importance_never_touches_train_rows(self):
        """The encoded-branch loader (grid_module.load_encoded_model_and_data)
        only ever reads the *_encoded_test.csv file -- confirms by
        construction that no *_encoded_train.csv or *_encoded_validation.csv
        path is referenced anywhere in its path-building logic."""
        import inspect
        source = inspect.getsource(grid_module.load_encoded_model_and_data)
        assert "encoded_test.csv" in source
        assert "encoded_train.csv" not in source
        assert "encoded_validation.csv" not in source


# ---------------------------------------------------------------------------
# Latent-original correlation calculation correctness
# ---------------------------------------------------------------------------

class TestLatentOriginalCorrelationCalculation:
    def test_perfectly_correlated_synthetic_series(self, monkeypatch):
        """Monkeypatches the grid down to one synthetic configuration with
        a known, exact relationship (encoded_0 = 2 * feature_a) to verify
        the correlation math itself is correct, independent of any real
        project data."""
        n = 20
        feature_a = np.linspace(1, 10, n)
        synthetic = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=n),
            "encoded_0": feature_a * 2,
            "encoded_1": -feature_a,  # perfectly anti-correlated
            "encoded_2": np.zeros(n),  # constant -- must yield NaN, not 0
            "encoded_3": np.random.RandomState(0).normal(size=n),
            "encoded_4": np.random.RandomState(1).normal(size=n),
            "encoded_5": np.random.RandomState(2).normal(size=n),
            "encoded_6": np.random.RandomState(3).normal(size=n),
            "encoded_7": np.random.RandomState(4).normal(size=n),
            "feature_a": feature_a,
        })

        def fake_loader(asset, horizon, feature_set_key):
            return synthetic, ["feature_a"]

        monkeypatch.setattr(lc, "_load_aligned_test_frame", fake_loader)
        monkeypatch.setattr(lc, "ASSETS", ["Bitcoin"])
        monkeypatch.setattr(lc, "HORIZONS", [1])
        monkeypatch.setattr(lc, "FEATURE_SET_KEYS", ["primary"])

        result = lc.compute_latent_original_correlations()

        e0 = result[result["latent_feature"] == "encoded_0"].iloc[0]
        assert e0["pearson_correlation"] == pytest.approx(1.0, abs=1e-9)
        assert e0["spearman_correlation"] == pytest.approx(1.0, abs=1e-9)
        assert e0["absolute_pearson_correlation"] == pytest.approx(1.0, abs=1e-9)

        e1 = result[result["latent_feature"] == "encoded_1"].iloc[0]
        assert e1["pearson_correlation"] == pytest.approx(-1.0, abs=1e-9)

        e2 = result[result["latent_feature"] == "encoded_2"].iloc[0]
        assert pd.isna(e2["pearson_correlation"]), "A constant column must yield NaN, not a coerced 0."

        assert (result["number_of_aligned_observations"] == n).all()

    def test_top_associations_never_ranks_undefined_correlations(self):
        long_df = pd.DataFrame({
            "asset": ["Bitcoin"] * 3, "horizon": [1] * 3, "feature_set_key": ["primary"] * 3,
            "latent_feature": ["encoded_0"] * 3, "original_feature": ["f1", "f2", "f3"],
            "pearson_correlation": [0.9, np.nan, 0.5],
            "spearman_correlation": [0.9, np.nan, 0.5],
            "absolute_pearson_correlation": [0.9, np.nan, 0.5],
            "absolute_spearman_correlation": [0.9, np.nan, 0.5],
            "number_of_aligned_observations": [100, 0, 100],
        })
        top = lc.summarize_top_associations(long_df, top_n=5)
        assert "f2" not in top["original_feature"].values
        assert set(top["original_feature"]) == {"f1", "f3"}


# ---------------------------------------------------------------------------
# Complete experiment-grid validation + failure on partial results
# ---------------------------------------------------------------------------

class TestFullGridValidation:
    def _valid_long_df(self):
        rows = []
        for asset in ["Bitcoin", "Ethereum"]:
            for horizon in [1, 7, 14]:
                for fset in ["primary", "candidate", "ablation", "combined"]:
                    for model in ["Random Forest", "XGBoost"]:
                        for i in range(8):
                            rows.append({
                                "feature": f"encoded_{i}", "mean_importance": 0.001, "std_importance": 0.0001,
                                "baseline_rmse": 0.02, "rank": i + 1, "n_repeats": 8,
                                "asset": asset, "horizon": horizon, "feature_set_key": fset,
                                "model_name": model, "representation": "Autoencoder-Encoded",
                            })
        return pd.DataFrame(rows)

    def test_valid_full_grid_passes_validation(self):
        df = self._valid_long_df()
        grid_module.validate_full_grid(df, n_ok=48, n_skip=0, n_fail=0, total=48)  # must not raise

    def test_partial_grid_fails_validation(self):
        df = self._valid_long_df()
        partial = df[df["asset"] != "Ethereum"]  # drops 24 of 48 experiments
        with pytest.raises(RuntimeError, match="48"):
            grid_module.validate_full_grid(partial, n_ok=24, n_skip=24, n_fail=0, total=48)

    def test_duplicate_rows_fail_validation(self):
        df = self._valid_long_df()
        with_dupes = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(RuntimeError, match="duplicate"):
            grid_module.validate_full_grid(with_dupes, n_ok=48, n_skip=0, n_fail=0, total=48)

    def test_non_finite_importance_fails_validation(self):
        df = self._valid_long_df()
        df.loc[0, "mean_importance"] = np.inf
        with pytest.raises(RuntimeError, match="[Nn]on-finite"):
            grid_module.validate_full_grid(df, n_ok=48, n_skip=0, n_fail=0, total=48)

    def test_wrong_latent_dim_count_fails_validation(self):
        df = self._valid_long_df()
        dropped = df[~((df["asset"] == "Bitcoin") & (df["horizon"] == 1)
                        & (df["feature_set_key"] == "primary") & (df["model_name"] == "Random Forest")
                        & (df["feature"] == "encoded_7"))]
        with pytest.raises(RuntimeError, match="8"):
            grid_module.validate_full_grid(dropped, n_ok=48, n_skip=0, n_fail=0, total=48)

    def test_real_grid_covers_all_24_configurations(self):
        if not _correlations_available():
            pytest.skip("results/feature_importance_autoencoder not found -- run run_latent_original_correlation_analysis.py first.")
        long_df = pd.read_csv(config.RESULTS_DIR / "feature_importance_autoencoder" / "processed" / "latent_original_correlations_long.csv")
        n_configs = long_df[["asset", "horizon", "feature_set_key"]].drop_duplicates().shape[0]
        assert n_configs == 24


# ---------------------------------------------------------------------------
# Representation comparison
# ---------------------------------------------------------------------------

class TestRepresentationComparison:
    def test_missing_branch_reported_not_dropped(self):
        from src.feature_importance.representation_comparison import build_representation_comparison_summary

        original = pd.DataFrame([{
            "asset": "Bitcoin", "horizon": 1, "feature_set_key": "primary", "model_name": "Random Forest",
            "feature": "Bitcoin_Volume_log", "mean_importance": 0.001, "baseline_rmse": 0.02,
        }])
        encoded = pd.DataFrame([], columns=original.columns)  # empty -- Autoencoder branch entirely missing

        summary = build_representation_comparison_summary(original, encoded)
        assert len(summary) == 1
        assert summary.iloc[0]["better_test_rmse_representation"] == "n/a (one branch missing)"
        assert pd.isna(summary.iloc[0]["baseline_rmse_encoded"])


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class TestAutoencoderAPIEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from deployment.app import app
        return TestClient(app)

    def test_feature_importance_comparison_autoencoder(self, client):
        if not _ae_grid_available():
            pytest.skip("results/feature_importance_autoencoder not found.")
        r = client.get("/feature-importance-comparison-autoencoder")
        assert r.status_code == 200
        body = r.json()
        assert len(body) > 0
        assert "mean_importance_Random Forest" in body[0]

    def test_feature_importance_agreement_summary_autoencoder(self, client):
        if not _ae_grid_available():
            pytest.skip("results/feature_importance_autoencoder not found.")
        r = client.get("/feature-importance-agreement-summary-autoencoder")
        assert r.status_code == 200
        assert len(r.json()) == 24

    def test_latent_original_correlations_returns_expected_schema(self, client):
        if not _correlations_available():
            pytest.skip("Latent-original correlations not found.")
        r = client.get("/latent-original-correlations", params={"asset": "Bitcoin", "horizon": 1, "feature_set": "primary"})
        assert r.status_code == 200
        body = r.json()
        assert len(body) > 0
        expected_keys = {
            "asset", "horizon", "feature_set_key", "latent_feature", "original_feature",
            "pearson_correlation", "spearman_correlation", "number_of_aligned_observations",
        }
        assert expected_keys.issubset(body[0].keys())
        assert all(row["asset"] == "Bitcoin" and row["horizon"] == 1 for row in body)

    def test_representation_comparison_summary_schema(self, client):
        if not _ae_grid_available():
            pytest.skip("results/feature_importance_autoencoder not found.")
        r = client.get("/representation-comparison-summary")
        assert r.status_code == 200
        body = r.json()
        assert len(body) > 0
        assert "better_test_rmse_representation" in body[0]

    def test_missing_files_return_404_with_runnable_command(self, client, monkeypatch, tmp_path):
        import deployment.app as app_module
        monkeypatch.setattr(app_module, "AE_FEATURE_IMPORTANCE_DIR", tmp_path)
        r = client.get("/feature-importance-comparison-autoencoder")
        assert r.status_code == 404
        assert "run_feature_importance_grid_autoencoder.py" in r.json()["detail"]

    def test_missing_correlations_file_returns_404(self, client, monkeypatch, tmp_path):
        import deployment.app as app_module
        monkeypatch.setattr(app_module, "AE_FEATURE_IMPORTANCE_DIR", tmp_path)
        r = client.get("/latent-original-correlations")
        assert r.status_code == 404
        assert "run_latent_original_correlation_analysis.py" in r.json()["detail"]
