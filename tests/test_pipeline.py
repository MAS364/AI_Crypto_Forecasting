"""Tests for src/pipeline.py -- the unified train/evaluate/predict entry
point. Skips gracefully where real saved artefacts aren't present.

Run:
    pytest tests/test_pipeline.py -v
"""

import pandas as pd
import pytest

from src import config
from src import pipeline as p


def _real_feature_set_available():
    """Return True when the real Bitcoin market-only feature set exists."""
    return (
        config.FEATURES_DIR / "bitcoin_market_only_features.csv"
    ).exists()


def _saved_predictions_available():
    """Return True when Notebook 07/model training predictions exist."""
    return (
        config.RESULTS_DIR
        / "modeling"
        / "predictions"
        / "all_model_predictions.csv"
    ).exists()


class TestRunEvaluate:
    """run_evaluate() must never train or predict anything -- only load
    already-saved results and build comparison tables.
    """

    def test_returns_expected_keys(self):
        if not _saved_predictions_available():
            pytest.skip(
                "No saved predictions found -- run Notebook 07 "
                "(or src.pipeline train) first."
            )

        result = p.run_evaluate()

        assert set(result.keys()) == {
            "master_comparison",
            "validation_summary",
            "checks",
        }

    def test_all_validation_checks_pass_on_real_data(self):
        if not _saved_predictions_available():
            pytest.skip("No saved predictions found.")

        result = p.run_evaluate()

        assert all(result["checks"].values()), (
            f"Some validation checks failed: {result['checks']}"
        )

    def test_saves_comparison_tables_to_disk(self):
        if not _saved_predictions_available():
            pytest.skip("No saved predictions found.")

        p.run_evaluate()

        assert (
            p.EVALUATION_DIR / "master_comparison.csv"
        ).exists()

        assert (
            p.EVALUATION_DIR / "validation_summary.csv"
        ).exists()

    def test_raises_clear_error_on_missing_data(
        self,
        monkeypatch,
        tmp_path,
    ):
        """Point loaders at an empty directory and confirm evaluation
        fails loudly rather than silently producing an empty report.
        """
        from src.evaluation import loaders

        monkeypatch.setattr(
            loaders,
            "MODELING_DIR",
            tmp_path / "does_not_exist",
        )

        with pytest.raises(FileNotFoundError):
            p.run_evaluate()


class TestBuildFeaturesIfMissing:
    """The skip logic is the whole point of this function -- it must
    never rebuild a feature file that's already on disk.
    """

    def test_skips_existing_files_without_touching_them(
        self,
        tmp_path,
        monkeypatch,
    ):
        fake_features_dir = tmp_path / "features"
        fake_features_dir.mkdir()

        fake_file = (
            fake_features_dir
            / "bitcoin_market_only_features.csv"
        )

        fake_file.write_text(
            "Date,Bitcoin_Price\n"
            "2021-01-01,100\n"
        )

        original_mtime = fake_file.stat().st_mtime

        monkeypatch.setattr(
            p.config,
            "FEATURES_DIR",
            fake_features_dir,
        )

        monkeypatch.setattr(
            p.config,
            "MERGED_FILE",
            tmp_path / "merged.csv",
        )

        (
            tmp_path / "merged.csv"
        ).write_text(
            "Date\n2021-01-01\n"
        )

        monkeypatch.setattr(
            p,
            "FEATURE_SET_PARAMS",
            {
                "primary": (
                    False,
                    False,
                )
            },
        )

        p._build_features_if_missing(
            horizons=[1],
            assets=["Bitcoin"],
        )

        assert (
            fake_file.stat().st_mtime
            == original_mtime
        ), (
            "Existing feature file was touched/rebuilt "
            "when it should have been skipped"
        )

    def test_raises_clear_error_if_merged_dataset_missing(
        self,
        tmp_path,
        monkeypatch,
    ):
        monkeypatch.setattr(
            p.config,
            "MERGED_FILE",
            tmp_path / "does_not_exist.csv",
        )

        monkeypatch.setattr(
            p.config,
            "FEATURES_DIR",
            tmp_path / "features",
        )

        with pytest.raises(
            FileNotFoundError,
            match="run 03_data_preparation",
        ):
            p._build_features_if_missing(
                horizons=[1],
                assets=["Bitcoin"],
            )


class TestRunPredict:
    """run_predict() must never retrain.

    These tests exercise Original, Autoencoder-Encoded, and LSTM
    inference using already-saved artefacts.
    """

    @pytest.fixture(scope="class")
    def sample_input(self):
        if not _real_feature_set_available():
            pytest.skip(
                "Real feature set not found -- "
                "run Notebook 06 first."
            )

        df = pd.read_csv(
            config.FEATURES_DIR
            / "bitcoin_market_only_features.csv",
            parse_dates=["Date"],
        )

        # Enough chronological history for LSTM sequence construction.
        return df.tail(100)

    def test_predict_with_original_representation(
        self,
        sample_input,
    ):
        result = p.run_predict(
            horizon=1,
            asset="Bitcoin",
            feature_set_key="primary",
            model_name="Random Forest",
            representation="Original",
            input_df=sample_input,
        )

        assert set(result.keys()) == {
            "horizon",
            "asset",
            "model",
            "feature_set",
            "representation",
            "input_price_level",
            "predicted_return",
            "predicted_price",
        }

        assert isinstance(
            result["predicted_return"],
            float,
        )

        assert result["predicted_price"] > 0

    def test_predict_with_autoencoder_encoded_representation(
        self,
        sample_input,
    ):
        """Autoencoder-Encoded inference should use the saved encoded
        model artefact produced by Notebook 07.

        src.pipeline.run_predict() loads encoded models from:
            results/modeling/fitted_models/
        """

        expected_path = (
            config.RESULTS_DIR
            / "modeling"
            / "fitted_models"
            / "bitcoin_primary_xgboost_encoded.joblib"
        )

        if not expected_path.exists():
            pytest.skip(
                f"{expected_path} not found -- "
                "run Notebook 07 to generate the "
                "Autoencoder-Encoded fitted model artefacts."
            )

        result = p.run_predict(
            horizon=1,
            asset="Bitcoin",
            feature_set_key="primary",
            model_name="XGBoost",
            representation="Autoencoder-Encoded",
            input_df=sample_input,
        )

        assert (
            result["representation"]
            == "Autoencoder-Encoded"
        )

        assert result["model"] == "XGBoost"

        assert isinstance(
            result["predicted_return"],
            float,
        )

        assert isinstance(
            result["predicted_price"],
            float,
        )

        assert result["predicted_price"] > 0

    def test_price_reconstruction_is_consistent(
        self,
        sample_input,
    ):
        """predicted_price must equal:

        input_price_level * exp(predicted_return)
        """
        import numpy as np

        result = p.run_predict(
            horizon=1,
            asset="Bitcoin",
            feature_set_key="primary",
            model_name="Random Forest",
            representation="Original",
            input_df=sample_input,
        )

        expected_price = (
            result["input_price_level"]
            * np.exp(result["predicted_return"])
        )

        assert result[
            "predicted_price"
        ] == pytest.approx(
            expected_price,
            rel=1e-9,
        )

    def test_arima_raises_clear_error(
        self,
        sample_input,
    ):
        with pytest.raises(
            ValueError,
            match="ARIMA",
        ):
            p.run_predict(
                horizon=1,
                asset="Bitcoin",
                feature_set_key="primary",
                model_name="ARIMA",
                representation="Original",
                input_df=sample_input,
            )

    def test_naive_raises_clear_error(
        self,
        sample_input,
    ):
        with pytest.raises(
            ValueError,
            match="Naive",
        ):
            p.run_predict(
                horizon=1,
                asset="Bitcoin",
                feature_set_key="primary",
                model_name="Naive",
                representation="Original",
                input_df=sample_input,
            )

    def test_lstm_original_inference(
        self,
        sample_input,
    ):
        """LSTM Original inference should use an already-saved model
        and return a valid prediction without retraining.
        """

        result = p.run_predict(
            horizon=1,
            asset="Bitcoin",
            feature_set_key="primary",
            model_name="LSTM",
            representation="Original",
            input_df=sample_input,
        )

        assert result["model"] == "LSTM"

        assert (
            result["representation"]
            == "Original"
        )

        assert isinstance(
            result["predicted_return"],
            float,
        )

        assert isinstance(
            result["predicted_price"],
            float,
        )

        assert result["predicted_price"] > 0

    def test_missing_level_column_raises_clear_error(
        self,
    ):
        bad_input = pd.DataFrame(
            {
                "some_feature": [
                    1,
                    2,
                    3,
                ]
            }
        )

        with pytest.raises(
            ValueError,
            match="level",
        ):
            p.run_predict(
                horizon=1,
                asset="Bitcoin",
                feature_set_key="primary",
                model_name="Random Forest",
                representation="Original",
                input_df=bad_input,
            )

    def test_missing_saved_model_raises_clear_error(
        self,
        sample_input,
    ):
        with pytest.raises(FileNotFoundError):
            p.run_predict(
                horizon=14,
                asset="Bitcoin",
                feature_set_key="nonexistent_key",
                model_name="Random Forest",
                representation="Original",
                input_df=sample_input,
            )