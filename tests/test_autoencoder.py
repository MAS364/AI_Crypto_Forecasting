"""Tests for src/models/autoencoder.py -- training, encoding, save/load,
and the true inference-time entry point (predict_encode).

Run: pytest tests/test_autoencoder.py -v
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.autoencoder import (
    build_autoencoder, train_autoencoder, encode, reconstruction_loss,
    save_autoencoder, load_autoencoder, predict_encode,
)

TEST_SAVE_DIR = Path("/tmp/test_autoencoder_artifacts")


def _synthetic_features(n_rows=300, n_features=14, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(rng.randn(n_rows, n_features), columns=[f"feature_{i}" for i in range(n_features)])


@pytest.fixture(scope="module")
def trained_autoencoder():
    X_train = _synthetic_features(300)
    X_val = _synthetic_features(80, seed=1)
    autoencoder, encoder, scaler, history = train_autoencoder(
        X_train, X_val, encoding_dim=4, epochs=10, patience=5, seed=42,
    )
    return autoencoder, encoder, scaler, history, X_train, X_val


class TestBuildAutoencoder:
    def test_encoder_output_dim_matches_requested(self):
        autoencoder, encoder = build_autoencoder(input_dim=14, encoding_dim=6)
        assert encoder.output_shape[-1] == 6

    def test_autoencoder_input_and_output_dims_match(self):
        """A true autoencoder reconstructs its input -- input and output
        dimensions must be identical."""
        autoencoder, encoder = build_autoencoder(input_dim=20, encoding_dim=8)
        assert autoencoder.input_shape[-1] == autoencoder.output_shape[-1] == 20


class TestTrainAutoencoder:
    def test_training_produces_expected_encoding_dimension(self, trained_autoencoder):
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        assert encoder.output_shape[-1] == 4

    def test_training_history_has_loss_and_val_loss(self, trained_autoencoder):
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        assert "loss" in history.history
        assert "val_loss" in history.history
        assert len(history.history["loss"]) > 0

    def test_scaler_fit_only_on_training_data(self, trained_autoencoder):
        """Leakage check: the scaler's learned mean must match X_train's mean,
        not some blend of train and validation data."""
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        np.testing.assert_allclose(scaler.mean_, X_train.mean().values, atol=1e-8)

    def test_reproducible_with_same_seed(self):
        """Same seed, same data -> identical encoded output. This is the
        project's core reproducibility guarantee for the autoencoder branch."""
        X_train = _synthetic_features(300)
        X_val = _synthetic_features(80, seed=1)

        _, encoder1, scaler1, _ = train_autoencoder(X_train, X_val, encoding_dim=4, epochs=5, patience=3, seed=42)
        _, encoder2, scaler2, _ = train_autoencoder(X_train, X_val, encoding_dim=4, epochs=5, patience=3, seed=42)

        out1 = encode(encoder1, scaler1, X_train.iloc[:5])
        out2 = encode(encoder2, scaler2, X_train.iloc[:5])
        np.testing.assert_array_equal(out1, out2)


class TestEncodeAndReconstructionLoss:
    def test_encode_output_shape(self, trained_autoencoder):
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        X_test = _synthetic_features(50, seed=2)
        encoded = encode(encoder, scaler, X_test)
        assert encoded.shape == (50, 4)

    def test_reconstruction_loss_is_non_negative(self, trained_autoencoder):
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        X_test = _synthetic_features(50, seed=2)
        loss = reconstruction_loss(autoencoder, scaler, X_test)
        assert loss >= 0

    def test_reconstruction_loss_does_not_refit_scaler(self, trained_autoencoder):
        """Calling reconstruction_loss on test data must not change the
        scaler's learned parameters -- it should only transform, never fit."""
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        mean_before = scaler.mean_.copy()
        X_test = _synthetic_features(50, seed=2)
        reconstruction_loss(autoencoder, scaler, X_test)
        np.testing.assert_array_equal(scaler.mean_, mean_before)


class TestSaveAndLoad:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        if TEST_SAVE_DIR.exists():
            shutil.rmtree(TEST_SAVE_DIR)

    def test_save_creates_all_expected_files(self, trained_autoencoder):
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        paths = save_autoencoder(autoencoder, encoder, scaler, history, TEST_SAVE_DIR / "test_prefix")
        for key in ["autoencoder", "encoder", "scaler", "history", "metadata"]:
            assert Path(paths[key]).exists(), f"{key} was not saved"

    def test_metadata_contains_required_fields(self, trained_autoencoder):
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        save_autoencoder(autoencoder, encoder, scaler, history, TEST_SAVE_DIR / "test_prefix")
        loaded = load_autoencoder(TEST_SAVE_DIR / "test_prefix")
        assert loaded["metadata"]["encoding_dim"] == 4
        assert loaded["metadata"]["input_dim"] == 14
        assert "epochs_trained" in loaded["metadata"]

    def test_load_missing_prefix_raises_clear_error(self):
        with pytest.raises(FileNotFoundError):
            load_autoencoder(TEST_SAVE_DIR / "nonexistent_prefix")

    def test_loaded_encoder_produces_identical_output_to_original(self, trained_autoencoder):
        """The whole point of save/load: a freshly loaded encoder must
        produce exactly the same output as the one still in memory."""
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        save_autoencoder(autoencoder, encoder, scaler, history, TEST_SAVE_DIR / "test_prefix")

        X_test = _synthetic_features(20, seed=3)
        original_output = encode(encoder, scaler, X_test)

        loaded = load_autoencoder(TEST_SAVE_DIR / "test_prefix")
        loaded_output = encode(loaded["encoder"], loaded["scaler"], X_test)

        np.testing.assert_allclose(original_output, loaded_output, atol=1e-5)


class TestPredictEncodeInference:
    """predict_encode() is the true production inference entry point --
    these tests specifically verify it never retrains anything."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        if TEST_SAVE_DIR.exists():
            shutil.rmtree(TEST_SAVE_DIR)

    def test_predict_encode_matches_direct_encode(self, trained_autoencoder):
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        save_autoencoder(autoencoder, encoder, scaler, history, TEST_SAVE_DIR / "test_prefix")

        X_test = _synthetic_features(20, seed=4)
        direct = encode(encoder, scaler, X_test)
        via_inference = predict_encode(TEST_SAVE_DIR / "test_prefix", X_test)
        np.testing.assert_allclose(direct, via_inference, atol=1e-5)

    def test_predict_encode_rejects_wrong_column_count(self, trained_autoencoder):
        """A production inference pipeline must fail loudly, not silently,
        if the input doesn't match what the saved encoder expects."""
        autoencoder, encoder, scaler, history, X_train, X_val = trained_autoencoder
        save_autoencoder(autoencoder, encoder, scaler, history, TEST_SAVE_DIR / "test_prefix")

        wrong_shape_input = _synthetic_features(20, n_features=5, seed=5)
        with pytest.raises(ValueError, match="columns"):
            predict_encode(TEST_SAVE_DIR / "test_prefix", wrong_shape_input)
