"""Tests for src/evaluation/metrics.py -- RMSE, MAE, R2, DirAcc/Hit Rate,
price reconstruction, and the combined evaluate() function.

Run: pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest

from src.evaluation.metrics import rmse, mae, mape, dir_acc, r_squared, to_price, evaluate


class TestRMSE:
    def test_zero_error_gives_zero_rmse(self):
        a = np.array([0.01, -0.02, 0.03])
        assert rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        # errors of [1, -1] -> RMSE = 1
        assert rmse([0, 0], [1, -1]) == pytest.approx(1.0)

    def test_matches_manual_calculation(self):
        a = np.array([0.05, -0.03, 0.02, 0.01])
        b = np.array([0.04, -0.01, 0.00, 0.02])
        expected = np.sqrt(np.mean((a - b) ** 2))
        assert rmse(a, b) == pytest.approx(expected)


class TestMAE:
    def test_zero_error_gives_zero_mae(self):
        a = np.array([0.01, -0.02, 0.03])
        assert mae(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        assert mae([0, 0], [1, -1]) == pytest.approx(1.0)


class TestRSquared:
    def test_perfect_prediction_gives_r2_one(self):
        a = np.array([0.05, -0.03, 0.02, 0.01, -0.04])
        assert r_squared(a, a) == pytest.approx(1.0)

    def test_predicting_the_mean_gives_r2_zero(self):
        a = np.array([0.05, -0.03, 0.02, 0.01, -0.04])
        pred = np.full_like(a, np.mean(a))
        assert r_squared(a, pred) == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean_gives_negative_r2(self):
        a = np.array([0.05, -0.03, 0.02, 0.01, -0.04])
        # deliberately bad predictions, far from actual
        pred = -a * 5
        assert r_squared(a, pred) < 0

    def test_matches_sklearn(self):
        sklearn = pytest.importorskip("sklearn.metrics")
        rng = np.random.RandomState(42)
        a = rng.randn(100) * 0.02
        b = a + rng.randn(100) * 0.015
        assert r_squared(a, b) == pytest.approx(sklearn.r2_score(a, b))


class TestDirAccAndHitRate:
    def test_all_correct_direction_gives_100(self):
        actual = np.array([0.01, -0.02, 0.03, -0.01])
        predicted = np.array([0.05, -0.01, 0.02, -0.03])  # same signs
        assert dir_acc(actual, predicted) == pytest.approx(100.0)

    def test_all_wrong_direction_gives_0(self):
        actual = np.array([0.01, -0.02, 0.03, -0.01])
        predicted = np.array([-0.05, 0.01, -0.02, 0.03])  # opposite signs
        assert dir_acc(actual, predicted) == pytest.approx(0.0)

    def test_half_correct_gives_50(self):
        actual = np.array([0.01, -0.02, 0.03, -0.01])
        predicted = np.array([0.05, 0.01, 0.02, 0.03])  # 2 of 4 match sign
        assert dir_acc(actual, predicted) == pytest.approx(50.0)


class TestPriceReconstruction:
    def test_zero_return_gives_unchanged_price(self):
        level = np.array([50000.0, 3000.0])
        ret = np.array([0.0, 0.0])
        result = to_price(level, ret)
        np.testing.assert_allclose(result, level)

    def test_known_return_matches_manual_calc(self):
        level = np.array([50000.0])
        ret = np.array([0.01])  # ~1% up
        expected = 50000.0 * np.exp(0.01)
        assert to_price(level, ret)[0] == pytest.approx(expected)

    def test_reconstruction_is_invertible(self):
        """log(price_t+1/price_t) reconstruction should invert the return definition exactly."""
        price_t = 45000.0
        price_t1 = 46500.0
        log_return = np.log(price_t1 / price_t)
        reconstructed = to_price(np.array([price_t]), np.array([log_return]))[0]
        assert reconstructed == pytest.approx(price_t1)


class TestEvaluateIntegration:
    """evaluate() is the single function every model in the project calls --
    these tests check its full output schema and internal consistency."""

    def test_returns_all_required_keys(self):
        rng = np.random.RandomState(0)
        level = rng.uniform(20000, 60000, 30)
        y_true = rng.randn(30) * 0.02
        y_pred = y_true + rng.randn(30) * 0.01
        result = evaluate(level, y_true, y_pred)

        required_keys = {
            "RMSE_returns", "MAE_returns", "R2_returns",
            "DirAcc_%", "Hit_Rate_%", "RMSE_price", "MAE_price", "MAPE_price_%",
        }
        assert required_keys.issubset(result.keys())

    def test_hit_rate_equals_dir_acc(self):
        """Hit_Rate_% and DirAcc_% must always be identical -- same underlying
        quantity, stored under both names (see src/evaluation/metrics.py docstring)."""
        rng = np.random.RandomState(1)
        level = rng.uniform(20000, 60000, 30)
        y_true = rng.randn(30) * 0.02
        y_pred = y_true + rng.randn(30) * 0.01
        result = evaluate(level, y_true, y_pred)
        assert result["DirAcc_%"] == result["Hit_Rate_%"]

    def test_perfect_predictions_give_zero_error_metrics(self):
        level = np.array([50000.0, 51000.0, 49000.0])
        y_true = np.array([0.01, -0.02, 0.015])
        result = evaluate(level, y_true, y_true)  # predict = actual
        assert result["RMSE_returns"] == pytest.approx(0.0, abs=1e-10)
        assert result["MAE_returns"] == pytest.approx(0.0, abs=1e-10)
        assert result["DirAcc_%"] == pytest.approx(100.0)
        assert result["R2_returns"] == pytest.approx(1.0)
