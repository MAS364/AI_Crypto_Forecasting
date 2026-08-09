"""Evaluation metrics for comparing forecasting models.

Extracted from 07_modeling.ipynb, where these functions were originally
defined and tested against real predictions from all five models (Naive,
ARIMA, ARIMAX, Random Forest, XGBoost, LSTM). The notebook now imports
from this module instead of defining these functions inline, matching
the pattern already used for the feature engineering stage
(src/analysis/feature_engineering.py backing 06_feature_engineering.ipynb).

Every model in this project is scored through the same evaluate()
function, so no model gets a bespoke evaluation path that could
advantage it unfairly.
"""

import numpy as np


def rmse(a, b) -> float:
    """Root mean squared error."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b) -> float:
    """Mean absolute error."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.abs(a - b)))


def mape(a, b) -> float:
    """Mean absolute percentage error.

    Rows where `a` is exactly zero are dropped to avoid division by
    zero. This is why MAPE is used on reconstructed prices (never zero)
    rather than on returns (frequently close to zero, where percentage
    error becomes unstable or meaningless).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = a != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - b[mask]) / a[mask])) * 100)


def dir_acc(a, b) -> float:
    """Directional accuracy / Hit Rate: percentage of days where the predicted
    sign matches the actual sign. "Hit Rate" is the same quantity under a
    different name -- both are reported wherever this metric is displayed.

    Computed on returns, not price, since "up or down" is a return-space
    question.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.mean(np.sign(a) == np.sign(b)) * 100)


def r_squared(a, b) -> float:
    """Coefficient of determination, R^2 = 1 - SS_res/SS_tot, matching
    sklearn.metrics.r2_score exactly (verified). Can legitimately be
    negative on noisy daily/weekly return data -- this means the model
    performs worse than simply predicting the historical mean, and is
    reported as-is rather than floored at zero.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def to_price(level, ret):
    """Reconstruct a price from a known price level and a (true or predicted) log return.

    price_t+1 = price_level_t * exp(log_return_t+1)

    `level` is always the real, known price on the day the forecast is
    made from -- it is never itself predicted.
    """
    return np.asarray(level, float) * np.exp(np.asarray(ret, float))


def evaluate(level, y_true, y_pred) -> dict:
    """Full evaluation of one model's predictions: return-space and reconstructed price-space.

    This is the single function every model in this project calls to
    get its results, so every model is scored identically. `DirAcc_%`
    and `Hit_Rate_%` are the same quantity computed once (`dir_acc`) and
    stored under both names -- Hit_Rate_% was previously only created by
    a notebook-level rename (`df["Hit_Rate_%"] = df["DirAcc_%"]`), which
    meant anyone calling this function directly (e.g. a production
    inference or evaluation pipeline, without the notebook) never saw a
    column literally named Hit_Rate_%. Both names are now always present
    directly from evaluate() itself.
    """
    true_price = to_price(level, y_true)
    pred_price = to_price(level, y_pred)
    directional_accuracy = dir_acc(y_true, y_pred)
    return {
        "RMSE_returns": rmse(y_true, y_pred),
        "MAE_returns": mae(y_true, y_pred),
        "R2_returns": r_squared(y_true, y_pred),
        "DirAcc_%": directional_accuracy,
        "Hit_Rate_%": directional_accuracy,
        "RMSE_price": rmse(true_price, pred_price),
        "MAE_price": mae(true_price, pred_price),
        "MAPE_price_%": mape(true_price, pred_price),
    }
