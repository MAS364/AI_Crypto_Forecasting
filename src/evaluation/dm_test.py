"""Diebold-Mariano test: statistical significance of forecast accuracy
differences between two models -- answers "does Model A actually beat
Model B, or does it just look slightly better by chance" rather than
comparing point RMSE/MAE values alone.

Reference: Diebold, F.X. and Mariano, R.S. (1995). "Comparing Predictive
Accuracy." Journal of Business & Economic Statistics, 13(3), 253-263.
Small-sample correction: Harvey, D., Leybourne, S. and Newbold, P.
(1997). "Testing the equality of prediction mean squared errors."
International Journal of Forecasting, 13(2), 281-291.
"""

import numpy as np
from scipy import stats


def diebold_mariano_test(
    actual, pred1, pred2, h: int = 1, loss: str = "squared", harvey_correction: bool = True
) -> dict:
    """Diebold-Mariano test comparing two models' forecast accuracy.

    H0: both models have equal forecast accuracy. H1: they differ.
    A positive mean_loss_diff means model1's errors are larger on average
    (i.e. model2 is more accurate); negative means the reverse.

    h : forecast horizon in steps ahead -- used for Newey-West HAC
    variance correction. h-step-ahead forecast errors are autocorrelated
    up to lag h-1 by construction (an efficient h-step forecast's errors
    form an MA(h-1) process), so the naive variance estimate understates
    the true variance and overstates significance for h > 1 -- this
    matters directly for this project, which forecasts 1, 7, and 14 days
    ahead, not just 1. Always pass the actual horizon being evaluated,
    not a default of 1, for the 7-day and 14-day comparisons.

    harvey_correction : applies the Harvey, Leybourne & Newbold (1997)
    small-sample correction -- recommended for the sample sizes typical
    of a held-out test period (well under a few hundred observations).

    Returns a dict: dm_stat, p_value, mean_loss_diff, n_obs,
    significant_5pct.
    """
    actual = np.asarray(actual, dtype=float)
    pred1 = np.asarray(pred1, dtype=float)
    pred2 = np.asarray(pred2, dtype=float)
    n = len(actual)
    if not (len(pred1) == len(pred2) == n):
        raise ValueError("actual, pred1, pred2 must be the same length")
    if n < 10:
        raise ValueError(f"Need a reasonable sample size for a meaningful test -- got {n}")

    e1 = actual - pred1
    e2 = actual - pred2

    if loss == "squared":
        g1, g2 = e1 ** 2, e2 ** 2
    elif loss == "absolute":
        g1, g2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    d = g1 - g2
    d_bar = d.mean()

    # Newey-West HAC variance using h-1 lags -- the standard DM choice.
    max_lag = max(h - 1, 0)
    var_d = np.var(d, ddof=0)
    for lag in range(1, max_lag + 1):
        if len(d) > lag:
            cov = np.cov(d[lag:], d[:-lag])[0, 1]
            var_d += 2 * (1 - lag / (max_lag + 1)) * cov
    var_d = max(var_d, 1e-12)  # guard against numerical zero/negative from noisy small samples

    dm_stat = d_bar / np.sqrt(var_d / n)

    if harvey_correction:
        correction = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
        dm_stat *= correction

    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": float(d_bar),
        "n_obs": int(n),
        "significant_5pct": bool(p_value < 0.05),
    }


def compare_all_models_to_baseline(
    predictions_df, baseline_model: str = "Naive", loss: str = "squared",
    actual_col: str = "actual_price", predicted_col: str = "predicted_price",
) -> "pd.DataFrame":
    """Runs the DM test for every (asset, horizon, feature_set,
    representation) combination in a predictions dataframe (matching
    loaders.load_predictions()'s real schema -- actual_price/
    predicted_price for price-space comparison, or pass actual_col=
    "actual_return", predicted_col="predicted_return" for return-space),
    comparing every non-baseline model against `baseline_model` on the
    test split. Returns one row per (asset, horizon, feature_set,
    representation, model) comparison.

    Skips a combination silently (does not raise) if either the
    baseline's or the challenger's predictions are missing for it --
    reports what it could compute rather than failing the whole table
    over one gap, since a fully-populated grid across 2 assets x 3
    horizons x 4 feature sets x 2 representations x 4 non-baseline
    models is 192 combinations, and not every one may be equally
    complete.
    """
    import pandas as pd

    test_df = predictions_df[predictions_df["split"] == "test"]
    rows = []

    group_cols = ["asset", "horizon", "feature_set", "representation"]
    for group_key, group in test_df.groupby(group_cols):
        asset, horizon, feature_set, representation = group_key
        baseline_rows = group[group["model"] == baseline_model].sort_values("event_date")
        if baseline_rows.empty:
            continue

        for model_name in group["model"].unique():
            if model_name == baseline_model:
                continue
            model_rows = group[group["model"] == model_name].sort_values("event_date")

            merged = baseline_rows.merge(
                model_rows, on="event_date", suffixes=("_baseline", "_model")
            )
            if len(merged) < 10:
                continue

            result = diebold_mariano_test(
                actual=merged[f"{actual_col}_baseline"],
                pred1=merged[f"{predicted_col}_baseline"],
                pred2=merged[f"{predicted_col}_model"],
                h=int(horizon), loss=loss,
            )
            rows.append({
                "asset": asset, "horizon": horizon, "feature_set": feature_set,
                "representation": representation, "model": model_name,
                "baseline": baseline_model, **result,
            })

    return pd.DataFrame(rows)
