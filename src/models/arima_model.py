"""ARIMA statistical model.

Extracted from 07_modeling.ipynb. ARIMA is univariate on the log-return
series only -- the full feature matrix used by Random Forest, XGBoost,
and LSTM is never given to it. ARIMAX was explored earlier in this
project and explicitly removed at the supervisor's request; this
module was never re-introduced to keep both layers using identical
models.

The model is fit once, then forecasts one day at a time, updating
their history with the real value after each day (via `.append(...,
refit=False)`) rather than producing one long-range forecast. This
keeps the comparison fair against Random Forest and XGBoost, whose lag
features always give them genuine recent history at every row -- a
single long-range ARIMA forecast would drift toward the series mean
almost immediately and would not be a fair comparison.

`.append(refit=False)` extends the model's forecasting state without
re-estimating its MLE parameters on every new observation -- this is
standard, accepted rolling-forecast practice, not a shortcut.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.evaluation.metrics import evaluate


def pick_arima_order(series: pd.Series, max_p: int = 3, max_q: int = 3) -> tuple:
    """Pick ARIMA(p, 0, q) by lowest AIC on the training series. d=0: already stationary."""
    best_aic, best_order = np.inf, (1, 0, 0)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # maxiter bounds the optimizer explicitly -- without
                    # it, a difficult series/order combination can run
                    # for a very long time with no guarantee of
                    # convergence (confirmed directly: a real process
                    # sample showed this exact call stuck inside the
                    # Kalman filter's likelihood optimization for
                    # several minutes with the default, unbounded
                    # setting). 50 iterations is generous for a
                    # low-order ARIMA on a daily return series; if the
                    # optimizer hasn't converged by then, this order is
                    # skipped, same as any other non-converging order.
                    m = ARIMA(series, order=(p, 0, q)).fit(method_kwargs={"maxiter": 50})
                if m.aic < best_aic:
                    best_aic, best_order = m.aic, (p, 0, q)
            except Exception:
                pass  # some orders just don't converge, skip them
    return best_order


def arima_rolling(history: pd.Series, daily_actuals: pd.Series, order: tuple, horizon: int = 1) -> pd.Series:
    """Fit once, then forecast `horizon` steps ahead at every row, updating history one real
    day at a time. At horizon > 1, each row's forecast is `horizon` individual daily-return
    forecasts summed into one cumulative prediction (log returns are additive), while the
    walk-forward step itself still only advances by ONE real day, keeping ARIMA on the same
    daily update cadence as Random Forest, XGBoost, and LSTM, whose lag features are
    refreshed every single day regardless of the target horizon."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Same bound as pick_arima_order() -- this is the exact call a
        # real process sample caught stuck inside BLAS/Kalman filter
        # optimization with no time bound.
        model = ARIMA(history, order=order).fit(method_kwargs={"maxiter": 50})

    preds = []
    for actual_next_day in daily_actuals:
        forecast_steps = model.forecast(steps=horizon)
        cumulative_forecast = forecast_steps.sum() if horizon > 1 else forecast_steps.iloc[0]
        preds.append(cumulative_forecast)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model.append([actual_next_day], refit=False)

    return pd.Series(preds, index=daily_actuals.index)


def run_arima(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, price_col: str, horizon: int = 1) -> dict:
    ret = f"{price_col}_log_return"
    target = f"{price_col}_log_return_target_t_plus_{horizon}"
    level = f"{price_col}_level"

    order = pick_arima_order(train[ret])
    print(f"  arima order: {order}")

    history = train[ret].copy()
    preds, metrics = {}, {}

    for name, split in [("validation", val), ("test", test)]:
        # daily_actuals (single-day returns) advances history one real day at a time;
        # target (cumulative horizon-day return) is what the summed forecast is scored against.
        p = arima_rolling(history, split[ret], order, horizon=horizon)
        metrics[name] = evaluate(split[level], split[target], p)
        preds[name] = pd.DataFrame({"event_date": split["Date"], "actual": split[target], "predicted": p.values})
        history = pd.concat([history, split[ret]], ignore_index=True)

    return {"predictions": preds, "metrics": metrics, "order": order}
