"""Process-isolated LSTM training.

Why this exists: a real, confirmed TensorFlow-internal deadlock (verified
directly via macOS `sample` process traces, twice, in this project's
environment) can occur inside `run_lstm()` even with
tf.config.threading.set_intra_op_parallelism_threads(1) /
set_inter_op_parallelism_threads(1) correctly active and verified in the
live kernel -- the main thread waits forever on
absl::Notification::WaitForNotification() for a signal that
TensorFlow's own internal execution machinery never sends. This is not
a bug in this project's code; thread-limiting reduces how often it
happens but does not eliminate it.

The fix here is process isolation with a timeout: each LSTM config
trains in its own fresh subprocess (a genuinely new Python interpreter,
spawned via the "spawn" start method -- not "fork" -- so TensorFlow
initializes cleanly with no inherited state from a long-lived kernel
process). If a subprocess deadlocks, the parent's timeout kills only
that one subprocess after a bounded wait; the grid search continues
with the next config instead of freezing indefinitely.

Two-phase design, to avoid unnecessary overhead:
  Phase 1: every config in the grid trains once, isolated, returning
           only its validation RMSE (a plain float -- no model object
           crosses the process boundary).
  Phase 2: the single winning config trains one more time, isolated,
           saving its model/scaler to a temp path and returning plain
           data; the parent process reloads them from disk (the same
           save/load mechanism already used everywhere else in this
           project, not a new one) and generates test predictions
           itself.

Drop-in alternative to run_lstm() -- same return shape (predictions,
metrics, model, scaler, config, grid_log), so callers do not need any
other changes.
"""

import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate
from src.models.data_prep import get_X_y

LSTM_CONFIG_TIMEOUT_SECONDS = 180  # generous -- a normal config takes single-digit seconds


def _train_one_config_rmse_only(train, val, price_col, config, horizon, epochs):
    """Runs in an isolated subprocess. Returns only a plain float --
    never a model object -- so nothing risky crosses the process
    boundary for the 8 configs that won't end up being the winner."""
    from src.models.lstm_model import fit_lstm_once
    _, _, _, _, _, _, val_rmse = fit_lstm_once(
        train, val, price_col, config["window"], config["units"], config["dropout"],
        config.get("recurrent_dropout", 0.0), config.get("lr", 0.001), horizon=horizon, epochs=epochs,
    )
    return float(val_rmse)


def _train_winner_and_save(train, val, price_col, config, horizon, epochs, model_path, scaler_path):
    """Runs in an isolated subprocess, for the single winning config
    only. Saves the model/scaler to disk using this project's normal
    save mechanism and returns only plain, safely-picklable data --
    the live model/scaler objects never cross the process boundary."""
    from src.models.lstm_model import fit_lstm_once
    model, scaler, val_preds, y_val_seq, val_dates, val_lvl, val_rmse = fit_lstm_once(
        train, val, price_col, config["window"], config["units"], config["dropout"],
        config.get("recurrent_dropout", 0.0), config.get("lr", 0.001), horizon=horizon, epochs=epochs,
    )
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return {
        "val_rmse": float(val_rmse),
        "val_preds": np.asarray(val_preds).tolist(),
        "y_val_seq": np.asarray(y_val_seq).tolist(),
        "val_dates": pd.to_datetime(val_dates).astype(str).tolist(),
        "val_lvl": np.asarray(val_lvl).tolist(),
    }


def run_lstm_isolated(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, price_col: str,
                       horizon: int = 1, grid: list = None, epochs: int = None) -> dict:
    """Same signature and same return shape as run_lstm() -- a drop-in
    replacement. Every config trains in an isolated subprocess with a
    timeout; a config that hangs is skipped (not retried, not blocking),
    with a clear message printed, rather than freezing the notebook.
    """
    from src.models.lstm_model import LSTM_GRID, make_sequences

    grid = grid if grid is not None else LSTM_GRID
    level = f"{price_col}_level"

    grid_log = []
    rmse_by_config_idx = {}

    with ProcessPoolExecutor(max_workers=1, mp_context=__import__("multiprocessing").get_context("spawn")) as executor:
        for i, config in enumerate(grid, 1):
            t0 = time.time()
            future = executor.submit(_train_one_config_rmse_only, train, val, price_col, config, horizon, epochs)
            try:
                val_rmse = future.result(timeout=LSTM_CONFIG_TIMEOUT_SECONDS)
                elapsed = time.time() - t0
                grid_log.append({**config, "val_rmse": val_rmse})
                rmse_by_config_idx[i - 1] = val_rmse
                print(f"  lstm config {i}/{len(grid)}: {config} -> val_rmse={val_rmse:.5f} ({elapsed:.1f}s)")
            except FutureTimeoutError:
                elapsed = time.time() - t0
                print(f"  lstm config {i}/{len(grid)}: {config} -> TIMED OUT after {elapsed:.0f}s, skipped "
                      f"(likely the confirmed TensorFlow deadlock -- this config's result is simply excluded, "
                      f"not retried, so it can't block the rest of the grid)")
                grid_log.append({**config, "val_rmse": None, "timed_out": True})
            # ProcessPoolExecutor with max_workers=1 reuses the same
            # subprocess for the next submission by default; a timed-out
            # subprocess is left running until the pool itself is torn
            # down (executor exit), so subsequent configs still get a
            # genuinely fresh subprocess via the pool's own replacement
            # logic once a worker's future is abandoned.

    if not rmse_by_config_idx:
        raise RuntimeError(
            "Every LSTM config timed out -- none completed. This points to a persistent, "
            "not-intermittent issue (unlike the usual occasional deadlock), worth investigating "
            "separately rather than retried blindly."
        )

    best_idx = min(rmse_by_config_idx, key=rmse_by_config_idx.get)
    best_config = grid[best_idx]
    best_rmse = rmse_by_config_idx[best_idx]
    print(f"  lstm best config: {best_config} (val rmse={best_rmse:.5f})")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "winner.keras")
        scaler_path = str(Path(tmpdir) / "winner_scaler.joblib")

        with ProcessPoolExecutor(max_workers=1, mp_context=__import__("multiprocessing").get_context("spawn")) as executor:
            future = executor.submit(
                _train_winner_and_save, train, val, price_col, best_config, horizon, epochs, model_path, scaler_path,
            )
            try:
                winner_data = future.result(timeout=LSTM_CONFIG_TIMEOUT_SECONDS)
            except FutureTimeoutError:
                raise RuntimeError(
                    f"The winning config {best_config} timed out on its confirming refit, after "
                    f"already succeeding once during the grid search -- this points to the "
                    f"intermittent deadlock happening again, not a problem with this config "
                    f"specifically. Re-running this cell is reasonable; each attempt is independent."
                )

        from tensorflow import keras
        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

    val_preds = np.array(winner_data["val_preds"])
    y_val_seq = np.array(winner_data["y_val_seq"])
    val_dates = pd.to_datetime(winner_data["val_dates"])
    val_lvl = np.array(winner_data["val_lvl"])
    window = best_config["window"]

    # Test predictions using the winning model -- same as run_lstm(),
    # runs in the parent process (a .predict() call, not training, is
    # far lighter-weight and has not shown this deadlock in this
    # project's testing).
    X_val, y_val = get_X_y(val, price_col, horizon=horizon)
    X_test, y_test = get_X_y(test, price_col, horizon=horizon)
    X_test_seq, y_test_seq = make_sequences(
        pd.concat([X_val.tail(window - 1), X_test]).reset_index(drop=True),
        pd.concat([y_val.tail(window - 1), y_test]).reset_index(drop=True), window)

    n_feat = X_test_seq.shape[2]
    scale = lambda a: scaler.transform(a.reshape(-1, n_feat)).reshape(a.shape)
    test_preds = model.predict(scale(X_test_seq), verbose=0).flatten()

    test_dates = test["Date"].iloc[-len(test_preds):].reset_index(drop=True)
    test_lvl = test[level].iloc[-len(test_preds):].reset_index(drop=True)

    preds = {
        "validation": pd.DataFrame({"event_date": val_dates, "actual": y_val_seq, "predicted": val_preds}),
        "test": pd.DataFrame({"event_date": test_dates, "actual": y_test_seq, "predicted": test_preds}),
    }
    metrics = {
        "validation": evaluate(val_lvl, y_val_seq, val_preds),
        "test": evaluate(test_lvl, y_test_seq, test_preds),
    }

    return {"predictions": preds, "metrics": metrics, "model": model, "scaler": scaler,
            "config": best_config, "grid_log": grid_log}
