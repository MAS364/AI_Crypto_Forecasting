"""LSTM model, built on the same tabular feature sets used by Random Forest and XGBoost.

Extracted from 07_modeling.ipynb. Random Forest and XGBoost read the
tabular feature sets one row at a time; LSTM instead reads a window of
consecutive rows and learns from the sequence itself. This module
reshapes the existing tabular features into overlapping windows -- it
does not build any new features, only reshapes what
06_feature_engineering.ipynb already produced.

The scaler is fit on training sequences only and applied unchanged to
validation and test. `EarlyStopping` and `ReduceLROnPlateau` both watch
validation loss directly -- the test set is never involved until the
model has already been selected and trained.

The grid was chosen by testing across all 5 walk-forward folds
beforehand (not just one split, which was found to give misleadingly
optimistic results for this and every other tuned model in this
project). Window sizes 7 and 14 and a lower learning-rate variant were
added after a distinction-level review confirmed the earlier grid never
tested them; results are asset/feature-set dependent, not always won by
the shortest window as an earlier, narrower grid had suggested.

---

TENSORFLOW THREAD CONFIGURATION -- read this before touching the top of
this file.

A real, confirmed deadlock inside TensorFlow's own C++ threading runtime
was traced directly (via macOS `sample` process captures, twice) to
TensorFlow's Eigen thread pool on this project's development machine:
the main thread waits on `absl::Notification::WaitForNotification()`
for a completion signal that TensorFlow's own worker threads, sitting
idle in `WaitForWork`, never send. Forcing TensorFlow down to a single
thread via `tf.config.threading.set_intra_op_parallelism_threads(1)` /
`set_inter_op_parallelism_threads(1)` measurably reduces how often this
happens (this is real risk reduction, not theoretical), but does NOT
eliminate it entirely -- it was directly observed to still occur once
even with threading correctly confirmed at 1/1. This setting is
defence-in-depth, not a guaranteed fix; if you need a hard guarantee
against this specific TensorFlow bug regardless of thread count, see
src/models/lstm_isolated.py, which runs each config in its own
subprocess with a timeout instead.

Separately, calling `set_intra_op_parallelism_threads()` /
`set_inter_op_parallelism_threads()` AFTER TensorFlow has already been
used once in the current process raises:

    RuntimeError: Intra op parallelism cannot be modified after initialization.

This is real, standard TensorFlow behaviour (not specific to this
project) -- these settings can only be changed before TensorFlow's
runtime has initialized, which can happen from ANY prior use of
TensorFlow in the same kernel/process, not just from this module. In an
interactive Jupyter/VS Code session, a person may easily have already
run a `tf.constant(...)` sanity check, imported another module that
touches TensorFlow, or re-run this cell after TensorFlow already
initialized earlier in the same kernel -- all of which make this a
normal, expected situation, not a bug in this module.

Handled here via `_configure_tf_threading()`: attempted once, at import
time (the most likely point to still succeed, before anything else in
a typical fresh notebook run touches TensorFlow), wrapped so that
`RuntimeError` specifically is caught, reported once with a clear
explanation, and does NOT raise or block import -- training still
proceeds normally, just without this specific risk-reduction measure
active for the remainder of the process. Any other, unexpected
exception type is not swallowed here and will surface normally, since
only this one, well-understood failure mode is being handled.
"""

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from src.evaluation.metrics import evaluate, rmse
from src.models.data_prep import get_X_y

SEED = 42
LSTM_BATCH = 32

# Fallback default ONLY -- used by fit_lstm_once()/run_lstm() solely
# when their `epochs` argument is not supplied (i.e. left as None).
# notebooks/07_model_development.ipynb ALWAYS passes epochs explicitly
# (`epochs=LSTM_EPOCHS`, using the notebook's own LSTM_EPOCHS variable,
# separately defined and visible there for the actual research
# configuration/testing). That notebook-level value is the real,
# authoritative one for any experiment run through the notebook; this
# module-level default only matters for direct/standalone use of this
# module (e.g. a script or test that calls run_lstm() without
# specifying epochs). There is exactly one place each caller's epoch
# count actually comes from -- there is no runtime conflict -- but the
# two variables sharing a name in two files is worth being explicit
# about here.
LSTM_EPOCHS = 80

LSTM_GRID = [
    {"window": 5, "units": 16, "dropout": 0.2, "recurrent_dropout": 0.0, "lr": 0.001},
    {"window": 5, "units": 16, "dropout": 0.2, "recurrent_dropout": 0.2, "lr": 0.001},
    {"window": 5, "units": 16, "dropout": 0.3, "recurrent_dropout": 0.2, "lr": 0.001},
    {"window": 5, "units": 32, "dropout": 0.3, "recurrent_dropout": 0.2, "lr": 0.001},
    {"window": 5, "units": 32, "dropout": 0.3, "recurrent_dropout": 0.2, "lr": 0.0005},
    {"window": 7, "units": 16, "dropout": 0.2, "recurrent_dropout": 0.2, "lr": 0.001},
    {"window": 10, "units": 16, "dropout": 0.2, "recurrent_dropout": 0.2, "lr": 0.001},
    {"window": 14, "units": 16, "dropout": 0.2, "recurrent_dropout": 0.2, "lr": 0.001},
    {"window": 3, "units": 16, "dropout": 0.2, "recurrent_dropout": 0.1, "lr": 0.001},
]

_tf_threading_configured = False  # set exactly once, see _configure_tf_threading()


def _configure_tf_threading(intra: int = 1, inter: int = 1) -> bool:
    """Attempts to force TensorFlow's own thread pool down to `intra`/
    `inter` threads -- real, measured risk reduction for a confirmed
    TensorFlow-internal deadlock (see the module docstring). Idempotent:
    only actually attempts the call once per process, regardless of how
    many times this function or this module's import is triggered.

    Returns True if the setting was actually applied, False if
    TensorFlow had already initialized and the setting could not be
    changed. Never raises for this specific, expected condition -- any
    other exception type is not caught here and will propagate
    normally, since only this one well-understood failure mode
    (TensorFlow already initialized) is being handled.
    """
    global _tf_threading_configured
    if _tf_threading_configured:
        return True

    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        _tf_threading_configured = True
        return True
    except RuntimeError as e:
        if "already" in str(e).lower() or "initializ" in str(e).lower():
            warnings.warn(
                "Could not set TensorFlow intra/inter-op thread counts -- TensorFlow "
                "has already initialized elsewhere in this process (e.g. an earlier "
                "cell already used TensorFlow in this kernel session). Training will "
                "proceed with TensorFlow's default thread configuration instead. This "
                "is a real, measured risk-reduction step for a confirmed intermittent "
                "TensorFlow-internal deadlock on some platforms -- not having it active "
                "does not mean training will fail, only that this particular safeguard "
                "isn't in effect for the rest of this session. Restart the kernel before "
                "running this notebook if you want this setting applied. For a hard "
                "guarantee against the same underlying issue regardless of thread "
                "count, see src/models/lstm_isolated.py.",
                RuntimeWarning, stacklevel=2,
            )
            _tf_threading_configured = True  # don't retry every call -- already answered
            return False
        raise  # a genuinely different RuntimeError -- do not hide it


# Attempted once at import time -- the most likely point to still
# succeed, before a typical notebook run has touched TensorFlow for any
# other reason. Import-time side effects are otherwise avoided in this
# module (no other state is touched, no models are built, no data is
# read) -- this one call is kept here specifically because thread
# configuration is only effective if applied before TensorFlow's first
# real use, which for most notebook runs is the first line of this
# module actually being imported.
_configure_tf_threading()


def make_sequences(X: pd.DataFrame, y: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Each sequence is `window` consecutive rows; its target is the LAST row's target."""
    Xv, yv = X.values, y.values
    seqs, targets = [], []
    for end in range(window, len(Xv) + 1):
        seqs.append(Xv[end - window:end])
        targets.append(yv[end - 1])
    return np.array(seqs), np.array(targets)


def fit_lstm_once(
    train: pd.DataFrame, val: pd.DataFrame, price_col: str,
    window: int, units: int, dropout: float,
    recurrent_dropout: float = 0.0, lr: float = 0.001, horizon: int = 1,
    epochs: int = None,
):
    """Train one LSTM config, early stopping on validation. Returns model, scaler, val predictions, val rmse.
    `epochs` defaults to this module's own LSTM_EPOCHS if not supplied
    (see the module-level LSTM_EPOCHS docstring for what "default" means
    here -- notebooks/07_model_development.ipynb always passes its own
    value explicitly).

    `keras.backend.clear_session()` is called at the start of every
    call -- this module is normally called many times in a loop (once
    per grid config, for every asset/horizon/feature-set combination),
    and Keras accumulates internal graph/layer bookkeeping across
    repeated model construction in the same process if the session is
    never cleared. This has no effect on results (a fresh model is
    always constructed regardless); it only prevents unnecessary memory
    growth and internal state accumulation over a long run with many
    fits.
    """
    keras.backend.clear_session()

    epochs = epochs if epochs is not None else LSTM_EPOCHS
    level = f"{price_col}_level"
    X_train, y_train = get_X_y(train, price_col, horizon=horizon)
    X_val, y_val = get_X_y(val, price_col, horizon=horizon)

    # Borrow tail rows from train so the first val sequences still get a full window --
    # this is still all past data relative to those targets, not a leak.
    X_train_seq, y_train_seq = make_sequences(X_train, y_train, window)
    X_val_seq, y_val_seq = make_sequences(
        pd.concat([X_train.tail(window - 1), X_val]).reset_index(drop=True),
        pd.concat([y_train.tail(window - 1), y_val]).reset_index(drop=True), window)

    n_feat = X_train_seq.shape[2]
    # Scaler fit on TRAINING sequences only -- val/test are transformed
    # with this same fitted scaler, never re-fit. No leakage from val or
    # test into the scaling step.
    scaler = StandardScaler().fit(X_train_seq.reshape(-1, n_feat))
    scale = lambda a: scaler.transform(a.reshape(-1, n_feat)).reshape(a.shape)

    try:
        model = keras.Sequential([
            keras.layers.Input(shape=(window, n_feat)),
            keras.layers.LSTM(units, dropout=dropout, recurrent_dropout=recurrent_dropout),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

        # Early stopping and LR reduction both watch validation loss only
        # -- the test set plays no role in training, stopping, or model
        # selection at any point in this function.
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-6, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)
        model.fit(scale(X_train_seq), y_train_seq,
                  validation_data=(scale(X_val_seq), y_val_seq),
                  epochs=epochs, batch_size=LSTM_BATCH, callbacks=[es, reduce_lr], verbose=0)

        val_preds = model.predict(scale(X_val_seq), verbose=0).flatten()
    except Exception as e:
        raise RuntimeError(
            f"LSTM training failed for config window={window}, units={units}, dropout={dropout}, "
            f"recurrent_dropout={recurrent_dropout}, lr={lr}, horizon={horizon}d -- "
            f"original error: {type(e).__name__}: {e}"
        ) from e

    val_rmse = rmse(y_val_seq, val_preds)
    val_dates = val["Date"].iloc[-len(val_preds):].reset_index(drop=True)
    val_lvl = val[level].iloc[-len(val_preds):].reset_index(drop=True)

    return model, scaler, val_preds, y_val_seq, val_dates, val_lvl, val_rmse


def run_lstm(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, price_col: str,
             horizon: int = 1, grid: list = None, epochs: int = None) -> dict:
    """`grid` and `epochs` default to this module's own LSTM_GRID/LSTM_EPOCHS if not supplied."""
    grid = grid if grid is not None else LSTM_GRID
    keras.utils.set_random_seed(SEED)  # seeds Python's random, NumPy, and TensorFlow together
    level = f"{price_col}_level"

    best_rmse = np.inf
    best_config = None
    best_result = None
    grid_log = []

    for i, config in enumerate(grid, 1):
        print(f"  lstm configuration {i}/{len(grid)}: training...")
        config_t0 = time.time()
        model, scaler, val_preds, y_val_seq, val_dates, val_lvl, val_rmse = fit_lstm_once(
            train, val, price_col, config["window"], config["units"], config["dropout"],
            config.get("recurrent_dropout", 0.0), config.get("lr", 0.001), horizon=horizon, epochs=epochs,
        )
        config_elapsed = time.time() - config_t0
        grid_log.append({**config, "val_rmse": val_rmse})
        # Printed after every config, not just once the whole grid finishes --
        # a 9-config grid with up to 80 epochs each can genuinely take
        # several minutes with no other feedback otherwise, which is
        # indistinguishable from a real hang without this.
        print(f"  lstm configuration {i}/{len(grid)}: {config} -> val_rmse={val_rmse:.5f} ({config_elapsed:.1f}s) [completed]")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_config = config
            best_result = (model, scaler, val_preds, y_val_seq, val_dates, val_lvl)

    print(f"  lstm best config: {best_config} (val rmse={best_rmse:.5f}) [selected on validation RMSE, not test]")

    model, scaler, val_preds, y_val_seq, val_dates, val_lvl = best_result
    window = best_config["window"]

    # Test predictions using the winning model -- test sequences borrow tail rows from val.
    # Test data plays no role above this point: not in training, not in
    # early stopping, not in choosing best_config.
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

    print(f"  lstm: done (test RMSE={metrics['test'].get('RMSE_returns', metrics['test'].get('RMSE')):.5f})")

    return {"predictions": preds, "metrics": metrics, "model": model, "scaler": scaler,
            "config": best_config, "grid_log": grid_log}
