"""Unified production entry point: train / evaluate / predict.

This module orchestrates the reusable functions already built in
src/analysis/feature_engineering.py, src/models/autoencoder.py, and
src/models/pipeline.py -- it does not reimplement any scientific logic,
only sequences existing, already-tested functions into three commands.

Run from the project root:

    python -m src.pipeline train
    python -m src.pipeline evaluate
    python -m src.pipeline predict --horizon 1 --asset Bitcoin \\
        --feature-set primary --representation Original --model "Random Forest" \\
        --input path/to/new_features.csv

TRAIN builds feature sets (if not already present), trains autoencoders
(if not already present), and trains every model -- exactly what
Notebooks 06, 06b, and 07 do, callable without opening Jupyter.

EVALUATE never retrains anything. It loads already-saved predictions and
metrics, runs the same validation checks and comparison tables Notebook 08
builds, and saves fresh copies to results/evaluation/ -- useful for
regenerating dissertation tables without touching model training at all.

PREDICT never retrains anything either. It loads one specific saved model
(and its scaler/encoder, if the model needs one), validates the input
shape, produces a return-space prediction, and reconstructs a price-space
prediction -- the same reconstruction formula (price_level * exp(return))
used everywhere else in this project.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.analysis import feature_engineering as fe
from src.models.autoencoder import train_autoencoder, encode as ae_encode, save_autoencoder
from src.models.pipeline import (
    run_modeling_pipeline, FORECAST_HORIZONS, ASSETS,
    FEATURE_FILE_TEMPLATES, DISPLAY_NAMES, _horizon_suffix,
)
from src.evaluation import loaders, comparison

AUTOENCODER_DIR = config.RESULTS_DIR / "autoencoder"
AE_MODELS_DIR = AUTOENCODER_DIR / "models"
AE_DATA_DIR = AUTOENCODER_DIR / "encoded_features"
EVALUATION_DIR = config.RESULTS_DIR / "evaluation"

# feature_set_name -> (include_external, include_ablation), matching the
# exact parameters Notebook 06 uses for each of the four configurations.
FEATURE_SET_PARAMS = {
    "primary": (False, False),
    "candidate": (True, False),
    "ablation": (False, True),
    "combined": (True, True),
}


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

def _build_features_if_missing(horizons, assets) -> None:
    """Builds and saves any feature CSVs that don't already exist -- matching
    Notebook 06's build_feature_set() calls exactly. Skips any (horizon,
    asset, feature_set) combination whose file is already on disk, so this
    is safe to call every time train() runs without wastefully rebuilding
    everything."""
    if not config.MERGED_FILE.exists():
        raise FileNotFoundError(
            f"{config.MERGED_FILE} not found -- run 03_data_preparation.ipynb "
            "(or src.preprocessing.pipeline) first."
        )
    df = pd.read_csv(config.MERGED_FILE, parse_dates=["Date"])

    for horizon in horizons:
        suffix = _horizon_suffix(horizon)
        for asset in assets:
            for key, (include_external, include_ablation) in FEATURE_SET_PARAMS.items():
                filename = FEATURE_FILE_TEMPLATES[key].format(a=asset.lower(), s=suffix)
                path = config.FEATURES_DIR / filename
                if path.exists():
                    continue
                feature_set = fe.build_feature_set(
                    df, asset=asset, feature_set_name=key,
                    include_external=include_external, include_ablation=include_ablation,
                    horizon=horizon,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                feature_set.to_csv(path, index=False)
                print(f"  built feature set: {path}")


def _train_autoencoders_if_missing(horizons, assets) -> None:
    """Trains and saves any autoencoder configurations that don't already
    exist -- matching 06b_autoencoder.ipynb exactly. Skips any combination
    whose encoded CSVs are already on disk."""
    from src.models.data_prep import split_data, get_X_y

    for horizon in horizons:
        suffix = _horizon_suffix(horizon)
        for asset, price_col in assets.items():
            for key in FEATURE_SET_PARAMS:
                prefix = f"{asset.lower()}_{key}{suffix}"
                encoded_test_path = AE_DATA_DIR / f"{prefix}_encoded_test.csv"
                if encoded_test_path.exists():
                    continue

                feature_path = config.FEATURES_DIR / FEATURE_FILE_TEMPLATES[key].format(a=asset.lower(), s=suffix)
                if not feature_path.exists():
                    raise FileNotFoundError(f"{feature_path} not found -- feature building should have created this already.")

                df = pd.read_csv(feature_path, parse_dates=["Date"])
                train, val, test = split_data(df)
                level_col = f"{price_col}_level"
                target_col = f"{price_col}_log_return_target_t_plus_{horizon}"

                X_train, _ = get_X_y(train, price_col, horizon)
                X_val, _ = get_X_y(val, price_col, horizon)
                X_test, _ = get_X_y(test, price_col, horizon)

                autoencoder, encoder, scaler, history = train_autoencoder(X_train, X_val)

                AE_DATA_DIR.mkdir(parents=True, exist_ok=True)
                encoded_cols = [f"encoded_{i}" for i in range(encoder.output_shape[-1])]
                for split_name, (X_split, source_df) in {
                    "train": (X_train, train), "validation": (X_val, val), "test": (X_test, test),
                }.items():
                    encoded = ae_encode(encoder, scaler, X_split)
                    edf = pd.DataFrame(encoded, columns=encoded_cols)
                    edf["Date"] = source_df["Date"].values
                    edf[level_col] = source_df[level_col].values
                    edf[target_col] = source_df[target_col].values
                    edf.to_csv(AE_DATA_DIR / f"{prefix}_encoded_{split_name}.csv", index=False)

                save_autoencoder(autoencoder, encoder, scaler, history, AE_MODELS_DIR / prefix,
                                  metadata={"horizon": horizon, "asset": asset, "feature_set": DISPLAY_NAMES[key]})
                print(f"  trained autoencoder: {prefix}")


def run_train(horizons: list = None, assets: dict = None, skip_if_exists: bool = True) -> dict:
    """Full training pipeline: feature engineering -> autoencoders -> models.
    Never touched by evaluate() or predict() -- this is the only command
    that fits anything."""
    horizons = horizons if horizons is not None else FORECAST_HORIZONS
    assets = assets if assets is not None else ASSETS

    print("=== Step 1/3: Feature engineering ===")
    _build_features_if_missing(horizons, assets)

    print("=== Step 2/3: Autoencoder training ===")
    _train_autoencoders_if_missing(horizons, assets)

    print("=== Step 3/3: Model training ===")
    results = run_modeling_pipeline(horizons=horizons, assets=assets)

    return results


# ---------------------------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------------------------

def run_evaluate() -> dict:
    """Never retrains or re-predicts anything. Loads saved predictions and
    metrics, runs validation checks and builds comparison tables, saves
    fresh copies to results/evaluation/ -- the CLI-callable equivalent of
    Notebook 08's Sections 2 and 3."""
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    predictions = loaders.load_predictions()
    metrics = loaders.load_metrics()

    checks = {
        "predictions_exist": len(predictions) > 0,
        "metrics_exist": len(metrics) > 0,
        "all_horizons_present": set(predictions["horizon"].unique()) == {1, 7, 14},
        "all_models_present": set(predictions["model"].unique()) == {"Naive", "ARIMA", "Random Forest", "XGBoost", "LSTM"},
        "every_run_succeeded": (metrics["status"] == "success").all(),
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    if not all(checks.values()):
        raise AssertionError("Validation failed -- see above. Fix the underlying saved artefacts, not this command.")

    master_comparison = comparison.build_master_comparison(metrics)
    validation_summary = comparison.build_validation_summary(metrics)

    master_comparison.to_csv(EVALUATION_DIR / "master_comparison.csv", index=False)
    validation_summary.to_csv(EVALUATION_DIR / "validation_summary.csv", index=False)

    print(f"\nSaved comparison tables to {EVALUATION_DIR}")
    return {"master_comparison": master_comparison, "validation_summary": validation_summary, "checks": checks}


# ---------------------------------------------------------------------------
# PREDICT
# ---------------------------------------------------------------------------

def run_predict(
    horizon: int, asset: str, feature_set_key: str, model_name: str,
    representation: str, input_df: pd.DataFrame, price_col: str = None,
) -> dict:
    """True inference: loads one saved model (never retrains), validates
    the input, predicts a return, and reconstructs a price. Supports all
    five models. `representation` must be "Original" or "Autoencoder-Encoded".
    """
    price_col = price_col or ASSETS.get(asset, f"{asset}_Price")
    level_col = f"{price_col}_level"

    if level_col not in input_df.columns:
        raise ValueError(f"Input is missing the required '{level_col}' column -- needed to reconstruct a price prediction.")

    if model_name in ("Naive", "ARIMA"):
        raise ValueError(
            f"{model_name} has no saved fitted model to load for inference -- it is fit fresh on "
            "each historical window rather than saved. Use run_modeling_pipeline() with real "
            "history if you need a Naive/ARIMA forecast, not this function."
        )

    target_col_guess = f"{price_col}_log_return_target_t_plus_{horizon}"
    feature_cols = [c for c in input_df.columns if c not in ("Date", level_col, target_col_guess)]
    X = input_df[feature_cols]

    if representation == "Autoencoder-Encoded":
        from src.models.autoencoder import predict_encode
        suffix = _horizon_suffix(horizon)
        prefix = f"{asset.lower()}_{feature_set_key}{suffix}"
        # predict_encode() loads the already-fitted encoder for this exact
        # (horizon, asset, feature_set) and validates X's column count
        # against it -- never retrains or refits anything.
        encoded = predict_encode(AE_MODELS_DIR / prefix, X)
        X = pd.DataFrame(encoded, columns=[f"encoded_{i}" for i in range(encoded.shape[1])])

    if representation == "Autoencoder-Encoded":
        # Encoded-branch models are saved by 07_model_development.ipynb to
        # a completely separate, FLAT directory -- results/modeling/
        # fitted_models/ -- not nested under models/trained/horizon_Xday/
        # asset/feature_dir/ the way Original-branch models are. Verified
        # directly against a real directory listing: e.g.
        # bitcoin_ablation_14day_random_forest_encoded.joblib,
        # bitcoin_ablation_14day_lstm_encoded.keras,
        # bitcoin_ablation_14day_lstm_encoded_scaler.joblib,
        # bitcoin_ablation_14day_lstm_encoded_config.json (config, not
        # metadata). The long asset/feature/horizon-prefixed filename is
        # required here specifically because this directory has no
        # subfolder structure to encode that context the way the
        # Original-branch directories do -- confirmed correct, not a
        # guess this time.
        fitted_models_dir = config.RESULTS_DIR / "modeling" / "fitted_models"
        model_slug = model_name.lower().replace(" ", "_")
        enc_prefix = f"{asset.lower()}_{feature_set_key}{_horizon_suffix(horizon)}_{model_slug}_encoded"
        if model_name == "LSTM":
            from tensorflow import keras
            model_path = fitted_models_dir / f"{enc_prefix}.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"{model_path} not found.")
            model = keras.models.load_model(model_path)
        else:
            model_path = fitted_models_dir / f"{enc_prefix}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"{model_path} not found.")
            model = joblib.load(model_path)
    else:
        fitted_models_dir = None
        enc_prefix = None
        model = loaders.load_trained_model(horizon, asset, feature_set_key, model_name)

    nan_note = None
    if model_name == "LSTM":
        # LSTM needs a WINDOW of consecutive rows (not a single row) and
        # its own fitted scaler -- see src/models/lstm_model.py's
        # make_sequences()/fit_lstm_once(). Window size, the exact
        # feature column order, and the scaler's path all live in the
        # saved metadata (verified directly against a real
        # lstm_metadata.json: "sequence_config": {"window": ...},
        # "feature_names": [...], "scaler_path": "...") -- reusing
        # load_model_metadata(), which already correctly reads this file
        # for ARIMA/Naive, rather than guessing a separate filename.
        if representation == "Autoencoder-Encoded":
            scaler_path = fitted_models_dir / f"{enc_prefix}_scaler.joblib"
            config_path = fitted_models_dir / f"{enc_prefix}_config.json"
            if not scaler_path.exists():
                raise FileNotFoundError(f"{scaler_path} not found -- LSTM's fitted scaler is required for inference.")
            if not config_path.exists():
                raise FileNotFoundError(f"{config_path} not found -- LSTM's window/config metadata is required for inference.")
            scaler = joblib.load(scaler_path)
            lstm_config = json.load(open(config_path))
            # The encoded-branch config.json (saved from run_lstm()'s
            # winning grid config) has no "feature_names" list the way
            # lstm_metadata.json does -- the encoded feature columns are
            # just encoded_0..encoded_N in a fixed order already, so
            # there's nothing to reorder/validate against by name.
            lstm_meta = {"sequence_config": {"window": lstm_config["window"]}, "feature_names": None}
        else:
            lstm_meta = loaders.load_model_metadata(horizon, asset, feature_set_key, "LSTM")
            scaler = joblib.load(config.PROJECT_ROOT / lstm_meta["scaler_path"])

        window = lstm_meta["sequence_config"]["window"]
        expected_features = lstm_meta.get("feature_names")
        if expected_features:
            missing = [c for c in expected_features if c not in X.columns]
            if missing:
                raise ValueError(f"Input is missing feature column(s) LSTM was trained on: {missing}")
            X_ordered = X[expected_features]
        else:
            X_ordered = X

        if len(X_ordered) < window:
            raise ValueError(
                f"LSTM needs at least {window} consecutive rows to build one input sequence "
                f"(this model's trained window size) -- got {len(X_ordered)}. Upload more historical rows."
            )

        # Last `window` consecutive rows, most recent last -- matches
        # make_sequences()'s convention exactly (each sequence's target is
        # the LAST row's target, i.e. the row we're predicting from).
        X_window = X_ordered.values[-window:]

        # NaN anywhere in this window silently propagates through every
        # LSTM layer and comes back out as NaN -- which then crashes
        # FastAPI's JSON response encoder (NaN isn't valid JSON) well
        # after this function has already returned, with no useful error
        # message at all. Catching it here, with the actual affected
        # columns/dates named, instead of a mystery 500 downstream.
        nan_mask = pd.isna(X_window)  # shape (window, n_features)
        if nan_mask.any():
            col_names = expected_features if expected_features else list(range(X_window.shape[1]))
            bad_cols = [col_names[i] for i in sorted(set(np.where(nan_mask.any(axis=0))[0]))]
            bad_dates = []
            if "Date" in input_df.columns:
                window_dates = input_df["Date"].iloc[-window:].reset_index(drop=True)
                bad_row_idx = sorted(set(np.where(nan_mask.any(axis=1))[0]))
                bad_dates = [str(window_dates.iloc[i]) for i in bad_row_idx]
            raise ValueError(
                f"Input has missing/NaN values in the {window}-row window LSTM needs, in "
                f"column(s) {bad_cols}"
                + (f" on date(s) {bad_dates}" if bad_dates else "")
                + " -- fill or remove these before uploading, LSTM cannot predict through missing data."
            )

        n_feat = X_window.shape[1]
        X_scaled = scaler.transform(X_window.reshape(-1, n_feat)).reshape(1, window, n_feat)
        # model.predict() sets up a tf.data pipeline and distribution
        # strategy meant for large batches -- genuinely slow overhead for
        # a single sequence, separate from the model's actual math.
        # Calling the model directly skips that path entirely; this is
        # the standard fix for single-sample Keras inference latency.
        predicted_return = float(model(X_scaled, training=False).numpy().flatten()[0])
    else:
        # Random Forest and XGBoost have native missing-value handling in
        # their split logic (learned during training: "if this feature is
        # missing, go left/right") and predict fine on a NaN input row --
        # confirmed directly by real successful predictions on this exact
        # kind of data before any check existed here. So this is NOT
        # blocked (an earlier version of this check incorrectly did) --
        # just surfaced as a transparency note on the result, since the
        # user should still know NaN was present even though the model
        # handled it, rather than that being invisible.
        last_row = X.values[-1]
        nan_note = None
        if pd.isna(last_row).any():
            bad_cols = [c for c, is_nan in zip(X.columns, pd.isna(last_row)) if is_nan]
            nan_note = (
                f"Column(s) {bad_cols} had missing values in the most recent row -- "
                f"{model_name} handled this automatically via its built-in missing-value "
                "routing (unlike LSTM, which cannot). Worth checking these aren't affecting "
                "prediction quality if this happens often."
            )
        predicted_return = float(model.predict(X.values)[-1])

    price_level = float(input_df[level_col].iloc[-1])
    predicted_price = price_level * np.exp(predicted_return)

    # Final catch-all -- covers any other NaN/inf source (e.g. a
    # zero-variance feature making the scaler produce inf) that the
    # column-level checks above didn't anticipate. Same reasoning: fail
    # here with a real message, not downstream in FastAPI's JSON encoder.
    if not np.isfinite(predicted_return) or not np.isfinite(predicted_price):
        raise ValueError(
            f"Prediction came out non-finite (predicted_return={predicted_return}, "
            f"predicted_price={predicted_price}) -- check the input for missing values, "
            "zero-variance columns, or values far outside the training range."
        )

    result = {
        "horizon": horizon, "asset": asset, "model": model_name,
        "feature_set": feature_set_key, "representation": representation,
        "input_price_level": price_level,
        "predicted_return": predicted_return,
        "predicted_price": predicted_price,
    }
    if nan_note:
        result["data_quality_note"] = nan_note
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, or predict -- production entry point.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Build features, train autoencoders, train all models.")
    subparsers.add_parser("evaluate", help="Load saved results and build comparison tables. Never retrains.")

    predict_parser = subparsers.add_parser("predict", help="Load a saved model and predict. Never retrains.")
    predict_parser.add_argument("--horizon", type=int, required=True, choices=[1, 7, 14])
    predict_parser.add_argument("--asset", required=True, choices=["Bitcoin", "Ethereum"])
    predict_parser.add_argument("--feature-set", required=True, choices=list(FEATURE_SET_PARAMS.keys()))
    predict_parser.add_argument("--representation", required=True, choices=["Original", "Autoencoder-Encoded"])
    predict_parser.add_argument("--model", required=True, choices=["Random Forest", "XGBoost", "LSTM"])
    predict_parser.add_argument("--input", required=True, help="Path to a CSV with the same feature columns Notebook 07 trained on.")

    args = parser.parse_args()

    if args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "predict":
        input_df = pd.read_csv(args.input, parse_dates=["Date"])
        result = run_predict(
            horizon=args.horizon, asset=args.asset, feature_set_key=args.feature_set,
            model_name=args.model, representation=args.representation, input_df=input_df,
        )
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
