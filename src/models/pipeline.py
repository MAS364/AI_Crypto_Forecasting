"""Model development pipeline: trains and evaluates every model, at every
forecast horizon, across every feature configuration.

This module is the production, script-based counterpart to
07_modeling.ipynb -- the notebook is the experimentation and dissertation
layer (markdown explanations, diagnostics, plots, tables for supervisor
review); this module is the reproducible, independently-runnable
implementation, calling the same underlying functions in src/models/ and
src/evaluation/metrics.py that back the notebook, so both layers use
identical scientific logic rather than two separately maintained copies.

Run from the project root:

    python -m src.models.pipeline

Trains: Naive, ARIMA, Random Forest, XGBoost, LSTM
Across: Bitcoin, Ethereum
At horizons: 1, 7, 14 days
Over feature configurations: Market Only, Market + Google Trends,
    Market + Fear & Greed Index, Market + Google Trends + Fear & Greed Index

Random Forest, XGBoost, and LSTM are additionally run on Autoencoder-Encoded
features (loaded from results/autoencoder/encoded_features/, produced by
06b_autoencoder_feature_extraction.ipynb) if those files are present --
every prediction and metric row carries a `representation` column
("Original" or "Autoencoder-Encoded") so both branches can be compared
directly. If the encoded files are not found, this branch is skipped with
a warning rather than failing the whole pipeline; ARIMA and Naive are never
part of this branch, since neither receives a feature matrix in the first
place.

This is a genuinely long-running script (the full grid, all models, all
horizons, all feature configurations -- the same combinatorial workload
as the notebook's main training loop). Expect it to take a comparable
amount of time to a full notebook run.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.evaluation.metrics import evaluate
from src.models.data_prep import split_data, get_X_y
from src.models.baseline import run_naive
from src.models.arima_model import run_arima
from src.models.tree_models import run_random_forest, run_xgboost
from src.models.lstm_model import run_lstm

logger = logging.getLogger(__name__)

FORECAST_HORIZONS = [1, 7, 14]
ASSETS = {"Bitcoin": "Bitcoin_Price", "Ethereum": "Ethereum_Price"}

# Internal keys, feature CSV filename templates, display names, and the
# horizon-suffix helper now live in src/models/feature_naming.py -- a
# dependency-free module, so code that only needs these naming constants
# (e.g. src/explainability/utils.py) doesn't have to import this module
# and, transitively, TensorFlow via src.models.lstm_model below. Re-exported
# here under the same names so every existing caller of
# `from src.models.pipeline import FEATURE_FILE_TEMPLATES` (etc.) is
# unaffected -- this is a relocation, not a behavior change.
from src.models.feature_naming import FEATURE_FILE_TEMPLATES, DISPLAY_NAMES, _horizon_suffix

# Saved model directories on disk use these names, not the raw internal
# keys above -- matches src/evaluation/loaders.py's DIR_NAMES exactly, and
# matches models/trained/manifest.json's actual saved paths. This was
# found to be missing here during production-pipeline testing: this
# module previously saved every model under the raw internal key
# (primary/candidate/ablation/combined) instead, silently diverging from
# what the manifest and loaders.py expected.
DIR_NAMES = {
    "primary": "market_only",
    "candidate": "market_google_trends",
    "ablation": "market_fear_greed",
    "combined": "market_all_external",
}

MODELING_RESULTS_DIR = config.RESULTS_DIR / "modeling"
MODELS_OUTPUT_DIR = config.MODELS_DIR / "trained"


def _load_feature_set(asset: str, feature_key: str, horizon: int) -> pd.DataFrame:
    fname = FEATURE_FILE_TEMPLATES[feature_key].format(a=asset.lower(), s=_horizon_suffix(horizon))
    path = config.FEATURES_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}. "
            "Run 06_feature_engineering.ipynb (or its src equivalent) first, "
            f"for horizon={horizon}."
        )
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def run_modeling_pipeline(
    horizons: list = None,
    assets: dict = None,
    output_dir: Optional[Path] = MODELING_RESULTS_DIR,
    save_models: bool = True,
) -> dict:
    """Run the full model development pipeline.

    Args:
        horizons: Forecast horizons to run. Defaults to FORECAST_HORIZONS (1, 7, 14).
        assets: Assets to run, as {display_name: price_column}. Defaults to ASSETS.
        output_dir: Where predictions, metrics, and feature importance are
            saved. If None, nothing is saved to disk.
        save_models: Whether to persist fitted Random Forest, XGBoost, and
            LSTM models (ARIMA has no picklable fitted object; Naive has none).

    Returns:
        A dict with keys "predictions", "metrics", "feature_importance" --
        each a single combined DataFrame across every horizon, asset,
        feature configuration, and model run.
    """
    horizons = horizons if horizons is not None else FORECAST_HORIZONS
    assets = assets if assets is not None else ASSETS

    logger.info(
        "Starting model development pipeline: %d horizons x %d assets x %d feature "
        "configurations x 5 models.", len(horizons), len(assets), len(FEATURE_FILE_TEMPLATES)
    )

    pred_dir = output_dir / "predictions" if output_dir else None
    metrics_dir = output_dir / "metrics" if output_dir else None
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pred_rows, metric_rows, importance_rows, log_rows = [], [], [], []
    t0 = time.time()

    for horizon in horizons:
        for asset, price_col in assets.items():
            train_naive = val_naive = test_naive = None  # populated on first feature set, reused across all four

            for feature_key in FEATURE_FILE_TEMPLATES:
                df = _load_feature_set(asset, feature_key, horizon)
                train, val, test = split_data(df)
                display_name = DISPLAY_NAMES[feature_key]

                logger.info("h=%dd | %s | %s", horizon, asset, display_name)

                # ARIMA and Naive are univariate / use only the asset's own price
                # series, identical across all four feature configurations --
                # computed once per (horizon, asset) via the "primary" pass,
                # reused for the other three to avoid four redundant,
                # numerically identical experiments.
                if feature_key == "primary":
                    naive_result = run_naive(train, val, test, price_col, horizon=horizon)
                    logger.info("  fitting ARIMA (shared across all feature configurations)")
                    arima_result = run_arima(train, val, test, price_col, horizon=horizon)
                    train_naive, val_naive, test_naive = train, val, test
                else:
                    naive_result = run_naive(train_naive, val_naive, test_naive, price_col, horizon=horizon)
                    arima_result = arima_result  # same fitted-forecast result as the primary pass

                runners = {
                    "Naive": lambda: naive_result,
                    "ARIMA": lambda: arima_result,
                    "Random Forest": lambda: run_random_forest(train, val, test, price_col, horizon=horizon),
                    "XGBoost": lambda: run_xgboost(train, val, test, price_col, horizon=horizon),
                    "LSTM": lambda: run_lstm(train, val, test, price_col, horizon=horizon),
                }

                for model_name, fn in runners.items():
                    t1 = time.time()
                    status = "success"
                    try:
                        result = fn()
                    except Exception as e:
                        logger.error("  %s failed: %s", model_name, e)
                        status = f"failed: {e}"
                        result = None
                    elapsed = 0.0 if model_name in ("Naive", "ARIMA") else time.time() - t1

                    log_rows.append({
                        "horizon": horizon, "asset": asset, "feature_set": display_name,
                        "model": model_name, "representation": "Original",
                        "status": status, "seconds": round(elapsed, 1),
                    })
                    if result is None:
                        continue

                    level_col = f"{price_col}_level"
                    for split_name, pred_df in result["predictions"].items():
                        source_df = {"validation": val, "test": test}[split_name]
                        price_lookup = source_df.set_index("Date")[level_col]
                        pdf = pred_df.copy()
                        pdf["price_level"] = pdf["event_date"].map(price_lookup).values
                        pdf["actual_price"] = pdf["price_level"] * np.exp(pdf["actual"])
                        pdf["predicted_price"] = pdf["price_level"] * np.exp(pdf["predicted"])
                        pdf["price_residual"] = pdf["actual_price"] - pdf["predicted_price"]
                        pdf = pdf.rename(columns={"actual": "actual_return", "predicted": "predicted_return"})
                        pdf["return_residual"] = pdf["actual_return"] - pdf["predicted_return"]
                        pdf["horizon"], pdf["asset"] = horizon, asset
                        pdf["model"], pdf["split"], pdf["feature_set"] = model_name, split_name, display_name
                        pdf["representation"] = "Original"
                        pred_rows.append(pdf[[
                            "horizon", "event_date", "asset", "model", "split", "feature_set", "representation",
                            "price_level", "actual_return", "predicted_return", "return_residual",
                            "actual_price", "predicted_price", "price_residual",
                        ]])

                    for split_name, m in result["metrics"].items():
                        metric_rows.append({
                            "horizon": horizon, "asset": asset, "feature_set": display_name,
                            "model": model_name, "representation": "Original", "split": split_name,
                            "seconds": round(elapsed, 1), "status": status, **m,
                        })

                    if model_name in ("Random Forest", "XGBoost") and "importance" in result:
                        imp = result["importance"].copy()
                        imp["horizon"], imp["asset"] = horizon, asset
                        imp["feature_set"], imp["model"] = display_name, model_name
                        imp["representation"] = "Original"
                        importance_rows.append(imp)

                    if save_models and model_name in ("Random Forest", "XGBoost", "LSTM"):
                        prefix = f"{asset.lower()}_{feature_key}{_horizon_suffix(horizon)}_{model_name.lower().replace(' ', '_')}"
                        target_dir = MODELS_OUTPUT_DIR / f"horizon_{horizon}day" / asset.lower() / DIR_NAMES[feature_key]
                        target_dir.mkdir(parents=True, exist_ok=True)
                        if model_name in ("Random Forest", "XGBoost"):
                            joblib.dump(result["model"], target_dir / f"{prefix}.joblib")
                        elif model_name == "LSTM":
                            result["model"].save(target_dir / f"{prefix}.keras")
                            joblib.dump(result["scaler"], target_dir / f"{prefix}_scaler.joblib")
                            json.dump(result["config"], open(target_dir / f"{prefix}_config.json", "w"))

    # --- Autoencoder-Encoded branch ---
    # Loads the encoded CSVs produced by 06b_autoencoder_feature_extraction.ipynb
    # (or its src equivalent, once built) -- does not train any autoencoder here.
    # Random Forest, XGBoost, and LSTM are re-run on the encoded features, using
    # the exact same functions as the Original branch above, since the encoded
    # dataframes share the identical structure (Date, features, level, target).
    # ARIMA and Naive are correctly excluded: ARIMA never receives a feature
    # matrix in the first place, and Naive has no features or parameters --
    # "encoded features" does not apply to either, structurally.
    AE_DATA_DIR = config.RESULTS_DIR / "autoencoder" / "encoded_features"
    ENCODING_DIM = 8  # must match 06b's ENCODING_DIM

    if not AE_DATA_DIR.exists() or not any(AE_DATA_DIR.iterdir()):
        logger.warning(
            "No encoded features found at %s -- skipping the Autoencoder-Encoded "
            "branch. Run 06b_autoencoder_feature_extraction.ipynb (or its src "
            "equivalent) first to include it.", AE_DATA_DIR
        )
    else:
        for horizon in horizons:
            for asset, price_col in assets.items():
                level_col = f"{price_col}_level"
                for feature_key in FEATURE_FILE_TEMPLATES:
                    display_name = DISPLAY_NAMES[feature_key]
                    prefix = f"{asset.lower()}_{feature_key}{_horizon_suffix(horizon)}"

                    split_frames = {}
                    missing = False
                    for split_key, split_filename in [("train", "train"), ("val", "validation"), ("test", "test")]:
                        path = AE_DATA_DIR / f"{prefix}_encoded_{split_filename}.csv"
                        if not path.exists():
                            logger.error("Encoded file missing: %s -- skipping this combination.", path)
                            missing = True
                            break
                        split_frames[split_key] = pd.read_csv(path, parse_dates=["Date"])
                    if missing:
                        continue

                    train_enc, val_enc, test_enc = split_frames["train"], split_frames["val"], split_frames["test"]
                    logger.info("h=%dd | %s | %s | [Encoded]", horizon, asset, display_name)

                    runners = {
                        "Random Forest": lambda: run_random_forest(train_enc, val_enc, test_enc, price_col, horizon=horizon),
                        "XGBoost": lambda: run_xgboost(train_enc, val_enc, test_enc, price_col, horizon=horizon),
                        "LSTM": lambda: run_lstm(train_enc, val_enc, test_enc, price_col, horizon=horizon),
                    }

                    for model_name, fn in runners.items():
                        t1 = time.time()
                        status = "success"
                        try:
                            result = fn()
                        except Exception as e:
                            logger.error("  [Encoded] %s failed: %s", model_name, e)
                            status = f"failed: {e}"
                            result = None
                        elapsed = time.time() - t1

                        log_rows.append({
                            "horizon": horizon, "asset": asset, "feature_set": display_name,
                            "model": model_name, "representation": "Autoencoder-Encoded",
                            "status": status, "seconds": round(elapsed, 1),
                        })
                        if result is None:
                            continue

                        for split_name, pred_df in result["predictions"].items():
                            source_df = {"validation": val_enc, "test": test_enc}[split_name]
                            price_lookup = source_df.set_index("Date")[level_col]
                            pdf = pred_df.copy()
                            pdf["price_level"] = pdf["event_date"].map(price_lookup).values
                            pdf["actual_price"] = pdf["price_level"] * np.exp(pdf["actual"])
                            pdf["predicted_price"] = pdf["price_level"] * np.exp(pdf["predicted"])
                            pdf["price_residual"] = pdf["actual_price"] - pdf["predicted_price"]
                            pdf = pdf.rename(columns={"actual": "actual_return", "predicted": "predicted_return"})
                            pdf["return_residual"] = pdf["actual_return"] - pdf["predicted_return"]
                            pdf["horizon"], pdf["asset"] = horizon, asset
                            pdf["model"], pdf["split"], pdf["feature_set"] = model_name, split_name, display_name
                            pdf["representation"] = "Autoencoder-Encoded"
                            pred_rows.append(pdf[[
                                "horizon", "event_date", "asset", "model", "split", "feature_set", "representation",
                                "price_level", "actual_return", "predicted_return", "return_residual",
                                "actual_price", "predicted_price", "price_residual",
                            ]])

                        for split_name, m in result["metrics"].items():
                            metric_rows.append({
                                "horizon": horizon, "asset": asset, "feature_set": display_name,
                                "model": model_name, "representation": "Autoencoder-Encoded", "split": split_name,
                                "seconds": round(elapsed, 1), "status": status, **m,
                            })

                        if model_name in ("Random Forest", "XGBoost") and "importance" in result:
                            imp = result["importance"].copy()
                            imp["horizon"], imp["asset"] = horizon, asset
                            imp["feature_set"], imp["model"] = display_name, model_name
                            imp["representation"] = "Autoencoder-Encoded"
                            importance_rows.append(imp)

                        if save_models and model_name in ("Random Forest", "XGBoost", "LSTM"):
                            enc_prefix = f"{prefix}_{model_name.lower().replace(' ', '_')}_encoded"
                            target_dir = MODELS_OUTPUT_DIR / f"horizon_{horizon}day" / asset.lower() / DIR_NAMES[feature_key]
                            target_dir.mkdir(parents=True, exist_ok=True)
                            if model_name in ("Random Forest", "XGBoost"):
                                joblib.dump(result["model"], target_dir / f"{enc_prefix}.joblib")
                            elif model_name == "LSTM":
                                result["model"].save(target_dir / f"{enc_prefix}.keras")
                                joblib.dump(result["scaler"], target_dir / f"{enc_prefix}_scaler.joblib")
                                json.dump(result["config"], open(target_dir / f"{enc_prefix}_config.json", "w"))

    all_preds = pd.concat(pred_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    importance_df = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    log_df = pd.DataFrame(log_rows)

    if pred_dir is not None:
        all_preds.to_csv(pred_dir / "all_model_predictions.csv", index=False)
        metrics_df.to_csv(metrics_dir / "model_development_metrics.csv", index=False)
        if not importance_df.empty:
            importance_df.to_csv(metrics_dir / "feature_importance.csv", index=False)
        log_df.to_csv(metrics_dir / "run_log.csv", index=False)
        logger.info("Saved predictions, metrics, feature importance, and run log to %s", output_dir)

    elapsed_min = (time.time() - t0) / 60
    logger.info("Model development pipeline completed in %.1f minutes.", elapsed_min)

    return {"predictions": all_preds, "metrics": metrics_df, "feature_importance": importance_df, "log": log_df}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )

    results = run_modeling_pipeline()

    print("\nRun summary (failures, if any):")
    failed = results["log"][results["log"]["status"] != "success"]
    print(failed.to_string(index=False) if len(failed) else "All runs succeeded.")

    print("\nTest-set RMSE by horizon, asset, model, representation (Market Only):")
    summary = results["metrics"][
        (results["metrics"]["feature_set"] == "Market Only") & (results["metrics"]["split"] == "test")
    ][["horizon", "asset", "model", "representation", "RMSE_returns", "R2_returns", "DirAcc_%"]]
    print(summary.sort_values(["horizon", "asset", "representation", "RMSE_returns"]).to_string(index=False))

    n_encoded = (results["metrics"]["representation"] == "Autoencoder-Encoded").sum()
    if n_encoded == 0:
        print("\nNote: no Autoencoder-Encoded results were produced -- run "
              "06b_autoencoder_feature_extraction.ipynb (or its src equivalent) "
              "first if that branch is expected.")
