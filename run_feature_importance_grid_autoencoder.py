"""Run the full Permutation Feature Importance (PFI) grid for the
AUTOENCODER-ENCODED branch (Random Forest + XGBoost trained on the 8
latent encoded_0..encoded_7 dimensions), which run_feature_importance_grid.py
does NOT cover -- that script only ever runs against the Original
(engineered-feature) branch.

Mirrors run_feature_importance_grid.py's structure and validation
discipline exactly, for the Autoencoder-Encoded representation instead
of Original. No SHAP, no retraining -- reuses the project's real,
existing PFI engine (src.feature_importance.permutation).

Grid: 2 assets x 3 horizons x 4 feature configurations x 2 models
      (Random Forest, XGBoost) = 48 experiments.

Outputs
-------
    results/feature_importance_autoencoder/raw/permutation_importance_autoencoder_long.csv
    results/feature_importance_autoencoder/processed/top_features_random_forest_autoencoder.csv
    results/feature_importance_autoencoder/processed/top_features_xgboost_autoencoder.csv
    results/feature_importance_autoencoder/processed/cross_model_comparison_autoencoder.csv
    results/feature_importance_autoencoder/processed/rank_agreement_summary_autoencoder.csv
    results/feature_importance_autoencoder/processed/feature_group_importance_autoencoder.csv

Usage
-----
Run from the project root:

    python run_feature_importance_grid_autoencoder.py

This script FAILS LOUDLY (non-zero exit via a raised exception) if any
of the 48 required experiments does not succeed, rather than reporting a
partial run as complete -- see validate_full_grid() below.
"""

import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.models.feature_naming import _horizon_suffix, DISPLAY_NAMES
from src.feature_importance.permutation import calculate_permutation_importance, N_REPEATS, SEED
from src.feature_importance.aggregate import (
    compare_model_rankings,
    compute_rank_agreement_summary,
    calculate_feature_group_importance,
    create_importance_summary,
)

AE_DATA_DIR = config.RESULTS_DIR / "autoencoder" / "encoded_features"
FITTED_MODELS_DIR = config.RESULTS_DIR / "modeling" / "fitted_models"

OUT_DIR = config.RESULTS_DIR / "feature_importance_autoencoder"
RAW_DIR = OUT_DIR / "raw"
PROCESSED_DIR = OUT_DIR / "processed"
for d in (RAW_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

REPRESENTATION = "Autoencoder-Encoded"

ASSETS = ["Bitcoin", "Ethereum"]
HORIZONS = [1, 7, 14]
FEATURE_SET_KEYS = ["primary", "candidate", "ablation", "combined"]
MODELS = ["Random Forest", "XGBoost"]

EXPECTED_EXPERIMENTS = len(ASSETS) * len(HORIZONS) * len(FEATURE_SET_KEYS) * len(MODELS)  # 48
EXPECTED_LATENT_DIMS = 8
EXPECTED_TOTAL_ROWS = EXPECTED_EXPERIMENTS * EXPECTED_LATENT_DIMS  # 384


def load_encoded_model_and_data(horizon: int, asset: str, feature_set_key: str, model_name: str) -> dict:
    """Autoencoder-Encoded-branch equivalent of
    src.feature_importance.model_loading.load_model_and_data(), which
    only supports the Original branch. Paths and column-naming
    conventions here match src/pipeline.py's own encoded-branch loading
    logic (run_predict()) exactly -- verified against that function
    directly, not guessed.
    """
    price_col = config.ASSET_VARIABLES[asset]["price"]
    suffix = _horizon_suffix(horizon)
    prefix = f"{asset.lower()}_{feature_set_key}{suffix}"
    model_slug = model_name.lower().replace(" ", "_")

    test_path = AE_DATA_DIR / f"{prefix}_encoded_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Encoded test set not found: {test_path}")

    model_path = FITTED_MODELS_DIR / f"{prefix}_{model_slug}_encoded.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Encoded model not found: {model_path}")

    df = pd.read_csv(test_path)
    encoded_cols = [c for c in df.columns if c.startswith("encoded_")]
    if len(encoded_cols) != EXPECTED_LATENT_DIMS:
        raise ValueError(
            f"{test_path} has {len(encoded_cols)} encoded_* columns, expected {EXPECTED_LATENT_DIMS}."
        )

    target_col = f"{price_col}_log_return_target_t_plus_{horizon}"
    if target_col not in df.columns:
        raise FileNotFoundError(f"Expected target column '{target_col}' not found in {test_path}")

    X_test = df[encoded_cols]
    y_test = df[target_col]

    if X_test.isna().any().any():
        raise ValueError(f"{test_path} contains missing values in encoded_* columns.")
    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError(f"{test_path} contains non-finite values in encoded_* columns.")
    if y_test.isna().any():
        raise ValueError(f"{test_path} contains missing values in target column '{target_col}'.")

    model = joblib.load(model_path)

    n_model_features = getattr(model, "n_features_in_", None)
    if n_model_features is not None and n_model_features != len(encoded_cols):
        raise ValueError(
            f"{model_path} was fitted on {n_model_features} features, but the encoded test "
            f"set at {test_path} has {len(encoded_cols)} encoded_* columns -- mismatch."
        )

    return {"model": model, "X_test": X_test, "y_test": y_test}


def run_one_experiment(horizon: int, asset: str, feature_set_key: str, model_name: str) -> pd.DataFrame:
    loaded = load_encoded_model_and_data(horizon, asset, feature_set_key, model_name)
    pfi = calculate_permutation_importance(loaded["model"], loaded["X_test"], loaded["y_test"])
    pfi["asset"] = asset
    pfi["horizon"] = horizon
    pfi["feature_set_key"] = feature_set_key
    pfi["model_name"] = model_name
    pfi["representation"] = REPRESENTATION
    return pfi


def validate_full_grid(long_df: pd.DataFrame, n_ok: int, n_skip: int, n_fail: int, total: int) -> None:
    """Fails LOUDLY (raises) rather than letting a partial run be reported
    as complete. Every check here corresponds to a Phase 1 verification
    requirement: exact experiment count, exact final row count, no
    duplicate configuration/model/feature rows, finite importance
    values, and exactly 8 latent-feature rows per experiment.
    """
    problems = []
    if n_ok != total:
        problems.append(f"Only {n_ok}/{total} experiments succeeded ({n_skip} skipped, {n_fail} failed).")
    if len(long_df) != EXPECTED_TOTAL_ROWS:
        problems.append(f"Expected {EXPECTED_TOTAL_ROWS} total rows (48 experiments x 8 latent dims), got {len(long_df)}.")

    dup_cols = ["asset", "horizon", "feature_set_key", "model_name", "feature"]
    n_dupes = long_df.duplicated(subset=dup_cols).sum()
    if n_dupes:
        problems.append(f"{n_dupes} duplicate (asset, horizon, feature_set_key, model_name, feature) rows found.")

    if not np.isfinite(long_df["mean_importance"]).all():
        problems.append("Non-finite values found in mean_importance.")
    if not np.isfinite(long_df["std_importance"]).all():
        problems.append("Non-finite values found in std_importance.")

    per_experiment_counts = long_df.groupby(["asset", "horizon", "feature_set_key", "model_name"]).size()
    bad_counts = per_experiment_counts[per_experiment_counts != EXPECTED_LATENT_DIMS]
    if len(bad_counts):
        problems.append(f"{len(bad_counts)} experiment(s) do not have exactly {EXPECTED_LATENT_DIMS} feature rows.")

    if problems:
        raise RuntimeError(
            "Autoencoder-Encoded permutation-importance grid failed validation:\n  - "
            + "\n  - ".join(problems)
        )


def main() -> None:
    specs = [
        {"horizon": h, "asset": a, "feature_set_key": f, "model_name": m}
        for h, a, f, m in product(HORIZONS, ASSETS, FEATURE_SET_KEYS, MODELS)
    ]
    total = len(specs)
    assert total == EXPECTED_EXPERIMENTS, f"Grid definition changed: expected {EXPECTED_EXPERIMENTS} specs, got {total}."

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {OUT_DIR}")
    print(f"Running {total} Autoencoder-Encoded permutation-importance experiments "
          f"(n_repeats={N_REPEATS}, seed={SEED})...\n")

    all_results, n_ok, n_skip, n_fail = [], 0, 0, 0

    for i, spec in enumerate(specs, start=1):
        fset_name = DISPLAY_NAMES[spec["feature_set_key"]]
        label = f"{spec['asset']} h={spec['horizon']}d {fset_name:45s} {spec['model_name']}"
        try:
            pfi = run_one_experiment(**spec)
            all_results.append(pfi)
            n_ok += 1
            print(f"[{i:2d}/{total}] OK:     {label} -- {len(pfi)} features")
        except FileNotFoundError as exc:
            n_skip += 1
            print(f"[{i:2d}/{total}] SKIP:   {label} -- {exc}")
        except Exception as exc:
            n_fail += 1
            print(f"[{i:2d}/{total}] FAILED: {label} -- {type(exc).__name__}: {exc}")

    print(f"\nCompleted: {n_ok} successful, {n_skip} skipped, {n_fail} failed.")

    if not all_results:
        raise RuntimeError(
            "No Autoencoder-Encoded permutation-importance results were produced. "
            "Check that results/autoencoder/encoded_features/ and the "
            "*_encoded.joblib files in results/modeling/fitted_models/ exist."
        )

    long_df = pd.concat(all_results, ignore_index=True)

    # Fail loudly on a partial/invalid grid BEFORE saving anything, so a
    # broken run never silently overwrites the last good results.
    validate_full_grid(long_df, n_ok, n_skip, n_fail, total)

    long_path = RAW_DIR / "permutation_importance_autoencoder_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\nSaved raw long-format results: rows={len(long_df)} path={long_path}")

    top_features = create_importance_summary(long_df, top_n=10)
    for model_name, filename in {
        "Random Forest": "top_features_random_forest_autoencoder.csv",
        "XGBoost": "top_features_xgboost_autoencoder.csv",
    }.items():
        top_features[top_features["model_name"] == model_name].to_csv(PROCESSED_DIR / filename, index=False)
    print("Saved top-10 feature tables for Random Forest and XGBoost (Autoencoder-Encoded)")

    cross_model = compare_model_rankings(long_df)
    cross_model.to_csv(PROCESSED_DIR / "cross_model_comparison_autoencoder.csv", index=False)

    rank_agreement = compute_rank_agreement_summary(cross_model)
    rank_agreement.to_csv(PROCESSED_DIR / "rank_agreement_summary_autoencoder.csv", index=False)

    print(f"Saved cross-model comparison ({len(cross_model)} rows)")
    print(f"Saved rank-agreement summary ({len(rank_agreement)} configurations)")

    if "spearman_rho" in rank_agreement.columns and not rank_agreement.empty:
        mean_rho = rank_agreement["spearman_rho"].mean()
        print(f"Mean Spearman rho across Autoencoder-Encoded configurations: {mean_rho:.3f}")

    group_importance = calculate_feature_group_importance(long_df)
    group_importance.to_csv(PROCESSED_DIR / "feature_group_importance_autoencoder.csv", index=False)
    print(f"Saved feature-group importance: {len(group_importance)} rows "
          "(all 8 encoded_N dimensions fall under the 'Other' group -- a learned, compressed "
          "representation has no individual real-world source variable, so this is expected, not a bug)")

    print("\n" + "=" * 70)
    print("Autoencoder-Encoded permutation feature importance completed and validated.")
    print(f"Results saved under: {OUT_DIR}")
    print(f"Successful experiments: {n_ok}/{total}")
    print(f"Skipped experiments:    {n_skip}/{total}")
    print(f"Failed experiments:     {n_fail}/{total}")
    print("No SHAP dependency used in this run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
