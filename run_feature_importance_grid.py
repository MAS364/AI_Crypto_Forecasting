"""Run the full Permutation Feature Importance (PFI) grid.

This script evaluates every real combination for which the project has
a trained Random Forest or XGBoost model:

    2 assets
    x 3 forecast horizons
    x 4 feature configurations
    x 2 models
    = 48 experiments

No SHAP functionality is used in this script or the modules it imports.

Permutation importance is computed using
`sklearn.inspection.permutation_importance` against already-fitted models
and the real chronological test set.

Outputs are saved under:

    results/feature_importance/

Files produced
--------------
raw/permutation_importance_long.csv

processed/feature_dictionary.csv
processed/top_features_random_forest.csv
processed/top_features_xgboost.csv
processed/cross_model_comparison.csv
processed/feature_group_importance.csv
processed/statistical_evidence.csv
processed/rank_agreement_summary.csv

Usage
-----
Run from the project root:

    python run_feature_importance_grid.py
"""

import sys
from itertools import product
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results" / "feature_importance"
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"

for directory in (RAW_DIR, PROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.feature_importance.permutation import (
    run_feature_importance_experiment,
    N_REPEATS,
    SEED,
)

from src.feature_importance.grouping import (
    build_feature_dictionary,
)

from src.feature_importance.aggregate import (
    compare_model_rankings,
    compute_rank_agreement_summary,
    calculate_feature_group_importance,
    create_importance_summary,
)

from src.feature_importance.statistical_evidence import (
    merge_with_statistical_evidence,
)


# ============================================================
# EXPERIMENT GRID
# ============================================================

ASSETS = [
    "Bitcoin",
    "Ethereum",
]

HORIZONS = [
    1,
    7,
    14,
]

FEATURE_SET_KEYS = [
    "primary",
    "candidate",
    "ablation",
    "combined",
]

MODELS = [
    "Random Forest",
    "XGBoost",
]


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete permutation-importance experiment grid."""

    specs = [
        {
            "horizon": horizon,
            "asset": asset,
            "feature_set_key": feature_set_key,
            "model_name": model_name,
        }
        for horizon, asset, feature_set_key, model_name in product(
            HORIZONS,
            ASSETS,
            FEATURE_SET_KEYS,
            MODELS,
        )
    ]

    total = len(specs)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Results directory: {RESULTS_DIR}")
    print(
        f"Running {total} permutation-importance experiments "
        f"(n_repeats={N_REPEATS}, seed={SEED})...\n"
    )

    all_results = []

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for index, spec in enumerate(specs, start=1):

        label = (
            f"{spec['asset']} "
            f"h={spec['horizon']}d "
            f"{spec['feature_set_key']:9s} "
            f"{spec['model_name']}"
        )

        try:
            pfi = run_feature_importance_experiment(
                **spec
            )

            all_results.append(pfi)

            n_ok += 1

            print(
                f"[{index:2d}/{total}] OK:   "
                f"{label} -- {len(pfi)} features"
            )

        except FileNotFoundError as exc:
            n_skip += 1

            print(
                f"[{index:2d}/{total}] SKIP: "
                f"{label} -- {exc}"
            )

        except Exception as exc:
            n_fail += 1

            print(
                f"[{index:2d}/{total}] FAILED: "
                f"{label} -- "
                f"{type(exc).__name__}: {exc}"
            )

    print(
        f"\nCompleted: "
        f"{n_ok} successful, "
        f"{n_skip} skipped, "
        f"{n_fail} failed."
    )

    if not all_results:
        raise RuntimeError(
            "No permutation-importance results were produced. "
            "Check that trained Random Forest/XGBoost models and "
            "the required feature datasets are available."
        )

    # ========================================================
    # RAW LONG-FORM RESULTS
    # ========================================================

    long_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    long_path = (
        RAW_DIR
        / "permutation_importance_long.csv"
    )

    long_df.to_csv(
        long_path,
        index=False,
    )

    print(
        "\nSaved raw long-format results:"
        f"\n  rows: {len(long_df)}"
        f"\n  path: {long_path}"
    )

    # ========================================================
    # FEATURE DICTIONARY
    # ========================================================

    feature_dict = build_feature_dictionary(
        sorted(
            long_df["feature"]
            .dropna()
            .unique()
        )
    )

    feature_dict_path = (
        PROCESSED_DIR
        / "feature_dictionary.csv"
    )

    feature_dict.to_csv(
        feature_dict_path,
        index=False,
    )

    print(
        f"Saved feature dictionary: "
        f"{len(feature_dict)} unique features"
    )

    # ========================================================
    # TOP FEATURES BY MODEL
    # ========================================================

    top_features = create_importance_summary(
        long_df,
        top_n=10,
    )

    model_outputs = {
        "Random Forest":
            "top_features_random_forest.csv",
        "XGBoost":
            "top_features_xgboost.csv",
    }

    for model_name, filename in model_outputs.items():

        model_df = top_features[
            top_features["model_name"]
            == model_name
        ]

        model_df.to_csv(
            PROCESSED_DIR / filename,
            index=False,
        )

    print(
        "Saved top-10 feature tables for "
        "Random Forest and XGBoost"
    )

    # ========================================================
    # CROSS-MODEL COMPARISON + RANK AGREEMENT
    # ========================================================

    try:
        cross_model = compare_model_rankings(
            long_df
        )

        cross_model_path = (
            PROCESSED_DIR
            / "cross_model_comparison.csv"
        )

        cross_model.to_csv(
            cross_model_path,
            index=False,
        )

        rank_agreement = (
            compute_rank_agreement_summary(
                cross_model
            )
        )

        rank_path = (
            PROCESSED_DIR
            / "rank_agreement_summary.csv"
        )

        rank_agreement.to_csv(
            rank_path,
            index=False,
        )

        print(
            "Saved cross-model comparison "
            f"({len(cross_model)} rows)"
        )

        print(
            "Saved rank-agreement summary "
            f"({len(rank_agreement)} configurations)"
        )

        if (
            "spearman_rho"
            in rank_agreement.columns
            and not rank_agreement.empty
        ):
            mean_rho = (
                rank_agreement[
                    "spearman_rho"
                ].mean()
            )

            print(
                "Mean Spearman rho across "
                f"configurations: {mean_rho:.3f}"
            )

    except Exception as exc:
        print(
            "Cross-model comparison skipped: "
            f"{type(exc).__name__}: {exc}"
        )

    # ========================================================
    # FEATURE-GROUP IMPORTANCE
    # ========================================================

    group_importance = (
        calculate_feature_group_importance(
            long_df
        )
    )

    group_path = (
        PROCESSED_DIR
        / "feature_group_importance.csv"
    )

    group_importance.to_csv(
        group_path,
        index=False,
    )

    print(
        "Saved feature-group importance: "
        f"{len(group_importance)} rows"
    )

    # ========================================================
    # STATISTICAL-EVIDENCE CROSS-REFERENCE
    # ========================================================

    with_evidence = (
        merge_with_statistical_evidence(
            long_df
        )
    )

    evidence_path = (
        PROCESSED_DIR
        / "statistical_evidence.csv"
    )

    with_evidence.to_csv(
        evidence_path,
        index=False,
    )

    print(
        "Saved statistical-evidence "
        f"cross-reference: "
        f"{len(with_evidence)} rows"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("Permutation feature importance completed.")
    print(f"Results saved under: {RESULTS_DIR}")
    print(
        f"Successful experiments: {n_ok}/{total}"
    )
    print(
        f"Skipped experiments:    {n_skip}/{total}"
    )
    print(
        f"Failed experiments:     {n_fail}/{total}"
    )
    print("No SHAP dependency used in this run.")
    print("=" * 70)


if __name__ == "__main__":
    main()