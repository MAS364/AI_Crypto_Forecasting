"""Generate visualizations from saved permutation-importance results.

The script reads the outputs produced by `run_feature_importance_grid.py`
and creates seven visualizations using matplotlib.

No SHAP plotting functions are used.

Produces:
- top10_random_forest.png
- top10_xgboost.png
- rf_vs_xgboost_comparison.png
- feature_group_importance.png
- configuration_comparison.png
- horizon_comparison.png
- btc_vs_eth_comparison.png

Usage
-----
Run from the project root:

    python generate_feature_importance_plots.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = PROJECT_ROOT / "results" / "feature_importance"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# ============================================================
# REPRESENTATIVE CONFIGURATION
# ============================================================

# Bitcoin, 14-day, Combined is used as the representative
# configuration for the detailed model-level comparisons.
REPRESENTATIVE = {
    "asset": "Bitcoin",
    "horizon": 14,
    "feature_set_key": "combined",
}


# ============================================================
# TOP-10 FEATURE IMPORTANCE
# ============================================================

def top10_by_model(long_df: pd.DataFrame) -> None:
    """Create top-10 permutation-importance plots for RF and XGBoost."""

    for model_name in ["Random Forest", "XGBoost"]:

        subset = long_df[
            (long_df["asset"] == REPRESENTATIVE["asset"])
            & (long_df["horizon"] == REPRESENTATIVE["horizon"])
            & (
                long_df["feature_set_key"]
                == REPRESENTATIVE["feature_set_key"]
            )
            & (long_df["model_name"] == model_name)
        ].sort_values(
            "mean_importance",
            ascending=False,
        ).head(10)

        if subset.empty:
            raise ValueError(
                "No permutation-importance results found for "
                f"{model_name}, {REPRESENTATIVE}."
            )

        fig, ax = plt.subplots(figsize=(9, 5.5))

        colors = [
            "#4c72b0" if value >= 0 else "#c44e52"
            for value in subset["mean_importance"][::-1]
        ]

        ax.barh(
            range(len(subset)),
            subset["mean_importance"][::-1],
            xerr=subset["std_importance"][::-1],
            color=colors,
        )

        ax.set_yticks(range(len(subset)))

        ax.set_yticklabels(
            subset["feature"][::-1],
            fontsize=8,
        )

        ax.set_xlabel(
            "Permutation Importance "
            "(Permuted RMSE - Baseline RMSE)"
        )

        ax.axvline(
            0,
            color="black",
            linewidth=0.8,
        )

        ax.set_title(
            f"Top-10 {model_name} Feature Importance (Permutation)\n"
            f"{REPRESENTATIVE['asset']}, "
            f"{REPRESENTATIVE['horizon']}-day horizon, "
            "Combined configuration"
        )

        plt.tight_layout()

        filename = (
            f"top10_{model_name.lower().replace(' ', '_')}.png"
        )

        path = FIGURES_DIR / filename

        plt.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {path}")


# ============================================================
# RANDOM FOREST VS XGBOOST
# ============================================================

def rf_vs_xgboost(long_df: pd.DataFrame) -> None:
    """Compare RF and XGBoost permutation importance."""

    subset = long_df[
        (long_df["asset"] == REPRESENTATIVE["asset"])
        & (long_df["horizon"] == REPRESENTATIVE["horizon"])
        & (
            long_df["feature_set_key"]
            == REPRESENTATIVE["feature_set_key"]
        )
    ]

    if subset.empty:
        raise ValueError(
            "No results found for representative configuration: "
            f"{REPRESENTATIVE}"
        )

    top_features = (
        subset.groupby("feature")["mean_importance"]
        .mean()
        .abs()
        .nlargest(12)
        .index
    )

    plot_df = subset[
        subset["feature"].isin(top_features)
    ]

    pivot = plot_df.pivot(
        index="feature",
        columns="model_name",
        values="mean_importance",
    )

    pivot = pivot.reindex(top_features)

    required_models = {
        "Random Forest",
        "XGBoost",
    }

    missing_models = required_models.difference(
        pivot.columns
    )

    if missing_models:
        raise ValueError(
            "Missing model results required for comparison: "
            f"{sorted(missing_models)}"
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    y = np.arange(len(pivot))
    height = 0.38

    ax.barh(
        y - height / 2,
        pivot["Random Forest"],
        height,
        label="Random Forest",
        color="#2ca02c",
    )

    ax.barh(
        y + height / 2,
        pivot["XGBoost"],
        height,
        label="XGBoost",
        color="#ff7f0e",
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        pivot.index,
        fontsize=8,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_xlabel(
        "Permutation Importance "
        "(Permuted RMSE - Baseline RMSE)"
    )

    ax.set_title(
        "Random Forest vs XGBoost -- Permutation Importance\n"
        f"{REPRESENTATIVE['asset']}, "
        f"{REPRESENTATIVE['horizon']}-day horizon, "
        "Combined configuration"
    )

    ax.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "rf_vs_xgboost_comparison.png"
    )

    plt.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# FEATURE-GROUP IMPORTANCE
# ============================================================

def feature_group_importance(group_df: pd.DataFrame) -> None:
    """Plot feature-group importance for the representative configuration."""

    subset = group_df[
        (group_df["asset"] == REPRESENTATIVE["asset"])
        & (group_df["horizon"] == REPRESENTATIVE["horizon"])
        & (
            group_df["feature_set_key"]
            == REPRESENTATIVE["feature_set_key"]
        )
    ]

    if subset.empty:
        raise ValueError(
            "No feature-group results found for "
            f"{REPRESENTATIVE}."
        )

    pivot = subset.pivot(
        index="feature_group",
        columns="model_name",
        values="mean_importance",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(
        kind="bar",
        ax=ax,
        color=["#2ca02c", "#ff7f0e"],
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "Mean Permutation Importance per Feature"
    )

    ax.set_xlabel("Feature Group")

    ax.set_title(
        "Feature-Group Importance\n"
        f"{REPRESENTATIVE['asset']}, "
        f"{REPRESENTATIVE['horizon']}-day horizon, "
        "Combined configuration"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "feature_group_importance.png"
    )

    plt.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# CONFIGURATION COMPARISON
# ============================================================

def configuration_comparison(
    group_df: pd.DataFrame,
) -> None:
    """Compare feature groups across feature configurations."""

    subset = group_df[
        group_df["asset"] == REPRESENTATIVE["asset"]
    ]

    subset = (
        subset.groupby(
            [
                "feature_set_key",
                "feature_group",
            ]
        )["mean_importance"]
        .mean()
        .reset_index()
    )

    pivot = subset.pivot(
        index="feature_group",
        columns="feature_set_key",
        values="mean_importance",
    )

    expected_columns = [
        "primary",
        "candidate",
        "ablation",
        "combined",
    ]

    pivot = pivot.reindex(
        columns=expected_columns
    )

    pivot.columns = [
        "Market Only",
        "Market + Google\nTrends",
        "Market + Fear\n& Greed",
        "Market + Google Trends\n+ Fear & Greed",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        pivot.values,
        cmap="RdYlGn",
        aspect="auto",
    )

    ax.set_xticks(
        range(len(pivot.columns))
    )

    ax.set_xticklabels(
        pivot.columns,
        fontsize=8,
    )

    ax.set_yticks(
        range(len(pivot.index))
    )

    ax.set_yticklabels(
        pivot.index,
        fontsize=9,
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):

            value = pivot.values[i, j]

            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    plt.colorbar(
        im,
        ax=ax,
        label="Mean Permutation Importance",
    )

    ax.set_title(
        "Feature-Group Importance Across Configurations\n"
        f"{REPRESENTATIVE['asset']}, averaged across "
        "models and horizons"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "configuration_comparison.png"
    )

    plt.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# HORIZON COMPARISON
# ============================================================

def horizon_comparison(
    group_df: pd.DataFrame,
) -> None:
    """Compare feature-group importance across forecast horizons."""

    subset = (
        group_df.groupby(
            [
                "horizon",
                "feature_group",
            ]
        )["mean_importance"]
        .mean()
        .reset_index()
    )

    pivot = subset.pivot(
        index="feature_group",
        columns="horizon",
        values="mean_importance",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(
        kind="bar",
        ax=ax,
        color=[
            "#4c72b0",
            "#dd8452",
            "#55a868",
        ],
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "Mean Permutation Importance per Feature"
    )

    ax.set_xlabel("Feature Group")

    ax.set_title(
        "Feature-Group Importance Across Forecast Horizons\n"
        "(averaged across assets, configurations, and models)"
    )

    ax.legend(
        title="Horizon (days)"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "horizon_comparison.png"
    )

    plt.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# BITCOIN VS ETHEREUM
# ============================================================

def btc_vs_eth_comparison(
    group_df: pd.DataFrame,
) -> None:
    """Compare Bitcoin and Ethereum feature-group importance."""

    subset = (
        group_df.groupby(
            [
                "asset",
                "feature_group",
            ]
        )["mean_importance"]
        .mean()
        .reset_index()
    )

    pivot = subset.pivot(
        index="feature_group",
        columns="asset",
        values="mean_importance",
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    pivot.plot(
        kind="bar",
        ax=ax,
        color=[
            "#f7931a",
            "#627eea",
        ],
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_ylabel(
        "Mean Permutation Importance per Feature"
    )

    ax.set_xlabel("Feature Group")

    ax.set_title(
        "Bitcoin vs Ethereum -- Feature-Group Importance\n"
        "(averaged across horizons, configurations, and models)"
    )

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "btc_vs_eth_comparison.png"
    )

    plt.savefig(
        path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Load saved importance results and generate all visualizations."""

    long_path = (
        RESULTS_DIR
        / "raw"
        / "permutation_importance_long.csv"
    )

    group_path = (
        RESULTS_DIR
        / "processed"
        / "feature_group_importance.csv"
    )

    if not long_path.exists():
        raise FileNotFoundError(
            f"Missing permutation-importance results:\n"
            f"{long_path}\n\n"
            "Run run_feature_importance_grid.py first."
        )

    if not group_path.exists():
        raise FileNotFoundError(
            f"Missing feature-group importance results:\n"
            f"{group_path}\n\n"
            "Run run_feature_importance_grid.py first."
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reading permutation importance: {long_path}")
    print(f"Reading feature-group importance: {group_path}")
    print(f"Figures directory: {FIGURES_DIR}\n")

    long_df = pd.read_csv(long_path)
    group_df = pd.read_csv(group_path)

    top10_by_model(long_df)
    rf_vs_xgboost(long_df)
    feature_group_importance(group_df)
    configuration_comparison(group_df)
    horizon_comparison(group_df)
    btc_vs_eth_comparison(group_df)

    print("\nAll visualizations generated successfully.")
    print(f"Saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()