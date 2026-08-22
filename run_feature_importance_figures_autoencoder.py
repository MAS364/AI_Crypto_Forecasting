"""Generates dissertation-ready figures for the feature-importance
chapter, especially a figure that VISUALIZES why Random Forest and
XGBoost disagree on the worst-agreement configuration in the Original-
branch grid -- the disagreement your supervisor asked to see visualised.

A note on "the Spearman 0.25 figure": Phase 1 verification of this
analysis checked this directly against
results/feature_importance/processed/rank_agreement_summary.csv --
0.25 (0.247) is the MEAN Spearman rho across all 24 Original-branch
configurations, not any single configuration's value. The single WORST
configuration (Ethereum, 7-day, Market + Fear & Greed Index) actually has
rho = -0.344, a stronger and more visually striking disagreement than the
mean figure suggests. This script visualises that worst-agreement
configuration (a legitimate and more dramatic choice for a "the two
models disagree" figure) and reports the mean 0.25 figure separately in
its own caption -- the two numbers are describing different things and
should not be conflated in the dissertation text.

Reuses only real, already-computed results (from
results/feature_importance/processed/, produced by
run_feature_importance_grid.py, and results/feature_importance_autoencoder/
processed/, produced by run_feature_importance_grid_autoencoder.py and
run_latent_original_correlation_analysis.py). Nothing is recomputed or
invented here; this script only reads CSVs and plots them.

HOW TO RUN
----------
1. Run run_feature_importance_grid.py (should already be done).
2. Run run_feature_importance_grid_autoencoder.py and
   run_combined_feature_importance_report.py, if you want the
   Autoencoder-Encoded agreement heatmap and representation-comparison
   figure too.
3. From the project root:
       python run_feature_importance_figures_autoencoder.py

OUTPUT (all PNGs, 300 dpi, saved to results/feature_importance_autoencoder/figures/
so nothing under results/feature_importance/ is touched):
    fig1_rank_agreement_heatmap_original.png
    fig2_worst_agreement_disagreement_detail.png
    fig2_worst_agreement_rank_scatter.png
    fig4_rank_agreement_heatmap_autoencoder.png     (if the Autoencoder grid was run)
    fig6_representation_comparison_rmse.png          (if run_combined_feature_importance_report.py was run)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.models.feature_naming import DISPLAY_NAMES

ORIGINAL_PROCESSED = config.RESULTS_DIR / "feature_importance" / "processed"
AE_PROCESSED = config.RESULTS_DIR / "feature_importance_autoencoder" / "processed"

FIGURES_DIR = config.RESULTS_DIR / "feature_importance_autoencoder" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RF_COL, XGB_COL = "mean_importance_Random Forest", "mean_importance_XGBoost"
RF_RANK_COL, XGB_RANK_COL = "rank_Random Forest", "rank_XGBoost"


def build_heatmap_pivot(agreement: pd.DataFrame) -> pd.DataFrame:
    df = agreement.copy()
    df["feature_set_display"] = df["feature_set_key"].map(DISPLAY_NAMES)
    df["config_label"] = df["asset"] + " " + df["horizon"].astype(str) + "d"
    pivot = df.pivot(index="feature_set_display", columns="config_label", values="spearman_rho")
    order = [DISPLAY_NAMES[k] for k in ["primary", "candidate", "ablation", "combined"]]
    return pivot.reindex(order)


def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black", fontsize=9)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", color="gray", fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman rank correlation (RF vs XGBoost)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_worst_agreement_detail(cross_model: pd.DataFrame, agreement: pd.DataFrame, out_prefix: str) -> None:
    """The core "why do they disagree" figure: finds the real
    configuration with the lowest Spearman rho (excluding NaN, which
    means "too few comparable features" rather than "perfect
    disagreement"), then plots RF vs XGBoost importance for its top
    features side by side, plus a rank-vs-rank scatter."""
    valid = agreement.dropna(subset=["spearman_rho"])
    if valid.empty:
        print("No configuration has a computable Spearman rho -- skipping disagreement figure.")
        return

    worst = valid.loc[valid["spearman_rho"].idxmin()]
    asset, horizon, fset = worst["asset"], worst["horizon"], worst["feature_set_key"]
    rho = worst["spearman_rho"]
    mean_rho = agreement["spearman_rho"].mean()
    fset_display = DISPLAY_NAMES.get(fset, fset)

    print(f"\nWorst rank-agreement configuration found: {asset}, {horizon}-day, "
          f"{fset_display} -- Spearman rho = {rho:.3f}")
    print(f"(Mean Spearman rho across all {len(agreement)} configurations: {mean_rho:.3f} -- "
          "this is the separate, distinct '~0.25' figure; see this script's module docstring.)")

    subset = cross_model[
        (cross_model["asset"] == asset) & (cross_model["horizon"] == horizon)
        & (cross_model["feature_set_key"] == fset) & (cross_model["both_available"])
    ].copy()
    if subset.empty:
        print("No shared-feature rows for this configuration -- skipping disagreement figure.")
        return

    top_disagree = subset.sort_values("abs_difference", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(top_disagree))
    height = 0.38
    ax.barh(y + height / 2, top_disagree[RF_COL], height=height, label="Random Forest", color="#2ca02c")
    ax.barh(y - height / 2, top_disagree[XGB_COL], height=height, label="XGBoost", color="#ff7f0e")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(top_disagree["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean permutation importance (Permuted RMSE - Baseline RMSE)")
    ax.set_title(
        f"Biggest RF vs XGBoost disagreements (worst-agreement configuration)\n"
        f"{asset}, {horizon}-day, {fset_display} (Spearman rho = {rho:.3f})"
    )
    ax.legend()
    fig.tight_layout()
    out_path = FIGURES_DIR / f"{out_prefix}_disagreement_detail.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(subset[RF_RANK_COL], subset[XGB_RANK_COL], alpha=0.75, edgecolor="black", linewidth=0.5)
    max_rank = max(subset[RF_RANK_COL].max(), subset[XGB_RANK_COL].max())
    ax.plot([0, max_rank], [0, max_rank], linestyle="--", color="gray",
             label="Perfect agreement (y = x)")
    for _, row in subset.sort_values("abs_difference", ascending=False).head(5).iterrows():
        ax.annotate(row["feature"], (row[RF_RANK_COL], row[XGB_RANK_COL]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Random Forest importance rank (1 = most important)")
    ax.set_ylabel("XGBoost importance rank (1 = most important)")
    ax.set_title(
        f"Feature rank agreement: RF vs XGBoost (worst-agreement configuration)\n"
        f"{asset}, {horizon}-day, {fset_display} (Spearman rho = {rho:.3f})"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.invert_yaxis()
    ax.invert_xaxis()
    fig.tight_layout()
    out_path2 = FIGURES_DIR / f"{out_prefix}_rank_scatter.png"
    fig.savefig(out_path2, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path2}")


def plot_representation_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    """One dot per (asset, horizon, feature_set_key, model_name) configuration:
    Original test RMSE vs Autoencoder-Encoded test RMSE. Points below the
    diagonal are configurations where encoding produced a LOWER
    (better) test RMSE than the original engineered features."""
    df = comparison.dropna(subset=["baseline_rmse_original", "baseline_rmse_encoded"])
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = {"Random Forest": "#1f77b4", "XGBoost": "#d62728"}
    for model_name, group in df.groupby("model_name"):
        ax.scatter(group["baseline_rmse_original"], group["baseline_rmse_encoded"],
                   label=model_name, alpha=0.75, edgecolor="black", linewidth=0.4,
                   color=colors.get(model_name))
    lims = [df[["baseline_rmse_original", "baseline_rmse_encoded"]].min().min(),
            df[["baseline_rmse_original", "baseline_rmse_encoded"]].max().max()]
    ax.plot(lims, lims, linestyle="--", color="gray", label="Equal RMSE")
    ax.set_xlabel("Original engineered features -- test RMSE")
    ax.set_ylabel("Autoencoder-Encoded features -- test RMSE")
    n_below = (df["baseline_rmse_encoded"] < df["baseline_rmse_original"]).sum()
    ax.set_title(
        f"Test RMSE: Original vs Autoencoder-Encoded representation\n"
        f"({n_below}/{len(df)} configurations: encoding gave a lower test RMSE)"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Figures directory: {FIGURES_DIR}\n")

    agreement_path = ORIGINAL_PROCESSED / "rank_agreement_summary.csv"
    cross_model_path = ORIGINAL_PROCESSED / "cross_model_comparison.csv"
    if agreement_path.exists() and cross_model_path.exists():
        agreement = pd.read_csv(agreement_path)
        cross_model = pd.read_csv(cross_model_path)

        pivot = build_heatmap_pivot(agreement)
        plot_heatmap(
            pivot,
            "RF vs XGBoost rank agreement (Spearman rho) -- Original engineered features",
            FIGURES_DIR / "fig1_rank_agreement_heatmap_original.png",
        )
        plot_worst_agreement_detail(cross_model, agreement, out_prefix="fig2_worst_agreement")
    else:
        print(f"Skipping Original-branch figures -- expected files not found:\n"
              f"  {agreement_path}\n  {cross_model_path}\n"
              f"  (run run_feature_importance_grid.py first)")

    ae_agreement_path = AE_PROCESSED / "rank_agreement_summary_autoencoder.csv"
    if ae_agreement_path.exists():
        ae_agreement = pd.read_csv(ae_agreement_path)
        ae_pivot = build_heatmap_pivot(ae_agreement)
        plot_heatmap(
            ae_pivot,
            "RF vs XGBoost rank agreement (Spearman rho) -- Autoencoder-Encoded features",
            FIGURES_DIR / "fig4_rank_agreement_heatmap_autoencoder.png",
        )
    else:
        print(f"\nSkipping Autoencoder-Encoded heatmap -- {ae_agreement_path} not found "
              f"(run run_feature_importance_grid_autoencoder.py first if you want this figure).")

    comparison_path = AE_PROCESSED / "representation_comparison_summary.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        plot_representation_comparison(comparison, FIGURES_DIR / "fig6_representation_comparison_rmse.png")
    else:
        print(f"\nSkipping representation-comparison figure -- {comparison_path} not found "
              f"(run run_combined_feature_importance_report.py first if you want this figure).")

    print("\nDone.")


if __name__ == "__main__":
    main()
