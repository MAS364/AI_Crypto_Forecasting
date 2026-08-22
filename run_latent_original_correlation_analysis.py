"""Computes descriptive Pearson + Spearman correlations between the 8
autoencoder latent dimensions (encoded_0..encoded_7) and every real
original engineered feature they were compressed from, for every
(asset, horizon, feature_set_key) configuration -- the analysis flagged
as entirely missing in the Phase 1 verification of this project's
feature-importance work.

Uses src.feature_importance.latent_correlation, which merges the
encoded test set and the matching original engineered-feature test rows
on the real Date column (never assumes matching row order). See that
module's docstring for the full alignment and interpretation discipline
this script relies on -- in particular: this is a DESCRIPTIVE analysis,
not a claim that any latent dimension "represents" a single original
variable, and it deliberately does not attempt to establish causality.

Outputs
-------
    results/feature_importance_autoencoder/processed/latent_original_correlations_long.csv
    results/feature_importance_autoencoder/processed/latent_original_top_associations.csv
    results/feature_importance_autoencoder/figures/fig5_latent_original_correlation_heatmap.png

Usage
-----
Run from the project root, any time after the encoded feature CSVs
exist (results/autoencoder/encoded_features/) -- does not depend on
run_feature_importance_grid_autoencoder.py having been run first:

    python run_latent_original_correlation_analysis.py
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
from src.feature_importance.latent_correlation import (
    compute_latent_original_correlations,
    summarize_top_associations,
    LATENT_FEATURES,
)

OUT_DIR = config.RESULTS_DIR / "feature_importance_autoencoder"
PROCESSED_DIR = OUT_DIR / "processed"
FIGURES_DIR = OUT_DIR / "figures"
for d in (PROCESSED_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def plot_correlation_heatmap(long_df: pd.DataFrame, out_path: Path) -> None:
    """One heatmap per latent dimension would be 24 configurations x 8
    dimensions -- too many to read at once. Instead this plots, for
    every (asset, horizon, feature_set_key) configuration on the y-axis
    and every latent dimension on the x-axis, the single strongest
    |Pearson correlation| that dimension reaches with ANY original
    feature in that configuration -- a genuinely dissertation-readable
    summary of "how strongly tied to the original features is this
    latent dimension here", with the actual top-associated feature
    available in latent_original_top_associations.csv for anyone who
    wants the detail behind a given cell."""
    pivot_rows = []
    for keys, group in long_df.groupby(["asset", "horizon", "feature_set_key"]):
        asset, horizon, feature_set_key = keys
        row = {"config_label": f"{asset} {horizon}d {feature_set_key}"}
        for latent in LATENT_FEATURES:
            sub = group[group["latent_feature"] == latent]["absolute_pearson_correlation"].dropna()
            row[latent] = sub.max() if not sub.empty else np.nan
        pivot_rows.append(row)

    pivot = pd.DataFrame(pivot_rows).set_index("config_label")[LATENT_FEATURES]

    fig, ax = plt.subplots(figsize=(9, 10))
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.6 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="Strongest |Pearson r| with any original feature")
    ax.set_title(
        "Latent dimension vs strongest-associated original engineered feature\n"
        "(descriptive only -- not a claim of a one-to-one mapping; see full CSV for which feature)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}\n")

    long_df = compute_latent_original_correlations()
    long_path = PROCESSED_DIR / "latent_original_correlations_long.csv"
    long_df.to_csv(long_path, index=False)
    print(f"Saved latent-original correlation long table: rows={len(long_df)} path={long_path}")

    n_configs = long_df[["asset", "horizon", "feature_set_key"]].drop_duplicates().shape[0]
    print(f"Configurations covered: {n_configs}/24")
    n_undefined = long_df["pearson_correlation"].isna().sum()
    if n_undefined:
        print(f"Note: {n_undefined}/{len(long_df)} pairs have an undefined correlation "
              "(a constant column over the test window, or too few aligned observations) -- "
              "left as NaN, not coerced to zero.")

    top = summarize_top_associations(long_df, top_n=5)
    top_path = PROCESSED_DIR / "latent_original_top_associations.csv"
    top.to_csv(top_path, index=False)
    print(f"Saved top-5 associations per latent dimension per configuration: rows={len(top)} path={top_path}")

    plot_correlation_heatmap(long_df, FIGURES_DIR / "fig5_latent_original_correlation_heatmap.png")

    print("\n" + "=" * 70)
    print("IMPORTANT: this is a descriptive correlation analysis only.")
    print("A latent dimension correlating with one or more original features does")
    print("NOT mean it 'represents' that variable, and correlation here does not")
    print("establish causality in either direction. See this script's module")
    print("docstring and src/feature_importance/latent_correlation.py for the")
    print("full interpretation caveats before citing these numbers.")
    print("=" * 70)


if __name__ == "__main__":
    main()
