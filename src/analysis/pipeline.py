"""Statistical analysis pipeline: stationarity, correlation, and Granger causality.

This module is the main entry point for the analysis presented in Chapter
4.7, converting the merged dataset into the stationarity, correlation, and
Granger causality tables and figures used in the dissertation.

Run from the project root:

    python -m src.analysis.pipeline
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from src import config
from src.analysis import stationarity, correlation, causality

logger = logging.getLogger(__name__)


def run_statistical_analysis(
    df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = config.CAUSALITY_RESULTS_DIR,
) -> dict:
    """Run the full statistical analysis workflow.

    Args:
        df: Optional pre-loaded merged dataset. If not provided, it is
            loaded from `config.MERGED_FILE`.
        output_dir: Location where tables and figures should be saved.
            If None, nothing is saved to disk.

    Returns:
        A dictionary containing every table produced, plus the selected
        Granger causality lag.
    """

    logger.info("Starting statistical analysis pipeline.")

    if df is None:
        if not config.MERGED_FILE.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at: {config.MERGED_FILE}. "
                "Run `python -m src.preprocessing.pipeline` first."
            )
        df = pd.read_csv(config.MERGED_FILE, parse_dates=["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    tables_dir = output_dir / "tables" if output_dir else None
    figures_dir = output_dir / "figures" if output_dir else None
    if tables_dir is not None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ------------------------------------------------------------------
    # Stationarity
    # ------------------------------------------------------------------
    logger.info("Running stationarity tests on levels and first differences.")

    stationarity_original = stationarity.test_stationarity(df, config.STATIONARITY_VARIABLES)
    stationarity_differenced = stationarity.test_stationarity_differenced(
        df, config.STATIONARITY_VARIABLES
    )

    results["stationarity_original"] = stationarity_original
    results["stationarity_differenced"] = stationarity_differenced

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------
    logger.info("Preparing the stationary dataset and running correlation analysis.")

    analysis_df, transformation_summary = correlation.prepare_stationary_dataset(
        df=df,
        correlation_variables=config.CORRELATION_VARIABLES,
        stationarity_results=stationarity_original,
        price_variables=config.PRICE_VARIABLES,
    )
    results["correlation_transformations"] = transformation_summary

    correlation_results = correlation.run_correlation_analysis(analysis_df)
    results.update(correlation_results)

    # ------------------------------------------------------------------
    # Granger causality
    # ------------------------------------------------------------------
    logger.info("Running Granger causality analysis.")

    causality_results = causality.run_causality_analysis(
        analysis_df=analysis_df,
        target_variables=config.GRANGER_TARGET_VARIABLES,
        predictor_variables=config.GRANGER_PREDICTOR_VARIABLES,
        granger_pairs=config.GRANGER_PAIRS,
        max_lag=14,
    )
    results.update(causality_results)

    # ------------------------------------------------------------------
    # Save tables and figures
    # ------------------------------------------------------------------
    if tables_dir is not None:
        _save_tables(results, tables_dir)
        _save_figures(results, figures_dir)
        logger.info("Statistical analysis tables and figures saved to %s", output_dir)

    logger.info("Statistical analysis completed successfully.")

    return results


def _save_tables(results: dict, tables_dir: Path) -> None:
    """Save every DataFrame produced by the pipeline to its own CSV file."""

    table_names = [
        "stationarity_original",
        "stationarity_differenced",
        "correlation_transformations",
        "pearson_correlation_matrix",
        "pearson_pvalue_matrix",
        "spearman_correlation_matrix",
        "spearman_pvalue_matrix",
        "correlation_pairwise_results",
        "granger_lag_selection",
        "granger_causality_results",
        "granger_raw_pvalue_matrix",
        "granger_adjusted_pvalue_matrix",
        "granger_significant_relationships",
        "granger_analysis_summary",
    ]

    for name in table_names:
        table = results[name]
        index_needed = name.endswith("_matrix")
        table.to_csv(tables_dir / f"{name}.csv", index=index_needed)


def _save_figures(results: dict, figures_dir: Path) -> None:
    """Save the Pearson, Spearman, and Granger heatmaps."""

    display_labels = config.DISPLAY_LABELS

    for method, matrix_key, filename in [
        ("Pearson", "pearson_correlation_matrix", "pearson_correlation_heatmap.png"),
        ("Spearman", "spearman_correlation_matrix", "spearman_correlation_heatmap.png"),
    ]:
        matrix = results[matrix_key].rename(index=display_labels, columns=display_labels)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

        plt.figure(figsize=(9, 7))
        sns.heatmap(
            matrix.astype(float), mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={"label": f"{method} correlation coefficient"},
        )
        plt.title(f"{method} Correlation Heatmap\nStationary Variable Representations")
        plt.xlabel("")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    # Granger heatmap, with significance asterisks on the adjusted p-values
    adjusted_matrix = results["granger_adjusted_pvalue_matrix"]
    adjusted_display = adjusted_matrix.rename(index=display_labels, columns=display_labels)

    annotations = adjusted_display.round(3).astype(str)
    for predictor in adjusted_matrix.index:
        for target in adjusted_matrix.columns:
            if adjusted_matrix.loc[predictor, target] < 0.05:
                row_label = display_labels.get(predictor, predictor)
                col_label = display_labels.get(target, target)
                annotations.loc[row_label, col_label] += "*"

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        adjusted_display.astype(float), annot=annotations, fmt="",
        cmap="viridis_r", vmin=0, vmax=1,
        cbar_kws={"label": "FDR-adjusted p-value"},
    )
    plt.title(f"Granger Causality Heatmap\nFDR-Adjusted p-values at Lag {results['selected_lag']}")
    plt.xlabel("Target (predicted return)")
    plt.ylabel("Predictor")
    plt.tight_layout()
    plt.savefig(figures_dir / "granger_causality_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )

    analysis_results = run_statistical_analysis()

    print("\nStationarity (differenced):")
    print(analysis_results["stationarity_differenced"].to_string(index=False))

    print("\nGranger causality summary:")
    print(analysis_results["granger_analysis_summary"].to_string(index=False))