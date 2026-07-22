"""Correlation analysis on stationary (returns/differenced) variables.

Computes Pearson and Spearman correlation matrices with p-values, using
pairwise-complete observations for each pair, and combines both methods
into a single results table with a Benjamini-Hochberg FDR correction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def prepare_stationary_dataset(
    df: pd.DataFrame,
    correlation_variables: list[str],
    stationarity_results: pd.DataFrame,
    price_variables: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the stationary dataset used for correlation and Granger causality.

    Each variable's transformation is read from its own stationarity
    conclusion: already-stationary variables are kept as levels, price
    variables that are not stationary are converted to log returns, and
    everything else is first-differenced.

    Returns the transformed dataset and a log of which transformation was
    applied to each variable.
    """

    stationarity_lookup = stationarity_results.set_index("Variable")["Overall Conclusion"]

    analysis_df = pd.DataFrame(index=df.index)
    transformation_log = []

    for variable in correlation_variables:
        conclusion = stationarity_lookup.get(variable, "Unknown")

        if conclusion == "Stationary":
            analysis_df[variable] = df[variable]
            transformation = "Level (already stationary)"
        elif variable in price_variables:
            analysis_df[variable] = np.log(df[variable]).diff()
            transformation = "Log return"
        else:
            analysis_df[variable] = df[variable].diff()
            transformation = "First difference"

        transformation_log.append({
            "Variable": variable,
            "Original Stationarity Conclusion": conclusion,
            "Transformation Applied": transformation,
        })

    return analysis_df, pd.DataFrame(transformation_log)


def pairwise_correlation_matrices(data: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a correlation coefficient matrix and a matching p-value matrix.

    Each pair uses only the rows where both columns have a valid value
    (pairwise-complete observations), since trading volume has missing
    values that would otherwise shrink every pair to the same sample.

    method must be either "pearson" or "spearman".
    """

    columns = data.columns
    coef_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    pval_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

    test_function = stats.pearsonr if method == "pearson" else stats.spearmanr

    for col_a in columns:
        for col_b in columns:

            if col_a == col_b:
                coef_matrix.loc[col_a, col_b] = 1.0
                pval_matrix.loc[col_a, col_b] = 0.0
                continue

            paired = data[[col_a, col_b]].dropna()
            series_a = paired[col_a]
            series_b = paired[col_b]

            if series_a.std() == 0 or series_b.std() == 0:
                coef_matrix.loc[col_a, col_b] = np.nan
                pval_matrix.loc[col_a, col_b] = np.nan
                continue

            r, p = test_function(series_a, series_b)
            coef_matrix.loc[col_a, col_b] = r
            pval_matrix.loc[col_a, col_b] = p

    return coef_matrix, pval_matrix


def strength_label(r: float) -> str:
    """Label the size of a correlation coefficient, independent of its sign."""

    r = abs(r)
    if r < 0.20:
        return "Very weak"
    elif r < 0.40:
        return "Weak"
    elif r < 0.60:
        return "Moderate"
    elif r < 0.80:
        return "Strong"
    else:
        return "Very strong"


def build_pairwise_table(
    coef_matrix: pd.DataFrame, pval_matrix: pd.DataFrame, method_name: str
) -> pd.DataFrame:
    """Reshape a correlation matrix into one row per unique pair, with FDR correction."""

    columns = coef_matrix.columns
    rows = []

    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            r = coef_matrix.loc[col_a, col_b]
            p = pval_matrix.loc[col_a, col_b]

            if pd.isna(r) or pd.isna(p):
                continue

            rows.append({
                "Method": method_name,
                "Variable A": col_a,
                "Variable B": col_b,
                "Coefficient": round(r, 3),
                "p-value": round(p, 4),
                "Direction": "Positive" if r > 0 else "Negative",
                "Strength": strength_label(r),
            })

    table = pd.DataFrame(rows)

    reject, adjusted_p, _, _ = multipletests(table["p-value"], alpha=0.05, method="fdr_bh")
    table["Adjusted p-value (FDR)"] = adjusted_p.round(4)
    table["Significant (raw, 5%)"] = table["p-value"] < 0.05
    table["Significant (FDR-corrected)"] = reject

    return table


def run_correlation_analysis(analysis_df: pd.DataFrame) -> dict:
    """Run the full correlation stage: Pearson, Spearman, and the combined table.

    Returns a dictionary of every table produced, keyed by the name it is
    saved under.
    """

    pearson_matrix, pearson_pvalues = pairwise_correlation_matrices(analysis_df, "pearson")
    spearman_matrix, spearman_pvalues = pairwise_correlation_matrices(analysis_df, "spearman")

    pearson_pairwise = build_pairwise_table(pearson_matrix, pearson_pvalues, "Pearson")
    spearman_pairwise = build_pairwise_table(spearman_matrix, spearman_pvalues, "Spearman")

    pairwise_results = pd.concat([pearson_pairwise, spearman_pairwise], ignore_index=True)

    return {
        "pearson_correlation_matrix": pearson_matrix,
        "pearson_pvalue_matrix": pearson_pvalues,
        "spearman_correlation_matrix": spearman_matrix,
        "spearman_pvalue_matrix": spearman_pvalues,
        "correlation_pairwise_results": pairwise_results,
    }