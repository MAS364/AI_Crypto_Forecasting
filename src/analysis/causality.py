"""Granger causality testing between external factors and cryptocurrency returns.

The lag used for testing is selected with a VAR model rather than chosen by
inspection, and every pair is tested once, at that single selected lag, to
avoid the multiple-testing inflation of scanning across many lags and
reporting only the smallest p-value found.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests


def select_lag_order(granger_df: pd.DataFrame, max_lag: int = 14) -> tuple[int, pd.DataFrame]:
    """Fit a VAR model and select a lag using BIC.

    BIC is used as the final criterion because it penalises additional
    lags more heavily than AIC, a more conservative and appropriate choice
    for a relatively short daily sample. AIC, HQIC, and FPE are reported
    alongside it for transparency, since the criteria do not always agree.
    """

    var_model = VAR(granger_df)
    lag_order_results = var_model.select_order(maxlags=max_lag)

    lag_selection_table = pd.DataFrame({
        "Criterion": ["AIC", "BIC", "HQIC", "FPE"],
        "Selected Lag": [
            lag_order_results.aic,
            lag_order_results.bic,
            lag_order_results.hqic,
            lag_order_results.fpe,
        ],
    })
    lag_selection_table["Selected Lag"] = lag_selection_table["Selected Lag"].astype(int)

    selected_lag = int(lag_order_results.bic)
    if selected_lag < 1:
        selected_lag = 1

    return selected_lag, lag_selection_table


def run_granger_tests(
    granger_df: pd.DataFrame, granger_pairs: list[tuple[str, str]], selected_lag: int
) -> pd.DataFrame:
    """Run one Granger causality test per predictor-target pair, at the given lag."""

    results = []

    for predictor, target in granger_pairs:
        pair_data = granger_df[[target, predictor]].dropna()

        try:
            test_result = grangercausalitytests(pair_data, maxlag=selected_lag, verbose=False)
            f_stat = test_result[selected_lag][0]["ssr_ftest"][0]
            p_value = test_result[selected_lag][0]["ssr_ftest"][1]
            status = "Successful"
        except Exception as error:
            f_stat, p_value, status = np.nan, np.nan, f"Failed: {error}"

        results.append({
            "Predictor": predictor,
            "Target": target,
            "Selected Lag": selected_lag,
            "F-statistic": f_stat,
            "p-value": p_value,
            "Test Status": status,
        })

    return pd.DataFrame(results)


def apply_fdr_correction(granger_table: pd.DataFrame) -> pd.DataFrame:
    """Apply a Benjamini-Hochberg FDR correction across all successful tests."""

    successful_tests = granger_table[
        (granger_table["Test Status"] == "Successful") & (granger_table["p-value"].notna())
    ].copy()

    if successful_tests.empty:
        raise ValueError("No successful Granger causality tests were available.")

    reject, adjusted_p, _, _ = multipletests(successful_tests["p-value"], alpha=0.05, method="fdr_bh")

    successful_tests["Adjusted p-value (FDR)"] = adjusted_p
    successful_tests["Significant (raw, 5%)"] = successful_tests["p-value"] < 0.05
    successful_tests["Significant (FDR-corrected)"] = reject

    return successful_tests.sort_values("p-value").reset_index(drop=True)


def build_pvalue_matrices(
    successful_tests: pd.DataFrame, predictor_variables: list[str], target_variables: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape the test results into raw and FDR-adjusted p-value matrices."""

    raw_matrix = pd.DataFrame(np.nan, index=predictor_variables, columns=target_variables)
    adjusted_matrix = pd.DataFrame(np.nan, index=predictor_variables, columns=target_variables)

    for _, row in successful_tests.iterrows():
        raw_matrix.loc[row["Predictor"], row["Target"]] = row["p-value"]
        adjusted_matrix.loc[row["Predictor"], row["Target"]] = row["Adjusted p-value (FDR)"]

    return raw_matrix, adjusted_matrix


def run_causality_analysis(
    analysis_df: pd.DataFrame,
    target_variables: list[str],
    predictor_variables: list[str],
    granger_pairs: list[tuple[str, str]],
    max_lag: int = 14,
) -> dict:
    """Run the full Granger causality stage and return every table produced."""

    granger_variables = target_variables + predictor_variables
    granger_df = analysis_df[granger_variables].dropna()

    selected_lag, lag_selection_table = select_lag_order(granger_df, max_lag=max_lag)

    granger_table = run_granger_tests(granger_df, granger_pairs, selected_lag)
    successful_tests = apply_fdr_correction(granger_table)

    raw_matrix, adjusted_matrix = build_pvalue_matrices(
        successful_tests, predictor_variables, target_variables
    )

    significant_relationships = successful_tests[
        successful_tests["Significant (FDR-corrected)"]
    ].copy()

    total_tests = len(successful_tests)
    raw_significant = int(successful_tests["Significant (raw, 5%)"].sum())
    fdr_significant = int(successful_tests["Significant (FDR-corrected)"].sum())

    summary = pd.DataFrame({
        "Measure": [
            "Directional tests completed",
            "Relationships significant before correction",
            "Relationships significant after FDR correction",
            "Percentage significant before correction",
            "Percentage significant after FDR correction",
            "Selected lag",
        ],
        "Value": [
            total_tests,
            raw_significant,
            fdr_significant,
            round(raw_significant / total_tests * 100, 2) if total_tests else 0,
            round(fdr_significant / total_tests * 100, 2) if total_tests else 0,
            selected_lag,
        ],
    })

    return {
        "granger_lag_selection": lag_selection_table,
        "granger_causality_results": successful_tests,
        "granger_raw_pvalue_matrix": raw_matrix,
        "granger_adjusted_pvalue_matrix": adjusted_matrix,
        "granger_significant_relationships": significant_relationships,
        "granger_analysis_summary": summary,
        "selected_lag": selected_lag,
    }