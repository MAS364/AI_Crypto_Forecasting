"""Matches engineered features to their real, tested source variables and
looks up the corresponding correlation/Granger causality results.

Extracted verbatim from src/explainability/comparison.py, which never
had a shap package dependency itself (only explainer.py and plots.py
did) -- moved here, unchanged, so src/explainability/ can be deleted
entirely once SHAP is no longer part of this project, without breaking
the feature-importance pipeline's cross-reference against this
project's real correlation and Granger causality results.
"""

import sys
import logging

import pandas as pd

from src import config
from src.exception import CustomException

logger = logging.getLogger(__name__)

CAUSALITY_TABLES_DIR = config.RESULTS_DIR / "causality_analysis" / "tables"

# The correlation table contains both Pearson and Spearman rows for every
# variable pair -- Pearson is this project's primary reported linear-
# correlation statistic elsewhere, so it's the default here too.
DEFAULT_CORRELATION_METHOD = "Pearson"


def load_correlation_significance() -> pd.DataFrame:
    """Loads Notebook 05's real, saved correlation results -- not recomputed."""
    path = CAUSALITY_TABLES_DIR / "correlation_pairwise_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 05_correlation_and_granger.ipynb first.")
    return pd.read_csv(path)


def load_granger_significance() -> pd.DataFrame:
    """Loads Notebook 05's real, saved Granger causality results -- not recomputed."""
    path = CAUSALITY_TABLES_DIR / "granger_causality_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run 05_correlation_and_granger.ipynb first.")
    return pd.read_csv(path)


def get_tested_variables(corr_df: pd.DataFrame, granger_df: pd.DataFrame) -> list[str]:
    """The real set of source variables Notebook 05 actually tested --
    read from the tables themselves, not hardcoded."""
    return sorted(
        set(corr_df["Variable A"]) | set(corr_df["Variable B"])
        | set(granger_df["Predictor"]) | set(granger_df["Target"])
    )


def match_feature_to_variable(feature_name: str, variable_names: list[str]) -> str | None:
    """Deterministic engineered-feature -> source-variable mapping, built
    from this project's actual feature-engineering naming convention:
    every engineered feature is named f"{source_variable}_{transform}",
    e.g. "Bitcoin_Search_Interest_lag_1".

    Anchored-prefix matching, checked longest-candidate-first. Encoded/
    autoencoder features and anything else with no matching source
    variable correctly return None.
    """
    for var in sorted(variable_names, key=len, reverse=True):
        if feature_name == var or feature_name.startswith(var + "_"):
            return var
    return None


def get_relevant_correlation(
    corr_df: pd.DataFrame, matched_var: str | None, asset: str, method: str = DEFAULT_CORRELATION_METHOD,
) -> float | None:
    """The correlation between matched_var and the specific asset's price
    target -- filtered by asset and method explicitly, not by row order.
    Returns None for a self-reference (matched_var is the asset's own
    price variable) or if no row matches."""
    if matched_var is None:
        return None
    target = f"{asset}_Price"
    if matched_var == target:
        return None

    match = corr_df[
        (corr_df["Method"] == method)
        & (
            ((corr_df["Variable A"] == matched_var) & (corr_df["Variable B"] == target))
            | ((corr_df["Variable A"] == target) & (corr_df["Variable B"] == matched_var))
        )
    ]
    if len(match) == 0:
        return None
    if len(match) > 1:
        logger.warning(
            "Ambiguous correlation match for var=%s asset=%s method=%s -- %d rows, using the first.",
            matched_var, asset, method, len(match),
        )
    return float(match.iloc[0]["Coefficient"])


def get_relevant_granger(granger_df: pd.DataFrame, matched_var: str | None, asset: str) -> bool | None:
    """Whether matched_var was found Granger-significant specifically for
    THIS asset's price target -- filtered by both Predictor and Target.
    Returns None if this (variable, asset) pair was never tested at all --
    distinct from False (tested, not significant)."""
    if matched_var is None:
        return None
    target = f"{asset}_Price"
    match = granger_df[(granger_df["Predictor"] == matched_var) & (granger_df["Target"] == target)]
    if len(match) == 0:
        return None
    if len(match) > 1:
        logger.warning(
            "Ambiguous Granger match for predictor=%s target=%s -- %d rows, using the first.",
            matched_var, target, len(match),
        )
    return bool(match.iloc[0]["Significant (FDR-corrected)"])


def build_statistical_evidence_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """For every unique (asset, feature) pair in the permutation
    importance results, attaches the real correlation coefficient and
    Granger significance for that feature's matched source variable.
    Never invents a value -- None/NaN where a feature has no tested
    source variable or where that (variable, asset) pair was never
    tested."""
    try:
        corr_df = load_correlation_significance()
        granger_df = load_granger_significance()
        tested_variables = get_tested_variables(corr_df, granger_df)

        pairs = long_df[["asset", "feature"]].drop_duplicates()
        rows = []
        for _, r in pairs.iterrows():
            matched_var = match_feature_to_variable(r["feature"], tested_variables)
            corr_value = get_relevant_correlation(corr_df, matched_var, r["asset"])
            granger_sig = get_relevant_granger(granger_df, matched_var, r["asset"])
            rows.append({
                "asset": r["asset"], "feature": r["feature"], "matched_variable": matched_var,
                "correlation_r": corr_value, "granger_significant": granger_sig,
            })
        return pd.DataFrame(rows)
    except Exception as e:
        raise CustomException(e, sys)


def merge_with_statistical_evidence(long_df: pd.DataFrame) -> pd.DataFrame:
    """Merges the real permutation importance long-format results with
    the real statistical evidence table above."""
    evidence = build_statistical_evidence_table(long_df)
    return long_df.merge(evidence, on=["asset", "feature"], how="left")
