"""Aggregation and cross-model comparison functions for permutation
feature importance results -- reusable across every (asset, horizon,
config) combination, not duplicated per-experiment code.
"""

import sys
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.exception import CustomException
from src.feature_importance.grouping import assign_feature_group

logger = logging.getLogger(__name__)

CONFIG_COLUMNS = ["asset", "horizon", "feature_set_key", "representation"]


def aggregate_feature_importance(long_df: pd.DataFrame) -> pd.DataFrame:
    """Adds feature_group to every row -- the single place group
    assignment happens, so it's never duplicated or drifted between
    different aggregation functions."""
    out = long_df.copy()
    out["feature_group"] = out["feature"].apply(assign_feature_group)
    return out


def compare_model_rankings(long_df: pd.DataFrame) -> pd.DataFrame:
    """TABLE 4 / TABLE 9: For every (asset, horizon, feature_set_key,
    representation) configuration where BOTH Random Forest and XGBoost
    were run, builds a wide RF-vs-XGBoost comparison with importance,
    rank, and computes Spearman rank correlation + top-5/top-10 overlap
    for that configuration.

    Only ever compares rows sharing an identical configuration -- same
    apples-to-apples discipline as the earlier SHAP work's
    build_comparison_table, just for permutation importance instead of
    SHAP.
    """
    try:
        pivot = long_df.pivot_table(
            index=CONFIG_COLUMNS + ["feature"], columns="model_name",
            values=["mean_importance", "rank"],
        )
        pivot.columns = [f"{val}_{model}" for val, model in pivot.columns]
        pivot = pivot.reset_index()

        rf_imp_col, xgb_imp_col = "mean_importance_Random Forest", "mean_importance_XGBoost"
        rf_rank_col, xgb_rank_col = "rank_Random Forest", "rank_XGBoost"
        if rf_imp_col not in pivot.columns or xgb_imp_col not in pivot.columns:
            raise ValueError(
                "compare_model_rankings requires both Random Forest and XGBoost results "
                "in long_df -- got model_name values: " + str(long_df["model_name"].unique())
            )

        pivot["abs_difference"] = (pivot[rf_imp_col] - pivot[xgb_imp_col]).abs()
        pivot["both_available"] = pivot[[rf_imp_col, xgb_imp_col]].notna().all(axis=1)

        return pivot.sort_values(CONFIG_COLUMNS + ["abs_difference"], ascending=[True, True, True, True, False])
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(e, sys)


def compute_rank_agreement_summary(comparison_df: pd.DataFrame, top_n: tuple = (5, 10)) -> pd.DataFrame:
    """TABLE 9: one row per configuration -- Spearman rank correlation
    between RF and XGBoost's full importance rankings, plus top-5 and
    top-10 overlap counts. Does not force a Spearman value when a
    configuration has fewer usable features than would make the
    statistic meaningful (fewer than 3 shared, non-NaN ranks)."""
    rows = []
    for keys, group in comparison_df.groupby(CONFIG_COLUMNS):
        both = group[group["both_available"]]
        row = dict(zip(CONFIG_COLUMNS, keys))
        row["n_features_compared"] = len(both)

        if len(both) >= 3:
            rho, pval = spearmanr(both["rank_Random Forest"], both["rank_XGBoost"])
            row["spearman_rho"] = rho
            row["spearman_pvalue"] = pval
        else:
            row["spearman_rho"] = np.nan
            row["spearman_pvalue"] = np.nan
            logger.info("Skipping Spearman for %s -- only %d shared features", keys, len(both))

        for n in top_n:
            rf_top = set(both.nsmallest(n, "rank_Random Forest")["feature"])
            xgb_top = set(both.nsmallest(n, "rank_XGBoost")["feature"])
            row[f"top_{n}_overlap"] = len(rf_top & xgb_top)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(CONFIG_COLUMNS).reset_index(drop=True)


def calculate_feature_group_importance(long_df: pd.DataFrame) -> pd.DataFrame:
    """TABLE 5: Feature-group-level aggregation, per (asset, horizon,
    feature_set_key, representation, model_name, feature_group).
    Reports both the MEAN importance per feature in the group (fair
    comparison across groups with different feature counts) and the
    SUM (total group contribution) side by side, with the feature count
    shown explicitly -- so a reader can see whether a high group sum is
    driven by many weak features or few strong ones, rather than
    blindly summing without that context.
    """
    try:
        grouped = aggregate_feature_importance(long_df)
        agg = grouped.groupby(CONFIG_COLUMNS + ["model_name", "feature_group"]).agg(
            n_features=("feature", "count"),
            mean_importance=("mean_importance", "mean"),
            total_importance=("mean_importance", "sum"),
        ).reset_index()
        agg["rank_within_config_model"] = agg.groupby(CONFIG_COLUMNS + ["model_name"])["mean_importance"] \
            .rank(ascending=False, method="min").astype(int)
        return agg.sort_values(CONFIG_COLUMNS + ["model_name", "rank_within_config_model"])
    except Exception as e:
        raise CustomException(e, sys)


def create_importance_summary(long_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """TABLE 2 / TABLE 3: top-N features per (asset, horizon,
    feature_set_key, representation, model_name) -- the reusable builder
    behind the Random Forest and XGBoost top-feature tables."""
    grouped = aggregate_feature_importance(long_df)
    return (
        grouped.sort_values(CONFIG_COLUMNS + ["model_name", "rank"])
        .groupby(CONFIG_COLUMNS + ["model_name"])
        .head(top_n)
        .reset_index(drop=True)
    )
