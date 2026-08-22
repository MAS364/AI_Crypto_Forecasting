"""A defensible Original-vs-Autoencoder-Encoded comparison.

Raw permutation-importance magnitudes are NOT directly comparable across
representations: the Original branch spreads importance across ~12-17
named engineered features per configuration, while the Autoencoder-
Encoded branch spreads it across exactly 8 unlabelled latent dimensions.
A bigger single number in one branch is at least partly an artefact of
"how many features were there to share importance with", not evidence
that one representation is more informative than the other. Simply
concatenating the two branches' long-format tables (which is all an
earlier version of this analysis did -- see full_feature_dictionary.csv's
companion script) does not fix this and was flagged as a defect in the
Phase 1 verification of this analysis.

This module instead reports three things that ARE legitimately
comparable across representations, kept clearly separate rather than
blended into one misleading number:

1. Predictive performance (test RMSE) -- the one number that is
   genuinely on the same scale regardless of representation, since both
   branches predict the same target on the same test rows. Read directly
   from each branch's own permutation-importance run (baseline_rmse is
   computed once per experiment, before any permutation, as part of
   calculate_permutation_importance()) -- not recomputed here.
2. Within-configuration normalised importance -- each feature's mean
   importance divided by that configuration's own largest absolute mean
   importance, giving every configuration a common [-1, 1] scale. This
   is a legitimate "how important was this relative to the most
   important feature in the same experiment" comparison; it is still
   not a claim that "0.5 normalised" means the same real-world thing in
   both branches.
3. Importance concentration -- what share of the whole configuration's
   total absolute importance sits in the single top-ranked feature. A
   high share means one feature dominates; a low share means importance
   is spread out. This is scale-free by construction, so it is one of
   the few statistics that really is apples-to-apples across a
   17-feature Original configuration and an 8-dimension Autoencoder-
   Encoded one.
"""

import sys
import logging

import numpy as np
import pandas as pd

from src.exception import CustomException

logger = logging.getLogger(__name__)

CONFIG_COLUMNS = ["asset", "horizon", "feature_set_key", "model_name"]


def _summarize_one_branch(long_df: pd.DataFrame, representation_label: str) -> pd.DataFrame:
    columns = [
        "asset", "horizon", "feature_set_key", "model_name", "representation", "n_features",
        "baseline_rmse", "top_feature", "top_feature_mean_importance",
        "top_feature_normalized_importance", "top_feature_importance_share",
    ]
    if long_df.empty:
        # An empty branch (e.g. the Autoencoder-Encoded grid hasn't been
        # run yet) must still return the right schema with zero rows, so
        # the outer merge below can find every expected column -- not an
        # empty DataFrame with no columns at all, which would make the
        # merge raise a KeyError instead of reporting the branch as
        # missing the way this function's docstring promises.
        return pd.DataFrame(columns=columns)

    rows = []
    for keys, group in long_df.groupby(CONFIG_COLUMNS):
        asset, horizon, feature_set_key, model_name = keys
        baseline_rmse = group["baseline_rmse"].iloc[0]  # identical across every row of one experiment
        abs_importance = group["mean_importance"].abs()
        total_abs_importance = abs_importance.sum()

        top_row = group.loc[abs_importance.idxmax()]
        max_abs_importance = abs(top_row["mean_importance"])

        rows.append({
            "asset": asset, "horizon": horizon, "feature_set_key": feature_set_key, "model_name": model_name,
            "representation": representation_label,
            "n_features": len(group),
            "baseline_rmse": baseline_rmse,
            "top_feature": top_row["feature"],
            "top_feature_mean_importance": top_row["mean_importance"],
            "top_feature_normalized_importance": (
                top_row["mean_importance"] / max_abs_importance if max_abs_importance > 0 else np.nan
            ),
            "top_feature_importance_share": (
                max_abs_importance / total_abs_importance if total_abs_importance > 0 else np.nan
            ),
        })
    return pd.DataFrame(rows, columns=columns)


def build_representation_comparison_summary(original_long: pd.DataFrame, encoded_long: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, horizon, feature_set_key, model_name) with the
    Original and Autoencoder-Encoded branches side by side as separate
    columns -- test RMSE, top feature, and the two scale-free
    concentration statistics described in this module's docstring.
    Only ever compares a configuration that has both branches present;
    a configuration missing one branch is reported as such (NaN on that
    side), never dropped silently.
    """
    try:
        original_summary = _summarize_one_branch(original_long, "Original")
        encoded_summary = _summarize_one_branch(encoded_long, "Autoencoder-Encoded")

        merged = original_summary.merge(
            encoded_summary, on=CONFIG_COLUMNS, how="outer", suffixes=("_original", "_encoded"),
        )
        merged["rmse_difference_encoded_minus_original"] = (
            merged["baseline_rmse_encoded"] - merged["baseline_rmse_original"]
        )
        merged["better_test_rmse_representation"] = np.where(
            merged["baseline_rmse_original"].isna() | merged["baseline_rmse_encoded"].isna(), "n/a (one branch missing)",
            np.where(merged["baseline_rmse_original"] <= merged["baseline_rmse_encoded"], "Original", "Autoencoder-Encoded"),
        )
        return merged.sort_values(CONFIG_COLUMNS).reset_index(drop=True)
    except Exception as e:
        raise CustomException(e, sys)
