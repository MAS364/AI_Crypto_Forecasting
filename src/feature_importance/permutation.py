"""Permutation Feature Importance (PFI) -- the primary, common
feature-importance technique for comparing Random Forest and XGBoost in
this project.

Model-agnostic by construction: sklearn.inspection.permutation_importance
only ever calls model.predict() repeatedly on a shuffled copy of one
column at a time. It never reads a model's internal tree structure the
way SHAP's TreeExplainer does -- which is the operation that caused
repeated, reproducible kernel crashes in this project's environment (see
src/explainability/'s module docstrings). PFI is not a workaround for
that instability; it is a legitimate, standard, model-agnostic technique
in its own right, and happens to also be computationally stable here.

No SHAP dependency anywhere in this module.
"""

import sys
import logging

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.exception import CustomException
from src.evaluation.metrics import rmse
from src.feature_importance.model_loading import load_model_and_data  # extracted from src.explainability, no SHAP dependency

logger = logging.getLogger(__name__)

SEED = 42  # matches src.models.tree_models.SEED -- same reproducibility discipline
N_REPEATS = 8  # within the requested 5-10 range


def calculate_permutation_importance(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    n_repeats: int = N_REPEATS, random_state: int = SEED,
) -> pd.DataFrame:
    """Runs sklearn's real permutation_importance against an already-
    fitted model and the real chronological test set -- the model is
    never retrained or refit at any point. Uses neg_root_mean_squared_error
    as the scoring function so permutation_importance's own sign
    convention (higher = better) is flipped to match this project's
    RMSE-increase convention (Permuted RMSE - Baseline RMSE); sklearn's
    importances_mean is already `baseline_score - permuted_score` under
    a scorer where higher is better, so negating a neg-RMSE scorer's
    output gives exactly `permuted_RMSE - baseline_RMSE` as specified.

    Negative importances are NOT clipped -- retained as real signal
    (permuting the feature by chance improved this particular metric on
    this particular test set; see this project's dissertation-facing
    documentation on interpreting negative PFI values).
    """
    try:
        baseline_pred = model.predict(X_test)
        baseline_rmse = rmse(y_test, baseline_pred)

        result = permutation_importance(
            model, X_test, y_test,
            scoring="neg_root_mean_squared_error",
            n_repeats=n_repeats, random_state=random_state, n_jobs=1,
        )
        # importances_mean here is already (baseline_neg_rmse - permuted_neg_rmse)
        # = (-baseline_rmse) - (-permuted_rmse) = permuted_rmse - baseline_rmse,
        # exactly the Permuted RMSE - Baseline RMSE definition requested.
        mean_importance = result.importances_mean
        std_importance = result.importances_std

        df = pd.DataFrame({
            "feature": X_test.columns, "mean_importance": mean_importance,
            "std_importance": std_importance, "baseline_rmse": baseline_rmse,
        })
        df["rank"] = df["mean_importance"].rank(ascending=False, method="min").astype(int)
        df["n_repeats"] = n_repeats
        return df.sort_values("rank").reset_index(drop=True)
    except Exception as e:
        raise CustomException(e, sys)


def run_feature_importance_experiment(
    horizon: int, asset: str, feature_set_key: str, model_name: str,
    representation: str = "Original", n_repeats: int = N_REPEATS,
) -> pd.DataFrame:
    """One full experiment: load the real saved model + real chronological
    test set (never retrains), compute real permutation importance for
    every feature. Returns a long-format DataFrame with full
    asset/horizon/config/model labeling, ready to concatenate across all
    48 experiments without ever losing track of which problem a row
    belongs to (same config-safety discipline as
    src.explainability.comparison's build_comparison_table)."""
    loaded = load_model_and_data(
        horizon=horizon, asset=asset, feature_set_key=feature_set_key,
        model_name=model_name, representation=representation,
    )
    pfi = calculate_permutation_importance(
        loaded["model"], loaded["X_test"], loaded["y_test"], n_repeats=n_repeats,
    )
    pfi["asset"] = asset
    pfi["horizon"] = horizon
    pfi["feature_set_key"] = feature_set_key
    pfi["model_name"] = model_name
    pfi["representation"] = representation
    return pfi
