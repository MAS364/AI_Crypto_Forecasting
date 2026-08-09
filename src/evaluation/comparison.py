"""Model comparison and quality-control checks -- shared logic for
08_evaluation.ipynb.

This module previously predated the multi-horizon rebuild and the
autoencoder-encoded representation branch (it referenced a
`predicted`/`actual`/`residual` schema that no longer matches the real
saved data, and had no horizon or representation awareness at all).
Rebuilt here to match the actual current schema produced by
07_modeling.ipynb / src/models/pipeline.py, verified directly against a
real saved predictions file rather than assumed.

Notebook 08 should call these functions against the saved CSVs in
results/modeling/ rather than reimplementing comparison or QC logic --
matching the same reuse pattern already established for feature
engineering (Notebook 06) and modelling (Notebook 07).
"""

import numpy as np
import pandas as pd


def build_validation_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Validation-only summary table, sorted by RMSE within each
    horizon/asset/feature-set/representation group.

    Not a final model comparison -- Notebook 08 should use the test split
    for that; this is the same validation-only view already produced
    inline in Notebook 07.
    """
    return metrics_df[metrics_df["split"] == "validation"][
        ["horizon", "asset", "feature_set", "model", "representation",
         "RMSE_returns", "MAE_returns", "R2_returns", "Hit_Rate_%", "MAPE_price_%", "seconds", "status"]
    ].sort_values(["horizon", "asset", "feature_set", "representation", "RMSE_returns"])


def build_master_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """The primary test-set comparison table: every horizon, asset, feature
    configuration, model, and representation, with every metric. This is
    the same table Notebook 07 already builds and saves as
    master_comparison_all_dimensions.csv -- provided here so Notebook 08
    can either reload that file directly or rebuild the same view from
    metrics_df without duplicating the column selection logic."""
    return metrics_df[metrics_df["split"] == "test"][[
        "horizon", "asset", "feature_set", "model", "representation",
        "RMSE_returns", "MAE_returns", "R2_returns", "DirAcc_%", "Hit_Rate_%",
        "RMSE_price", "MAE_price", "MAPE_price_%",
    ]].sort_values(["horizon", "asset", "feature_set", "representation", "RMSE_returns"]).reset_index(drop=True)


def run_qc_checks(all_preds: pd.DataFrame, splits: dict, log_df: pd.DataFrame) -> dict:
    """Run the standard set of quality-control checks on a completed model run.

    Returns a dict of check-name -> passed (bool), and prints a plain
    pass/fail line for each, matching what Notebook 07 already prints.
    Raises an AssertionError on the first genuine failure -- these are
    checks that should never fail if the pipeline is working correctly,
    not soft warnings.

    `splits` should be the same {(horizon, asset, feature_set): (train, val,
    test)} dict already built during modelling.
    """
    results = {}

    dupes = all_preds.groupby(["horizon", "asset", "model", "split", "feature_set", "representation"])["event_date"].apply(
        lambda d: d.duplicated().any()
    )
    assert not dupes.any(), "duplicate dates somewhere"
    results["no_duplicate_dates"] = True
    print("no duplicate dates")

    assert all_preds["predicted_return"].isnull().sum() == 0
    assert all_preds["predicted_price"].isnull().sum() == 0
    results["no_missing_predictions"] = True
    print("no missing predictions")

    value_cols = ["actual_return", "predicted_return", "return_residual",
                  "actual_price", "predicted_price", "price_residual"]
    assert np.isfinite(all_preds[value_cols].values).all()
    results["all_values_finite"] = True
    print("all return-space and price-space values finite")

    for key, (tr, va, te) in splits.items():
        assert tr["Date"].max() < va["Date"].min()
        assert va["Date"].max() < te["Date"].min()
    results["train_val_test_separation"] = True
    print("train/val/test separation holds everywhere")

    expected_columns = {
        "horizon", "event_date", "asset", "model", "split", "feature_set", "representation",
        "price_level", "actual_return", "predicted_return", "return_residual",
        "actual_price", "predicted_price", "price_residual",
    }
    assert set(all_preds.columns) == expected_columns
    results["schema_consistent"] = True
    print("output schema consistent across all models (return-space + price-space)")

    failed = log_df[log_df["status"] != "success"]
    results["all_runs_succeeded"] = len(failed) == 0
    if len(failed):
        print(f"\n{len(failed)} runs failed:")
        print(failed)
    else:
        print("everything ran fine")

    return results
