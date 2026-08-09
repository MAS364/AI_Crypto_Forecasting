"""Chronological splitting and predictor/target preparation for model development.

Extracted from 07_modeling.ipynb. No fixed 70/15/15 split was documented
in the methodology chapter, so it is defined here as the implemented
choice (matching the notebook), passed explicitly rather than hardcoded
as a hidden default, so it stays visible and easy to change from one
place if the methodology chapter is later revised.
"""

import pandas as pd

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def split_data(df: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a chronologically-sorted dataframe into train/validation/test, no shuffling.

    Asserts there is no overlap between partitions -- this is checked
    every time the function runs, not just once during development.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    assert train["Date"].max() < val["Date"].min(), "train/validation overlap"
    assert val["Date"].max() < test["Date"].min(), "validation/test overlap"

    return train, val, test


def get_X_y(df: pd.DataFrame, price_col: str, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature-set dataframe into predictors (X) and target (y).

    Drops Date (not a predictor), the price-level reference column
    (kept only for reconstructing price forecasts, never fed to a
    model), and the target itself. Everything else in the feature set
    becomes a predictor -- used identically by Random Forest, XGBoost,
    and LSTM. This includes the trailing-return feature at horizon > 1
    (added specifically for the Naive baseline) -- there is no leakage
    reason to withhold a legitimate backward-looking feature from the
    other models, so it is left available to them like any other
    engineered feature.
    """
    target_col = f"{price_col}_log_return_target_t_plus_{horizon}"
    level_col = f"{price_col}_level"
    X = df.drop(columns=["Date", level_col, target_col])
    y = df[target_col]
    return X, y
