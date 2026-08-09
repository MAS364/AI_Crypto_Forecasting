"""Random Forest and XGBoost tabular models.

Extracted from 07_modeling.ipynb. Both are trained on every column in a
feature set (primary, candidate, ablation, or combined) except Date,
the price-level reference column, and the target -- see
src.models.data_prep.get_X_y.

Random Forest is tuned with TimeSeriesSplit cross-validation inside the
training set only. XGBoost is tuned with early stopping against the
validation set. Both are then refit on train+validation combined using
the chosen configuration, before their single test-set evaluation --
the validation metrics reported come from a separate model fit on
training data alone, since the "final" refit model has already seen
validation data by that point and would give an artificially optimistic
validation score otherwise.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics import evaluate, rmse
from src.models.data_prep import get_X_y

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 42
CV_FOLDS = 3

RF_GRID = [
    {"n_estimators": 200, "max_depth": 3, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 4, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 3, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 2, "min_samples_split": 10, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"n_estimators": 800, "max_depth": 3, "min_samples_split": 15, "min_samples_leaf": 5, "max_features": "log2"},
    {"n_estimators": 300, "max_depth": 2, "min_samples_split": 20, "min_samples_leaf": 8, "max_features": "sqrt"},
    {"n_estimators": 300, "max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 500, "max_depth": 10, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": 0.5},
    {"n_estimators": 500, "max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt"},
]

XGB_GRID = [
    {"n_estimators": 500, "max_depth": 2, "learning_rate": 0.01, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 5.0},
    {"n_estimators": 500, "max_depth": 2, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0, "min_child_weight": 10, "reg_lambda": 10.0},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 5.0, "objective": "reg:pseudohubererror"},
    {"n_estimators": 500, "max_depth": 1, "learning_rate": 0.1, "subsample": 0.6, "colsample_bytree": 0.6, "min_child_weight": 10, "gamma": 0.5, "reg_lambda": 5.0},
    {"n_estimators": 500, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.6, "colsample_bytree": 0.8, "min_child_weight": 10, "gamma": 1.0, "reg_alpha": 1.0, "reg_lambda": 5.0},
]


def tune_rf(X_train: pd.DataFrame, y_train: pd.Series, grid: list = None, cv_folds: int = CV_FOLDS) -> tuple:
    """Pick the best Random Forest config using TimeSeriesSplit CV on the training set only."""
    grid = grid if grid is not None else RF_GRID
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    best_params, best_score = None, np.inf

    for params in grid:
        scores = []
        for tr_idx, va_idx in tscv.split(X_train):
            m = RandomForestRegressor(**params, random_state=SEED)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            scores.append(rmse(y_train.iloc[va_idx], m.predict(X_train.iloc[va_idx])))
        avg = np.mean(scores)
        if avg < best_score:
            best_score, best_params = avg, params

    return best_params, best_score


def run_random_forest(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, price_col: str,
                       horizon: int = 1, grid: list = None, cv_folds: int = None) -> dict:
    """`grid` and `cv_folds` default to this module's own RF_GRID/CV_FOLDS if not
    supplied -- callers (such as a notebook that wants its own grid definition
    to be the visible, authoritative source of the hyperparameter search space)
    can override either explicitly."""
    level = f"{price_col}_level"

    X_train, y_train = get_X_y(train, price_col, horizon=horizon)
    X_val, y_val = get_X_y(val, price_col, horizon=horizon)
    X_test, y_test = get_X_y(test, price_col, horizon=horizon)

    best_params, cv_score = tune_rf(X_train, y_train, grid=grid, cv_folds=cv_folds if cv_folds is not None else CV_FOLDS)
    print(f"  rf best params: {best_params} (cv rmse={cv_score:.5f})")

    # Honest validation score -- a model fit on training data alone,
    # since the final refit model below has already seen validation data.
    val_model = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=4)
    val_model.fit(X_train, y_train)
    val_preds = val_model.predict(X_val)

    # Final model for test -- train+val combined.
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    final_model = RandomForestRegressor(**best_params, random_state=SEED, n_jobs=4)
    final_model.fit(X_tv, y_tv)
    test_preds = final_model.predict(X_test)

    preds = {
        "validation": pd.DataFrame({"event_date": val["Date"], "actual": y_val.values, "predicted": val_preds}),
        "test": pd.DataFrame({"event_date": test["Date"], "actual": y_test.values, "predicted": test_preds}),
    }
    metrics = {
        "validation": evaluate(val[level], y_val, val_preds),
        "test": evaluate(test[level], y_test, test_preds),
    }
    importance = pd.DataFrame({"feature": X_tv.columns, "importance": final_model.feature_importances_}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    return {"predictions": preds, "metrics": metrics, "model": final_model, "importance": importance, "params": best_params}


def run_xgboost(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, price_col: str,
                 horizon: int = 1, grid: list = None) -> dict:
    """`grid` defaults to this module's own XGB_GRID if not supplied."""
    if not HAS_XGB:
        print("  skipping xgboost, not installed")
        return None

    grid = grid if grid is not None else XGB_GRID
    level = f"{price_col}_level"
    X_train, y_train = get_X_y(train, price_col, horizon=horizon)
    X_val, y_val = get_X_y(val, price_col, horizon=horizon)
    X_test, y_test = get_X_y(test, price_col, horizon=horizon)

    best_params, best_rmse, best_n_trees, best_val_preds = None, np.inf, None, None

    for params in grid:
        m = XGBRegressor(**params, early_stopping_rounds=20, random_state=SEED, n_jobs=4)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = m.predict(X_val)
        score = rmse(y_val, preds)
        if score < best_rmse:
            best_rmse, best_params, best_n_trees, best_val_preds = score, params, m.best_iteration + 1, preds

    print(f"  xgb best params: {best_params}, stopped at {best_n_trees} trees (val rmse={best_rmse:.5f})")

    # No further early stopping here: the training/validation data used
    # to determine the stopping point above is now merged into a single
    # combined training set, so there is nothing left to validate against.
    final_params = {**best_params, "n_estimators": best_n_trees}
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    final_model = XGBRegressor(**final_params, random_state=SEED, n_jobs=4)
    final_model.fit(X_tv, y_tv)
    test_preds = final_model.predict(X_test)

    preds = {
        "validation": pd.DataFrame({"event_date": val["Date"], "actual": y_val.values, "predicted": best_val_preds}),
        "test": pd.DataFrame({"event_date": test["Date"], "actual": y_test.values, "predicted": test_preds}),
    }
    metrics = {
        "validation": evaluate(val[level], y_val, best_val_preds),
        "test": evaluate(test[level], y_test, test_preds),
    }
    importance = pd.DataFrame({"feature": X_tv.columns, "importance": final_model.feature_importances_}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    return {"predictions": preds, "metrics": metrics, "model": final_model, "importance": importance, "params": final_params}
