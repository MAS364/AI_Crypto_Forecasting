"""Model and test-data loading for permutation feature importance.

Extracted from src/explainability/utils.py, which this project's feature
importance work originally reused directly -- that file itself never
imported the shap package (only src/explainability/explainer.py and
plots.py did), but living inside src/explainability/ meant this module
stayed dependent on that folder for no real reason once SHAP was
removed from the project entirely. Moved here so src/explainability/ can
be deleted without touching anything this module needs.

Behavior is unchanged from the original: loads an already-fitted, saved
model plus its real train/test predictor matrices. Never retrains
anything.
"""

import sys
from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import config
from src.exception import CustomException
from src.evaluation import loaders
from src.models.data_prep import split_data, get_X_y
from src.models.feature_naming import FEATURE_FILE_TEMPLATES, _horizon_suffix

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ModelType = Literal["tree"]


def detect_model_type(model) -> ModelType:
    """Random Forest / XGBoost only -- permutation importance in this
    project is deliberately scoped to the two tree-based models (see
    run_feature_importance_grid.py's module docstring for why LSTM,
    ARIMA, and Naive are excluded)."""
    try:
        if isinstance(model, RandomForestRegressor):
            return "tree"
        if HAS_XGB and isinstance(model, XGBRegressor):
            return "tree"
        raise ValueError(
            f"Unsupported model type for permutation importance: {type(model).__name__}. "
            "This project's feature importance analysis is scoped to Random Forest and "
            "XGBoost only."
        )
    except Exception as e:
        raise CustomException(e, sys)


def load_model_and_data(
    horizon: int, asset: str, feature_set_key: str, model_name: str,
    representation: str = "Original",
) -> dict:
    """Loads one saved, fitted model plus its real train/test predictor
    matrices -- never retrains anything. Same chronological split every
    other part of this project uses (src.models.data_prep), so the test
    set here is identical to the one the model was actually evaluated on.
    """
    try:
        price_col = config.ASSET_VARIABLES[asset]["price"] if hasattr(config, "ASSET_VARIABLES") else f"{asset}_Price"

        suffix = _horizon_suffix(horizon)
        filename = FEATURE_FILE_TEMPLATES[feature_set_key].format(a=asset.lower(), s=suffix)
        df = pd.read_csv(config.FEATURES_DIR / filename, parse_dates=["Date"])
        train, _, test = split_data(df)

        X_train, y_train = get_X_y(train, price_col, horizon=horizon)
        X_test, y_test = get_X_y(test, price_col, horizon=horizon)

        model = loaders.load_trained_model(horizon, asset, feature_set_key, model_name)

        return {
            "model": model, "model_type": detect_model_type(model),
            "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
            "feature_names": list(X_train.columns),
            "horizon": horizon, "asset": asset, "feature_set_key": feature_set_key,
            "model_name": model_name, "representation": representation,
        }
    except FileNotFoundError:
        raise
    except Exception as e:
        raise CustomException(e, sys)
