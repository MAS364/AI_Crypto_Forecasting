"""CSV upload inference pipeline: CSVLoader -> Validator -> Preprocessor ->
FeatureEngineer -> ModelLoader -> Predictor -> ReportGenerator.

Reuses the project's existing preprocessing validation functions
(src.preprocessing.validation) and feature engineering
(src.analysis.feature_engineering) directly -- this module orchestrates
them for a single user-uploaded CSV, it does not reimplement any of the
underlying logic. Never retrains or refits anything.
"""

import io
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.exception import CustomException
from src.preprocessing.validation import check_data_types, check_variable_ranges
from src.preprocessing.encoding import encode_sentiment
from src.analysis import feature_engineering as fe
from src.pipeline import run_predict, FEATURE_SET_PARAMS

logger = logging.getLogger(__name__)

# Minimum rows needed regardless of model: 30-day rolling statistics are
# the longest-lookback feature in this project (see
# src/analysis/feature_engineering.py's create_rolling_features).
MIN_ROWS_FOR_FEATURES = 30

REQUIRED_COLUMNS = {
    "Date": "date",
    "{asset}_Price": "numeric, the closing price",
    "{asset}_Market_Cap": "numeric, market capitalisation",
    "{asset}_Volume": "numeric, trading volume",
}
OPTIONAL_COLUMNS = {
    "{asset}_Search_Interest": "numeric, Google Trends search interest (0-100)",
    "Fear_Greed_Index": "numeric, 0-100",
    "Market_Sentiment": "text, one of: Extreme Fear, Fear, Neutral, Greed, Extreme Greed "
                        "-- required alongside Fear_Greed_Index (the ablation feature set "
                        "needs both the numeric score and this ordinal-encodable category)",
}


class CSVValidationError(Exception):
    """Raised with a list of human-readable problems -- never a raw
    exception string, matching the project's error-handling discipline."""
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Loads an uploaded CSV -- format and parse errors surface here as a
    clear CSVValidationError, not a raw pandas traceback."""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise CSVValidationError([f"Could not parse file as CSV: {e}"])
    if df.empty:
        raise CSVValidationError(["The uploaded file has no rows."])
    return df


def validate_csv(df: pd.DataFrame, asset: str, horizon: int, model_name: str) -> None:
    """Runs every check specified: format (handled by load_csv), required
    columns, dtypes, missing values, duplicate dates, date ordering,
    minimum row count for the selected model/horizon, numerical ranges.

    Raises CSVValidationError listing every problem found at once, not
    just the first -- so a user fixing their CSV doesn't have to
    re-upload once per individual error.
    """
    problems = []

    required = [f"{asset}_Price", f"{asset}_Market_Cap", f"{asset}_Volume", "Date"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing required column(s): {', '.join(missing_cols)}")
        raise CSVValidationError(problems)  # can't check anything else without these

    if df["Date"].isnull().any():
        problems.append("Date column has missing values.")
    else:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
        except Exception:
            problems.append("Date column could not be parsed -- use YYYY-MM-DD format.")

    if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        n_duplicates = df["Date"].duplicated().sum()
        if n_duplicates:
            problems.append(f"{n_duplicates} duplicate date(s) found.")
        if not df["Date"].is_monotonic_increasing:
            problems.append("Dates are not in chronological order -- sort ascending before uploading.")

    numeric_required = [f"{asset}_Price", f"{asset}_Market_Cap", f"{asset}_Volume"]
    for col in numeric_required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"'{col}' must be numeric.")
        elif (df[col] < 0).any():
            problems.append(f"'{col}' has negative value(s) -- prices, market cap, and volume cannot be negative.")

    for col in [f"{asset}_Search_Interest"]:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                problems.append(f"'{col}' must be numeric.")
            elif not df[col].between(0, 100).all():
                problems.append(f"'{col}' must be between 0 and 100.")

    if "Fear_Greed_Index" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["Fear_Greed_Index"]):
            problems.append("'Fear_Greed_Index' must be numeric.")
        elif not df["Fear_Greed_Index"].between(0, 100).all():
            problems.append("'Fear_Greed_Index' must be between 0 and 100.")
        if "Market_Sentiment" not in df.columns:
            problems.append(
                "'Fear_Greed_Index' was provided but 'Market_Sentiment' wasn't -- both are "
                "needed together (one numeric, one categorical) for the Fear & Greed feature set."
            )
        else:
            from src import config as _cfg
            unknown = set(df["Market_Sentiment"].dropna()) - set(_cfg.SENTIMENT_MAPPING)
            if unknown:
                problems.append(
                    f"'Market_Sentiment' has unrecognised categories: {sorted(unknown)} -- "
                    f"must be one of {list(_cfg.SENTIMENT_MAPPING)}."
                )

    if model_name == "Naive":
        problems.append("Naive has no trainable parameters and no saved model to load for this workflow.")

    # ARIMA fits directly on the return series and doesn't need 30-day
    # rolling statistics -- a much lower bar than the other three models.
    min_rows = 11 if model_name == "ARIMA" else MIN_ROWS_FOR_FEATURES
    if len(df) < min_rows:
        reason = "a usable return series" if model_name == "ARIMA" else "30-day rolling features"
        problems.append(f"Need at least {min_rows} rows of history ({reason} require it) -- got {len(df)}.")

    if problems:
        raise CSVValidationError(problems)


def run_arima_inference(df: pd.DataFrame, asset: str, horizon: int, feature_set_key: str = "primary") -> dict:
    """ARIMA-specific inference path -- NOT a "load a saved model" flow
    like Random Forest/XGBoost/LSTM. ARIMA has no single reusable fitted
    object (see src/models/arima_model.py's docstrings -- it refits at
    every rolling step by design). Instead: loads the already-selected
    (p, d, q) order for this asset/horizon from real saved metadata, then
    performs ONE real statistical fit using that fixed order on the
    uploaded price history -- the exact same fit arima_rolling() performs
    at each of its rolling steps, just run once here on new data. No new
    order-selection, no tuning -- the order was already decided during
    model development and is being reapplied, not rediscovered.
    """
    import numpy as np
    import warnings
    from statsmodels.tsa.arima.model import ARIMA
    from src.evaluation import loaders

    price_col = f"{asset}_Price"
    prices = df[price_col].reset_index(drop=True)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 10:
        raise CSVValidationError([f"Need at least 11 rows to compute a usable return series for ARIMA -- got {len(prices)}."])

    meta = loaders.load_model_metadata(horizon, asset, feature_set_key, "ARIMA")
    order = tuple(meta["selected_order"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(log_returns, order=order).fit()
        forecast_steps = model.forecast(steps=horizon)

    cumulative_return = float(forecast_steps.sum()) if horizon > 1 else float(forecast_steps.iloc[0])
    price_level = float(prices.iloc[-1])
    predicted_price = price_level * np.exp(cumulative_return)

    return {
        "horizon": horizon, "asset": asset, "model": "ARIMA",
        "feature_set_key": feature_set_key, "feature_set": feature_set_key,
        "representation": "Original",
        "input_price_level": price_level,
        "predicted_return": cumulative_return,
        "predicted_price": predicted_price,
        "arima_order": list(order),
        "inference_note": (
            "ARIMA has no saved fitted model -- this fit a fresh ARIMA(%d,%d,%d) on your "
            "uploaded price history, using the order already selected during model "
            "development (not re-selected here). This is one real statistical fit, "
            "not model training or tuning." % order
        ),
    }


def _determine_feature_set_key(df: pd.DataFrame, asset: str) -> str:
    """Auto-detects which of the 4 feature configurations the upload
    actually supports, based on which optional columns are present --
    the richest configuration the data allows, not a fixed default."""
    has_trends = f"{asset}_Search_Interest" in df.columns
    has_sentiment = "Fear_Greed_Index" in df.columns and "Market_Sentiment" in df.columns
    if has_trends and has_sentiment:
        return "combined"
    if has_trends:
        return "candidate"
    if has_sentiment:
        return "ablation"
    return "primary"


# Which optional columns each feature configuration actually needs --
# used to validate an explicit user override (e.g. requesting "combined"
# on an upload that never provided Fear & Greed columns should be a
# clear validation error, not a silent fall-back or a confusing crash
# deeper in feature engineering).
FEATURE_SET_REQUIREMENTS = {
    "primary": (False, False),    # (needs_trends, needs_sentiment)
    "candidate": (True, False),
    "ablation": (False, True),
    "combined": (True, True),
}


def _validate_requested_feature_set(df: pd.DataFrame, asset: str, feature_set_key: str) -> None:
    if feature_set_key not in FEATURE_SET_REQUIREMENTS:
        raise CSVValidationError([f"Unknown feature configuration '{feature_set_key}'."])
    needs_trends, needs_sentiment = FEATURE_SET_REQUIREMENTS[feature_set_key]
    has_trends = f"{asset}_Search_Interest" in df.columns
    has_sentiment = "Fear_Greed_Index" in df.columns and "Market_Sentiment" in df.columns
    problems = []
    if needs_trends and not has_trends:
        problems.append(
            f"The requested feature configuration needs '{asset}_Search_Interest', "
            "which isn't in your upload -- add it, or choose a configuration that doesn't need it."
        )
    if needs_sentiment and not has_sentiment:
        problems.append(
            "The requested feature configuration needs 'Fear_Greed_Index' and "
            "'Market_Sentiment' together, and at least one is missing from your upload."
        )
    if problems:
        raise CSVValidationError(problems)


def _check_lstm_input_for_nan(feature_set: pd.DataFrame, horizon: int, asset: str, feature_set_key: str) -> None:
    """Early, itemized check for missing/NaN values in the exact window
    LSTM needs -- raised as a proper CSVValidationError (fast, itemized,
    before any model gets loaded) instead of only being discovered deep
    inside run_predict() after paying the cost of loading TensorFlow and
    the model. LSTM has zero tolerance for NaN input -- unlike Random
    Forest/XGBoost, which route around missing values via their built-in
    split logic and don't need this check at all. See
    src/analysis/feature_engineering.py's create_volume_features()
    docstring: NaN here is very often a known, deliberately-preserved
    data anomaly (e.g. a flagged implausible-volume period), not a
    malformed upload -- the message below explains that distinction
    rather than just saying "your CSV is broken"."""
    from src.evaluation import loaders
    try:
        lstm_meta = loaders.load_model_metadata(horizon, asset, feature_set_key, "LSTM")
    except FileNotFoundError:
        return  # let run_predict() surface the real "no saved model" error, not this one

    window = lstm_meta["sequence_config"]["window"]
    expected_features = lstm_meta.get("feature_names")
    cols = [c for c in (expected_features or feature_set.columns) if c in feature_set.columns]
    if len(feature_set) < window:
        return  # let run_predict()'s row-count check produce that specific message instead

    tail = feature_set[cols].tail(window).reset_index(drop=True)
    nan_mask = tail.isna()
    if not nan_mask.any().any():
        return

    bad_cols = [c for c in cols if nan_mask[c].any()]
    bad_dates = []
    if "Date" in feature_set.columns:
        date_tail = feature_set["Date"].tail(window).reset_index(drop=True)
        bad_row = nan_mask.any(axis=1)
        bad_dates = [str(pd.Timestamp(d).date()) for d, is_bad in zip(date_tail, bad_row) if is_bad]

    raise CSVValidationError([
        f"LSTM needs {window} consecutive rows with no missing values, but column(s) "
        f"{bad_cols} have missing data"
        + (f" on {bad_dates}" if bad_dates else "")
        + ". This is often a known, deliberately-flagged data gap (e.g. an implausible-volume "
        "period set to missing rather than faked) that Random Forest and XGBoost can work "
        "around automatically, but LSTM cannot. Try Random Forest or XGBoost instead, a "
        "different horizon, or an upload whose most recent rows don't overlap the gap."
    ])


def preprocess_and_engineer(df: pd.DataFrame, asset: str, feature_set_key: str, horizon: int) -> pd.DataFrame:
    """Applies the same preprocessing and feature engineering used in
    training (src.preprocessing.encoding.encode_sentiment,
    src.analysis.feature_engineering.build_feature_set) -- never a
    separate, parallel implementation."""
    try:
        if "Market_Sentiment" in df.columns:
            df = encode_sentiment(df)
        include_external, include_ablation = FEATURE_SET_PARAMS[feature_set_key]
        feature_set = fe.build_feature_set(
            df, asset=asset, feature_set_name=feature_set_key,
            include_external=include_external, include_ablation=include_ablation,
            horizon=horizon,
        )
        return feature_set
    except Exception as e:
        raise CustomException(e, sys)


def run_csv_inference(
    file_bytes: bytes, asset: str, model_name: str, horizon: int,
    representation: str = "Original", requested_feature_set: Optional[str] = None,
) -> dict:
    """Full pipeline: load -> validate -> preprocess/engineer -> predict.
    Never retrains any model beyond ARIMA's one real, order-fixed
    statistical fit (see run_arima_inference's docstring). Returns a
    structured result ready for the API response and report generation.

    requested_feature_set: if given, overrides auto-detection -- must be
    one of the 4 real feature configuration keys, and the upload must
    actually contain the columns that configuration needs (checked via
    _validate_requested_feature_set). If omitted, falls back to the
    richest configuration the upload's columns support, same as before.
    Ignored for ARIMA, which is univariate and always uses "primary"
    regardless of what's requested (see the ARIMA branch below).
    """
    df = load_csv(file_bytes)
    validate_csv(df, asset, horizon, model_name)

    if model_name == "ARIMA":
        # ARIMA is univariate -- it never uses Google Trends or Fear &
        # Greed, and its saved metadata only exists under the "primary"
        # (Market Only) feature set, matching the default the dedicated
        # ARIMA page and /arima-info endpoint already use. Auto-detecting
        # a richer feature_set_key from the upload's optional columns
        # (the same helper RF/XGBoost/LSTM rely on) points at a folder
        # that was never created for ARIMA, and 404s -- see
        # run_arima_inference's docstring for why ARIMA doesn't vary by
        # feature configuration in the first place.
        detected_feature_set_key = _determine_feature_set_key(df, asset)
        result = run_arima_inference(df, asset, horizon, feature_set_key="primary")
        if detected_feature_set_key != "primary":
            result["inference_note"] += (
                " Your upload included optional columns (Google Trends and/or Fear & "
                "Greed), but ARIMA is univariate and never uses them -- this fit used "
                "only the price series, same as ARIMA always does regardless of "
                "feature configuration."
            )
        result["n_rows_uploaded"] = len(df)
        result["date_range"] = [str(df["Date"].min().date()), str(df["Date"].max().date())]
        return result

    if requested_feature_set:
        _validate_requested_feature_set(df, asset, requested_feature_set)
        feature_set_key = requested_feature_set
    else:
        feature_set_key = _determine_feature_set_key(df, asset)
    feature_set = preprocess_and_engineer(df, asset, feature_set_key, horizon)

    if model_name == "LSTM" and representation == "Original":
        _check_lstm_input_for_nan(feature_set, horizon, asset, feature_set_key)

    result = run_predict(
        horizon=horizon, asset=asset, feature_set_key=feature_set_key,
        model_name=model_name, representation=representation, input_df=feature_set,
    )
    result["feature_set_key"] = feature_set_key
    result["n_rows_uploaded"] = len(df)
    result["date_range"] = [str(df["Date"].min().date()), str(df["Date"].max().date())]
    return result
