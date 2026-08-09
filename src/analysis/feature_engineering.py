"""Feature selection and feature engineering for the modelling stage.

Turns the correlation and Granger causality findings from the statistical
analysis stage (05_Correlation_Causality_analysis.ipynb) into an explicit,
auditable feature-selection decision, then builds leakage-safe engineered
feature sets for Bitcoin and Ethereum forecasting.

This module deliberately does not re-run any statistical test. Every
correlation coefficient, p-value, and lag used below is read from the
tables already saved by the causality analysis stage
(results/causality_analysis/tables/), so the feature-selection decisions
made here stay traceable to that analysis rather than being recomputed or
re-justified independently.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# =======================================================================
# Validation
# =======================================================================

def validate_input_data(df: pd.DataFrame, required_columns: list[str]) -> dict:
    """Validate the merged dataset before any feature engineering runs.

    Checks that the date column is datetime, that observations are in
    chronological order, and reports duplicate dates and any missing
    calendar days. Raises only if a required column is absent, since
    that cannot be worked around; every other finding is returned for
    the caller to print and act on.
    """

    missing_columns = [c for c in required_columns if c not in df.columns]
    if missing_columns:
        raise KeyError(f"Required columns missing from input data: {missing_columns}")

    is_datetime = pd.api.types.is_datetime64_any_dtype(df["Date"])
    is_chronological = df["Date"].is_monotonic_increasing
    duplicate_dates = int(df["Date"].duplicated().sum())

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing_dates = full_range.difference(df["Date"])

    return {
        "is_datetime": is_datetime,
        "is_chronological": is_chronological,
        "duplicate_dates": duplicate_dates,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "date_min": df["Date"].min(),
        "date_max": df["Date"].max(),
        "missing_calendar_dates": len(missing_dates),
        "missing_calendar_dates_list": list(missing_dates),
        "missing_values_by_column": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
    }


# =======================================================================
# Feature-selection decision table
# =======================================================================

def build_feature_selection_decisions(causality_tables_dir: Path) -> pd.DataFrame:
    """Build the feature-selection decision table from saved analysis outputs.

    Reads the correlation and Granger causality results already produced
    by 05_Correlation_Causality_analysis.ipynb (via src/analysis/causality.py
    and src/analysis/correlation.py) rather than recomputing them, so the
    coefficients and p-values shown here always match the statistical
    analysis chapter exactly.

    The `selection_category` and `keep_in_*` columns encode the two-stage
    feature strategy already established for this dissertation: variables
    with no correlation or Granger evidence are excluded from the primary
    set by default; variables with weak or borderline evidence are kept
    only as candidates for empirical comparison during modelling, never
    assumed to help.
    """

    correlation_results = pd.read_csv(causality_tables_dir / "correlation_pairwise_results.csv")
    granger_results = pd.read_csv(causality_tables_dir / "granger_causality_results.csv")
    stationarity_original = pd.read_csv(causality_tables_dir / "stationarity_original.csv")

    stationarity_lookup = stationarity_original.set_index("Variable")["Overall Conclusion"]

    def pearson_row(var_a: str, var_b: str) -> Optional[pd.Series]:
        match = correlation_results[
            (correlation_results["Method"] == "Pearson")
            & (
                ((correlation_results["Variable A"] == var_a) & (correlation_results["Variable B"] == var_b))
                | ((correlation_results["Variable A"] == var_b) & (correlation_results["Variable B"] == var_a))
            )
        ]
        return match.iloc[0] if not match.empty else None

    def spearman_row(var_a: str, var_b: str) -> Optional[pd.Series]:
        match = correlation_results[
            (correlation_results["Method"] == "Spearman")
            & (
                ((correlation_results["Variable A"] == var_a) & (correlation_results["Variable B"] == var_b))
                | ((correlation_results["Variable A"] == var_b) & (correlation_results["Variable B"] == var_a))
            )
        ]
        return match.iloc[0] if not match.empty else None

    def granger_row(predictor: str, target: str) -> Optional[pd.Series]:
        match = granger_results[
            (granger_results["Predictor"] == predictor) & (granger_results["Target"] == target)
        ]
        return match.iloc[0] if not match.empty else None

    rows = []

    # -------------------------------------------------------------
    # Google Trends search interest (same-asset only, as tested)
    # -------------------------------------------------------------
    for asset, target_var in [("Bitcoin", "Bitcoin_Price"), ("Ethereum", "Ethereum_Price")]:
        predictor_var = f"{asset}_Search_Interest"

        pearson = pearson_row(predictor_var, target_var)
        spearman = spearman_row(predictor_var, target_var)
        granger = granger_row(predictor_var, target_var)

        rows.append({
            "variable": predictor_var,
            "variable_type": "external_factor",
            "tested_against": target_var,
            "stationarity_status_original": stationarity_lookup.get(predictor_var, "Unknown"),
            "pearson_r": pearson["Coefficient"] if pearson is not None else np.nan,
            "pearson_p": pearson["p-value"] if pearson is not None else np.nan,
            "pearson_significant_fdr": bool(pearson["Significant (FDR-corrected)"]) if pearson is not None else None,
            "spearman_r": spearman["Coefficient"] if spearman is not None else np.nan,
            "spearman_p": spearman["p-value"] if spearman is not None else np.nan,
            "spearman_significant_fdr": bool(spearman["Significant (FDR-corrected)"]) if spearman is not None else None,
            "granger_p": granger["p-value"] if granger is not None else np.nan,
            "granger_significant_fdr": bool(granger["Significant (FDR-corrected)"]) if granger is not None else None,
            "granger_selected_lag": int(granger["Selected Lag"]) if granger is not None else None,
            "selection_category": "Candidate (weak/borderline evidence)",
            "keep_in_primary_set": False,
            "keep_in_candidate_set": True,
            "keep_in_ablation_set": False,
            "justification": (
                "Correlation with own-asset price is weak (|r| < 0.2 in both tests) and the "
                f"Granger test against {target_var} is not significant at the evidence-selected "
                "lag, before or after FDR correction. Retained only as a candidate for empirical "
                "comparison during modelling, per the dissertation's two-stage feature strategy — "
                "not treated as a statistically confirmed predictor."
            ),
        })

    # -------------------------------------------------------------
    # Fear & Greed Index (tested against both targets)
    # -------------------------------------------------------------
    fg_var = config.SENTIMENT_SCORE_VARIABLE
    for target_var in ["Bitcoin_Price", "Ethereum_Price"]:
        pearson = pearson_row(fg_var, target_var)
        spearman = spearman_row(fg_var, target_var)
        granger = granger_row(fg_var, target_var)

        rows.append({
            "variable": fg_var,
            "variable_type": "external_factor",
            "tested_against": target_var,
            "stationarity_status_original": stationarity_lookup.get(fg_var, "Unknown"),
            "pearson_r": pearson["Coefficient"] if pearson is not None else np.nan,
            "pearson_p": pearson["p-value"] if pearson is not None else np.nan,
            "pearson_significant_fdr": bool(pearson["Significant (FDR-corrected)"]) if pearson is not None else None,
            "spearman_r": spearman["Coefficient"] if spearman is not None else np.nan,
            "spearman_p": spearman["p-value"] if spearman is not None else np.nan,
            "spearman_significant_fdr": bool(spearman["Significant (FDR-corrected)"]) if spearman is not None else None,
            "granger_p": granger["p-value"] if granger is not None else np.nan,
            "granger_significant_fdr": bool(granger["Significant (FDR-corrected)"]) if granger is not None else None,
            "granger_selected_lag": int(granger["Selected Lag"]) if granger is not None else None,
            "selection_category": "Excluded (no evidence)",
            "keep_in_primary_set": False,
            "keep_in_candidate_set": False,
            "keep_in_ablation_set": True,
            "justification": (
                f"No significant correlation with {target_var} (raw or FDR-corrected) and no "
                "significant Granger-causal relationship. Excluded from the primary and candidate "
                "feature sets. Retained only in an explicit, clearly labelled ablation dataset, "
                "not because the evidence supports it."
            ),
        })

    # -------------------------------------------------------------
    # Trading volume (never tested via Granger; internal market variable)
    # -------------------------------------------------------------
    for asset, target_var, volume_var in [
        ("Bitcoin", "Bitcoin_Price", "Bitcoin_Volume"),
        ("Ethereum", "Ethereum_Price", "Ethereum_Volume"),
    ]:
        pearson = pearson_row(volume_var, target_var)
        spearman = spearman_row(volume_var, target_var)

        rows.append({
            "variable": volume_var,
            "variable_type": "internal_market_variable",
            "tested_against": target_var,
            "stationarity_status_original": stationarity_lookup.get(volume_var, "Unknown"),
            "pearson_r": pearson["Coefficient"] if pearson is not None else np.nan,
            "pearson_p": pearson["p-value"] if pearson is not None else np.nan,
            "pearson_significant_fdr": bool(pearson["Significant (FDR-corrected)"]) if pearson is not None else None,
            "spearman_r": spearman["Coefficient"] if spearman is not None else np.nan,
            "spearman_p": spearman["p-value"] if spearman is not None else np.nan,
            "spearman_significant_fdr": bool(spearman["Significant (FDR-corrected)"]) if spearman is not None else None,
            "granger_p": np.nan,
            "granger_significant_fdr": None,
            "granger_selected_lag": None,
            "selection_category": "Not tested for causality (internal market variable)",
            "keep_in_primary_set": True,
            "keep_in_candidate_set": False,
            "keep_in_ablation_set": False,
            "justification": (
                "Volume was deliberately excluded from the Granger causality analysis in this "
                "dissertation (it is treated as a market variable, not an external factor). Its "
                "inclusion here as an own-asset autoregressive feature is justified by the "
                "stationarity analysis (first-differenced volume is confirmed stationary), not by "
                "correlation or causality evidence with price."
            ),
        })

    # -------------------------------------------------------------
    # Cross-asset price relationship (flagged, not actioned)
    # -------------------------------------------------------------
    pearson = pearson_row("Bitcoin_Price", "Ethereum_Price")
    rows.append({
        "variable": "Bitcoin_Price / Ethereum_Price (cross-asset)",
        "variable_type": "cross_asset_flag",
        "tested_against": "N/A",
        "stationarity_status_original": "N/A",
        "pearson_r": pearson["Coefficient"] if pearson is not None else np.nan,
        "pearson_p": pearson["p-value"] if pearson is not None else np.nan,
        "pearson_significant_fdr": bool(pearson["Significant (FDR-corrected)"]) if pearson is not None else None,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "spearman_significant_fdr": None,
        "granger_p": np.nan,
        "granger_significant_fdr": None,
        "granger_selected_lag": None,
        "selection_category": "Not tested (cross-asset Granger causality not established)",
        "keep_in_primary_set": False,
        "keep_in_candidate_set": False,
        "keep_in_ablation_set": False,
        "justification": (
            "Bitcoin and Ethereum returns are strongly correlated (r = 0.838, the strongest "
            "relationship in the entire correlation analysis), but this dissertation's Granger "
            "causality testing was restricted to same-asset external-factor pairs and never "
            "tested this cross-asset relationship. No cross-asset lagged feature is created "
            "here as a result."
        ),
    })

    return pd.DataFrame(rows)


# =======================================================================
# Target construction
# =======================================================================

def align_forecast_target(
    df: pd.DataFrame, price_column: str, horizon: int = 1
) -> pd.DataFrame:
    """Build the horizon-day-ahead cumulative log-return forecasting target.

    For horizon=1, this is identical to the original one-step-ahead
    target (np.log(price).diff().shift(-1)), matching the variable
    already tested for stationarity, correlation, and Granger causality.

    For horizon > 1, the target is the CUMULATIVE return over the next
    `horizon` days -- log(price_t+horizon / price_t) -- not the single
    daily return that happens to fall exactly `horizon` days ahead.
    These are different quantities: log_return.shift(-horizon) alone
    would give "the one-day return that occurs on day t+horizon", which
    is not what "predicting `horizon` days ahead" means. Log returns are
    additive, so the cumulative return equals the sum of the intervening
    daily returns; it is computed directly here as a price-ratio log
    rather than summed lag-by-lag, since that is simpler and numerically
    identical.

    Predictor construction and target construction are kept in separate
    functions so that a leakage check can verify the target was not used
    to build any predictor.
    """

    result = df[["Date"]].copy()
    log_return = np.log(df[price_column]).diff()

    result[f"{price_column}_log_return"] = log_return
    result[f"{price_column}_log_return_target_t_plus_{horizon}"] = (
        np.log(df[price_column].shift(-horizon)) - np.log(df[price_column])
    )

    return result


# =======================================================================
# Predictor construction
# =======================================================================

def create_lag_features(
    series: pd.Series, lags: list[int], name: str
) -> pd.DataFrame:
    """Create lagged versions of a series using only past information.

    `series.shift(k)` for k > 0 moves values forward in time (row t
    receives the value from row t-k), which is the leakage-safe direction.
    """

    return pd.DataFrame({
        f"{name}_lag_{lag}": series.shift(lag) for lag in lags
    })


def create_rolling_features(
    series: pd.Series,
    windows: list[int],
    name: str,
    statistics: list[str] = ("mean", "std"),
) -> pd.DataFrame:
    """Create rolling statistics using only past, non-centred observations.

    The series is shifted by one period before the rolling window is
    applied, so the rolling statistic available for row t is computed
    over rows [t-window, t-1], never including row t itself. This is the
    "shift before rolling" pattern required to avoid leaking the current
    observation into its own feature.
    """

    shifted = series.shift(1)
    features = {}

    for window in windows:
        rolling = shifted.rolling(window=window, center=False)
        for statistic in statistics:
            if statistic == "mean":
                features[f"{name}_rolling_mean_{window}"] = rolling.mean()
            elif statistic == "std":
                features[f"{name}_rolling_std_{window}"] = rolling.std()
            elif statistic == "min":
                features[f"{name}_rolling_min_{window}"] = rolling.min()
            elif statistic == "max":
                features[f"{name}_rolling_max_{window}"] = rolling.max()
            elif statistic == "median":
                features[f"{name}_rolling_median_{window}"] = rolling.median()
            else:
                raise ValueError(f"Unsupported rolling statistic: {statistic}")

    return pd.DataFrame(features)


def create_volume_features(
    df: pd.DataFrame, volume_column: str, conventional_lags: list[int]
) -> pd.DataFrame:
    """Build own-asset volume features.

    The known nine-day corrupt volume window (documented in the data
    preparation stage) is linearly interpolated here before any lag,
    pct_change, log, or rolling calculation -- so every feature derived
    from volume is fully populated, for every row, in every context this
    function is used in (including live single-window inference, which
    cannot simply drop an affected row the way training can).

    This is a deliberate change from an earlier version of this function,
    which left these NaNs in place on principle ("handled explicitly at
    the row-alignment stage" -- fine for training, where dropping a
    handful of rows out of 1,700+ costs nothing, but unworkable for
    inference on a fixed recent window). A short, isolated 9-day gap in
    an otherwise continuous daily series is a reasonable case for linear
    interpolation: volume moves gradually day to day, and 9 days is small
    relative to the 30-day rolling window it would otherwise contaminate
    for roughly the next 30 days after the gap.

    Note this means the "_raw_lag_N" feature names are no longer
    strictly raw/unmodified for rows whose lag reaches into the
    interpolated window -- the naming is kept unchanged for column-name
    compatibility with already-saved models and metadata, but the values
    within the interpolated window are estimates, not observed readings.
    The detection/masking decision itself (implausible volume -> NaN) is
    unchanged and lives upstream in src/preprocessing/validation.py --
    this function only changes how that masked gap is treated once it
    reaches feature engineering.
    """

    volume = df[volume_column].interpolate(method="linear", limit_direction="both")
    log_volume = np.log(volume.clip(lower=1e-9))

    features = {}

    lag_features = create_lag_features(volume, conventional_lags, name=f"{volume_column}_raw")
    features.update(lag_features.to_dict("series"))

    pct_change = volume.pct_change()
    for lag in conventional_lags:
        features[f"{volume_column}_pct_change_lag_{lag}"] = pct_change.shift(lag)

    features[f"{volume_column}_log"] = log_volume

    rolling_features = create_rolling_features(
        volume,
        windows=[config.ROLLING_WINDOW_EDA_PRECEDENT],
        name=volume_column,
        statistics=["mean", "std"],
    )

    result = pd.concat([pd.DataFrame(features, index=df.index), rolling_features], axis=1)
    return result


def create_external_features(
    df: pd.DataFrame,
    variable: str,
    evidence_lag: int,
    conventional_lag: int = 1,
) -> pd.DataFrame:
    """Build a minimal, evidence-anchored feature set for a candidate external variable.

    Important: the Granger causality test that selected `evidence_lag` (see
    05_Correlation_Causality_analysis.ipynb) was run on the first-differenced,
    stationary version of this variable, not on its raw level — the variable
    was flagged "Inconclusive" for stationarity in levels and transformed via
    `.diff()` before testing (src/analysis/correlation.py,
    prepare_stationary_dataset). The true evidence-anchored feature is
    therefore `{variable}_diff_lag_{evidence_lag}`, built from the
    differenced series at the tested lag, not a lag of the raw level.

    Four features are created:

    - `{variable}_lag_{conventional_lag}`: raw level, conventional lag.
      Not evidence-based — no test was ever run on the raw level.
    - `{variable}_lag_{evidence_lag}`: raw level, evidence-selected lag
      number. Included for completeness/comparison only; still not what
      was actually Granger-tested, since the test used the differenced
      series, not this raw level.
    - `{variable}_diff_lag_1`: differenced series, lag 1. Conventional,
      not the tested lag.
    - `{variable}_diff_lag_{evidence_lag}`: differenced series, at the
      evidence-selected lag. This is the one feature that genuinely
      reproduces the variable-and-lag combination validated by the
      Granger causality test.
    """

    series = df[variable]
    diff = series.diff()

    features = pd.DataFrame({
        f"{variable}_lag_{conventional_lag}": series.shift(conventional_lag),
        f"{variable}_lag_{evidence_lag}": series.shift(evidence_lag),
        f"{variable}_diff_lag_1": diff.shift(1),
        f"{variable}_diff_lag_{evidence_lag}": diff.shift(evidence_lag),
    })

    return features


def create_sentiment_ablation_features(
    df: pd.DataFrame,
    score_variable: str,
    ordinal_variable: str,
    evidence_lag: int,
    conventional_lag: int = 1,
    short_window: int = 7,
) -> pd.DataFrame:
    """Build a deliberately minimal, clearly labelled sentiment ablation set.

    As with `create_external_features`, the Granger test for this variable
    was run on its first-differenced series, so `{score_variable}_diff_lag_
    {evidence_lag}` — not the raw-level lag — is the feature that actually
    reproduces the tested variable-and-lag combination.

    Reuses the ordinal sentiment encoding already created during
    preprocessing (Market_Sentiment_Encoded) rather than creating a new
    one. That encoding treats the five sentiment categories as equally
    spaced (0 to 4); this assumption is not tested anywhere in this
    project and is inherited, not validated, here.
    """

    score = df[score_variable]
    ordinal = df[ordinal_variable]
    diff = score.diff()

    features = pd.DataFrame({
        f"{score_variable}_lag_{conventional_lag}": score.shift(conventional_lag),
        f"{score_variable}_lag_{evidence_lag}": score.shift(evidence_lag),
        f"{score_variable}_diff_lag_1": diff.shift(1),
        f"{score_variable}_diff_lag_{evidence_lag}": diff.shift(evidence_lag),
        f"{score_variable}_rolling_mean_{short_window}": score.shift(1).rolling(short_window).mean(),
        f"{ordinal_variable}_lag_{conventional_lag}": ordinal.shift(conventional_lag),
    })

    return features


# =======================================================================
# Assembly and validation
# =======================================================================

def build_feature_set(
    df: pd.DataFrame,
    asset: str,
    feature_set_name: str,
    include_external: bool = False,
    include_ablation: bool = False,
    horizon: int = 1,
) -> pd.DataFrame:
    """Assemble one named feature set (primary / candidate / ablation) for one asset.

    Always includes the baseline autoregressive block (own-asset return
    lags, volume features, rolling statistics). Optionally adds the
    external-candidate block (Google Trends) and/or the sentiment
    ablation block (Fear & Greed), each clearly separated by column
    prefix so the source of every feature remains traceable.

    `horizon` controls how many days ahead the target looks (default 1,
    matching the project's original one-day-ahead design). All predictor
    blocks below are backward-looking only (built from information
    available at time t) and are therefore unaffected by `horizon` --
    only the target itself, and the persistence-baseline reference
    feature, depend on it. See align_forecast_target() for the exact
    target definition at a given horizon.

    Also carries a raw price *reference* column (`{price}_level`)
    forward, undifferenced and unlagged. This is not a predictor and is
    not used by any model — its only purpose is to let the evaluation
    stage reconstruct price-level forecasts from predicted log returns
    (predicted_price_t+horizon = price_level_t * exp(predicted_log_return)),
    since the dissertation's aim is price forecasting even though the
    models themselves are built on log returns. Any model training code
    downstream must explicitly exclude this column from its predictor
    matrix.
    """

    asset_vars = config.ASSET_VARIABLES[asset]
    price_column = asset_vars["price"]
    volume_column = asset_vars["volume"]
    search_column = asset_vars["search_interest"]

    target_block = align_forecast_target(df, price_column=price_column, horizon=horizon)
    log_return = target_block[f"{price_column}_log_return"]

    # Trailing horizon-day cumulative return: log(price_t / price_t-horizon).
    # Only meaningful (and only added) when horizon > 1 -- it is the direct
    # analogue of the horizon=1 persistence baseline (which just uses
    # log_return_lag_1) at the new horizon: "the last `horizon` days predict
    # the next `horizon` days". Needed for a sensible Naive baseline in
    # Notebook 07, not used as a predictor by any other model here.
    trailing_return_block = None
    if horizon > 1:
        trailing_return_block = pd.DataFrame({
            f"{price_column}_trailing_{horizon}day_return": (
                np.log(df[price_column]) - np.log(df[price_column].shift(horizon))
            )
        })

    price_reference = pd.DataFrame({f"{price_column}_level": df[price_column].values})

    all_lags = sorted(set(config.CONVENTIONAL_LAGS + [config.EVIDENCE_SELECTED_LAG]))
    return_lag_features = create_lag_features(log_return, all_lags, name=f"{price_column}_log_return")

    return_rolling_features = create_rolling_features(
        log_return,
        windows=[config.ROLLING_WINDOW_EDA_PRECEDENT],
        name=f"{price_column}_log_return",
        statistics=["mean", "std"],
    )

    momentum_window = max(config.CONVENTIONAL_LAGS)
    momentum_features = pd.DataFrame({
        f"{price_column}_momentum_{momentum_window}": df[price_column].shift(1) - df[price_column].shift(1 + momentum_window),
    })

    volume_features = create_volume_features(df, volume_column, config.CONVENTIONAL_LAGS)

    blocks = [target_block, price_reference, return_lag_features, return_rolling_features, momentum_features, volume_features]
    if trailing_return_block is not None:
        blocks.append(trailing_return_block)

    if include_external:
        blocks.append(
            create_external_features(
                df, search_column,
                evidence_lag=config.EVIDENCE_SELECTED_LAG,
                conventional_lag=min(config.CONVENTIONAL_LAGS),
            )
        )

    if include_ablation:
        blocks.append(
            create_sentiment_ablation_features(
                df,
                score_variable=config.SENTIMENT_SCORE_VARIABLE,
                ordinal_variable=config.SENTIMENT_ORDINAL_VARIABLE,
                evidence_lag=config.EVIDENCE_SELECTED_LAG,
                conventional_lag=min(config.CONVENTIONAL_LAGS),
                short_window=config.ROLLING_WINDOW_SHORT,
            )
        )

    feature_set = pd.concat(blocks, axis=1)
    feature_set = feature_set.loc[:, ~feature_set.columns.duplicated()]
    date_column = feature_set.pop("Date")
    feature_set.insert(0, "Date", date_column)
    feature_set.attrs["feature_set_name"] = feature_set_name
    feature_set.attrs["asset"] = asset

    return feature_set


def validate_no_leakage(
    df: pd.DataFrame, feature_set: pd.DataFrame, source_column: str, lag: int, feature_name: str
) -> bool:
    """Spot-check that a lag feature exactly equals the source column shifted by `lag`.

    Compares on overlapping non-null rows only, since the feature set may
    already have had unusable rows removed by the time this is called.
    """

    expected = df[source_column].shift(lag)
    actual = feature_set[feature_name]

    aligned = pd.concat([expected, actual], axis=1, keys=["expected", "actual"]).dropna()
    if aligned.empty:
        logger.warning("No overlapping non-null rows to validate leakage for %s.", feature_name)
        return True

    matches = np.isclose(aligned["expected"], aligned["actual"], equal_nan=True)
    return bool(matches.all())


def run_leakage_checks(df: pd.DataFrame, feature_set: pd.DataFrame, asset: str, horizon: int = 1) -> pd.DataFrame:
    """Run the standard set of leakage checks against one assembled feature set."""

    asset_vars = config.ASSET_VARIABLES[asset]
    price_column = asset_vars["price"]

    checks = []

    checks.append({
        "check": "Chronological order preserved",
        "passed": feature_set["Date"].is_monotonic_increasing,
    })
    checks.append({
        "check": "No duplicate dates",
        "passed": not feature_set["Date"].duplicated().any(),
    })

    price_lag_col = f"{price_column}_log_return_lag_{config.EVIDENCE_SELECTED_LAG}"
    if price_lag_col in feature_set.columns:
        log_return = np.log(df[price_column]).diff()
        expected = log_return.shift(config.EVIDENCE_SELECTED_LAG)
        aligned = pd.concat(
            [expected, feature_set[price_lag_col]], axis=1, keys=["expected", "actual"]
        ).dropna()
        matches = np.isclose(aligned["expected"], aligned["actual"], equal_nan=True).all() if not aligned.empty else True
        checks.append({"check": f"{price_lag_col} equals source shifted by {config.EVIDENCE_SELECTED_LAG}", "passed": bool(matches)})

    target_col = f"{price_column}_log_return_target_t_plus_{horizon}"
    if target_col in feature_set.columns:
        expected_target = np.log(df[price_column].shift(-horizon)) - np.log(df[price_column])
        aligned = pd.concat(
            [expected_target, feature_set[target_col]], axis=1, keys=["expected", "actual"]
        ).dropna()
        matches = np.isclose(aligned["expected"], aligned["actual"], equal_nan=True).all() if not aligned.empty else True
        checks.append({"check": f"Target aligned to t+{horizon} (cumulative {horizon}-day return)", "passed": bool(matches)})

    if horizon > 1:
        trailing_col = f"{price_column}_trailing_{horizon}day_return"
        if trailing_col in feature_set.columns:
            expected_trailing = np.log(df[price_column]) - np.log(df[price_column].shift(horizon))
            aligned = pd.concat(
                [expected_trailing, feature_set[trailing_col]], axis=1, keys=["expected", "actual"]
            ).dropna()
            matches = np.isclose(aligned["expected"], aligned["actual"], equal_nan=True).all() if not aligned.empty else True
            checks.append({"check": f"{trailing_col} uses only information available at t (no future leakage)", "passed": bool(matches)})

    checks.append({
        "check": "No infinite values in feature set",
        "passed": not np.isinf(feature_set.select_dtypes(include=[np.number])).any().any(),
    })

    return pd.DataFrame(checks)


def save_feature_outputs(
    feature_set: pd.DataFrame,
    decisions_table: pd.DataFrame,
    metadata_table: pd.DataFrame,
    output_path: Path,
    decisions_path: Path,
    metadata_path: Path,
) -> None:
    """Save an assembled feature set plus the shared decision and metadata tables."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_set.to_csv(output_path, index=False)

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_table.to_csv(decisions_path, index=False)
    metadata_table.to_csv(metadata_path, index=False)
