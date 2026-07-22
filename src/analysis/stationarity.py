"""Stationarity testing for the statistical analysis stage.

Runs the Augmented Dickey-Fuller (ADF) and KPSS tests on each variable, in
levels and after first differencing. The two tests have opposite null
hypotheses, so a variable is only treated as stationary when both tests
agree.
"""

import logging

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


def run_adf(series: pd.Series) -> dict:
    """Run the Augmented Dickey-Fuller test on a single series.

    The ADF null hypothesis is that the series has a unit root
    (is non-stationary), so a small p-value supports stationarity.
    """

    series = series.dropna()
    result = adfuller(series, autolag="AIC")

    return {
        "ADF Statistic": result[0],
        "ADF p-value": result[1],
        "Lags Used": result[2],
        "Observations": result[3],
        "Stationary (ADF)": "Yes" if result[1] < 0.05 else "No",
    }


def run_kpss(series: pd.Series) -> dict:
    """Run the KPSS test on a single series.

    The KPSS null hypothesis is that the series is stationary, so a
    large p-value supports stationarity (the opposite direction to ADF).
    """

    series = series.dropna()
    statistic, p_value, lags, _ = kpss(series, regression="c", nlags="auto")

    return {
        "KPSS Statistic": statistic,
        "KPSS p-value": p_value,
        "Lags Used": lags,
        "Stationary (KPSS)": "Yes" if p_value > 0.05 else "No",
    }


def overall_stationarity(adf_result: str, kpss_result: str) -> str:
    """Combine the ADF and KPSS verdicts into a single conclusion.

    The two tests can disagree, in which case the variable is reported as
    "Inconclusive" rather than forcing a verdict either test alone does
    not support.
    """

    if adf_result == "Yes" and kpss_result == "Yes":
        return "Stationary"
    elif adf_result == "No" and kpss_result == "No":
        return "Non-Stationary"
    else:
        return "Inconclusive"


def test_stationarity(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Run ADF and KPSS on each variable and return a combined results table."""

    rows = []

    for variable in variables:
        series = df[variable].dropna()

        if len(series) == 0:
            logger.warning("Skipping %s: no data available.", variable)
            continue

        adf = run_adf(series)
        kpss_result = run_kpss(series)

        rows.append({
            "Variable": variable,
            "ADF Statistic": round(adf["ADF Statistic"], 4),
            "ADF p-value": round(adf["ADF p-value"], 4),
            "ADF Stationary": adf["Stationary (ADF)"],
            "KPSS Statistic": round(kpss_result["KPSS Statistic"], 4),
            "KPSS p-value": round(kpss_result["KPSS p-value"], 4),
            "KPSS Stationary": kpss_result["Stationary (KPSS)"],
        })

    results = pd.DataFrame(rows)
    results["Overall Conclusion"] = results.apply(
        lambda row: overall_stationarity(row["ADF Stationary"], row["KPSS Stationary"]),
        axis=1,
    )

    return results


def test_stationarity_differenced(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Run ADF and KPSS on the first difference of each variable.

    Uses the same full table structure as `test_stationarity` (statistics,
    Yes/No verdicts, and an overall conclusion), applied to the
    differenced series rather than the original levels.
    """

    differenced_df = pd.DataFrame({
        variable: df[variable].diff() for variable in variables
    })

    return test_stationarity(differenced_df, variables)