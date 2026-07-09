"""Data quality checks on the merged dataset.

These functions report problems rather than fixing them. A quality issue in a
dissertation dataset is something to investigate and write about, not something
the code should quietly patch. The one exception is check_missing_values, which
raises: an inner join should never leave gaps, so if it has, something upstream
is wrong.
"""

import logging
from typing import Dict, List

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame) -> None:
    """Confirm the merged dataset has no missing values.

    Raises:
        ValueError: If any column contains nulls.
    """
    nulls = df.isnull().sum()
    columns_with_nulls = nulls[nulls > 0]

    if not columns_with_nulls.empty:
        raise ValueError(f"Merged dataset has missing values:\n{columns_with_nulls}")


def find_missing_dates(df: pd.DataFrame) -> List[pd.Timestamp]:
    """List calendar days absent from an otherwise continuous daily series."""
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    return list(full_range.difference(df["Date"]))


def find_missing_date_sources(
    datasets: Dict[str, pd.DataFrame], missing_dates: List[pd.Timestamp]
) -> pd.DataFrame:
    """Show which source dataset each missing date came from.

    This separates gaps that the providers had from gaps the merge introduced.

    Args:
        datasets: The four cleaned datasets used for the merge.
        missing_dates: Dates absent from the merged dataset.

    Returns:
        One row per missing date, with a True/False column per source showing
        whether that source contained the date.
    """
    rows = []
    for date in missing_dates:
        row = {"Missing Date": date.date()}
        for name, df in datasets.items():
            row[name] = bool((df["Date"] == date).any())
        rows.append(row)

    return pd.DataFrame(rows)


def count_duplicate_dates(df: pd.DataFrame) -> int:
    """Count repeated dates. Should be zero for a daily series."""
    return int(df["Date"].duplicated().sum())


def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Confirm Date is a datetime and the measured variables are numeric.

    Returns:
        A column-to-problem map. Empty when all types are correct.
    """
    problems = {}

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        problems["Date"] = f"expected datetime, found {df['Date'].dtype}"

    for column in config.NON_NEGATIVE_COLUMNS + config.BOUNDED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            problems[column] = f"expected numeric, found {df[column].dtype}"

    return problems


def check_variable_ranges(df: pd.DataFrame) -> Dict[str, str]:
    """Confirm every variable sits inside its plausible range.

    Prices, market caps and volumes must not be negative. Search interest and
    the Fear & Greed Index are defined on a 0-100 scale.

    Returns:
        A column-to-problem map. Empty when all values are valid.
    """
    problems = {}

    for column in config.NON_NEGATIVE_COLUMNS:
        negatives = (df[column] < 0).sum()
        if negatives:
            problems[column] = f"{negatives} negative value(s)"

    for column in config.BOUNDED_COLUMNS:
        out_of_range = (~df[column].between(0, 100)).sum()
        if out_of_range:
            problems[column] = f"{out_of_range} value(s) outside 0-100"

    return problems


def validate_merged_dataset(
    df: pd.DataFrame, datasets: Dict[str, pd.DataFrame]
) -> dict:
    """Run every quality check and return a summary.

    Args:
        df: The merged dataset.
        datasets: The cleaned source datasets, used to trace missing dates.

    Returns:
        A summary dict with row and column counts, missing dates, duplicate
        dates, and any type or range problems found.

    Raises:
        ValueError: If the merged dataset contains missing values.
    """
    check_missing_values(df)

    missing_dates = find_missing_dates(df)
    duplicate_dates = count_duplicate_dates(df)
    type_problems = check_data_types(df)
    range_problems = check_variable_ranges(df)

    if missing_dates:
        logger.warning("%d calendar date(s) missing from the series", len(missing_dates))
        sources = find_missing_date_sources(datasets, missing_dates)
        logger.info("Missing dates by source:\n%s", sources.to_string(index=False))

    if type_problems:
        logger.warning("Data type problems: %s", type_problems)

    if range_problems:
        logger.warning("Variable range problems: %s", range_problems)

    if not (type_problems or range_problems):
        logger.info("All quality checks passed")

    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": 0,  # guaranteed by check_missing_values above
        "missing_dates": [date.date() for date in missing_dates],
        "duplicate_dates": duplicate_dates,
        "type_problems": type_problems,
        "range_problems": range_problems,
    }


def summary_table(report: dict) -> pd.DataFrame:
    """Turn a validation report into the one-row table used in the write-up."""
    return pd.DataFrame(
        [
            {
                "Dataset": "Merged Dataset",
                "Rows": report["rows"],
                "Columns": report["columns"],
                "Missing Values": report["missing_values"],
                "Missing Dates": len(report["missing_dates"]),
                "Duplicate Dates": report["duplicate_dates"],
            }
        ]
    )
