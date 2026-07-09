"""Clean each dataset before the four sources are merged.

The steps run in a fixed order: parse dates, drop timezones, rename columns,
keep the study period, sort by date. Each function returns a new DataFrame
rather than modifying its input.
"""

import logging
from typing import Dict

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def convert_dates(df: pd.DataFrame, date_column: str, drop_timezone: bool) -> pd.DataFrame:
    """Parse the source date column, drop its timezone, and rename it 'Date'.

    Args:
        df: Raw dataset.
        date_column: Name of the date column in this source.
        drop_timezone: True for the timezone-aware CoinGecko files.

    Returns:
        A copy with a timezone-naive datetime column named 'Date' in position 0.

    Raises:
        ValueError: If the column cannot be parsed as dates.
    """
    df = df.copy()

    try:
        dates = pd.to_datetime(df[date_column])
    except (ValueError, TypeError) as error:
        raise ValueError(f"Could not parse '{date_column}' as dates: {error}") from error

    if drop_timezone:
        dates = dates.dt.tz_localize(None)

    df = df.drop(columns=[date_column])
    df.insert(0, "Date", dates)
    return df


def filter_study_period(
    df: pd.DataFrame,
    start: str = config.STUDY_START,
    end: str = config.STUDY_END,
) -> pd.DataFrame:
    """Keep only the rows inside the study period, both ends inclusive.

    Raises:
        ValueError: If start is after end, or if no rows survive the filter.
    """
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"Study start {start} is after study end {end}")

    inside_period = df["Date"].between(pd.Timestamp(start), pd.Timestamp(end))
    filtered = df[inside_period].reset_index(drop=True)

    if filtered.empty:
        raise ValueError(f"No rows left after filtering to {start} - {end}")

    return filtered


def clean_dataset(
    df: pd.DataFrame,
    name: str,
    start: str = config.STUDY_START,
    end: str = config.STUDY_END,
) -> pd.DataFrame:
    """Run the full cleaning sequence on one dataset.

    Args:
        df: Raw dataset.
        name: Dataset key, used to look up its date column and renames.
        start: First date of the study period.
        end: Last date of the study period.

    Returns:
        The cleaned dataset, sorted chronologically.
    """
    df = convert_dates(
        df,
        date_column=config.DATE_COLUMNS[name],
        drop_timezone=name in config.TIMEZONE_AWARE,
    )

    # Rename now so the merged frame has no ambiguous column names.
    df = df.rename(columns=config.RENAME_COLUMNS[name])

    df = filter_study_period(df, start, end)
    df = df.sort_values("Date").reset_index(drop=True)

    logger.info(
        "Cleaned %s: %d rows, %s to %s",
        name,
        len(df),
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    return df


def clean_all(
    datasets: Dict[str, pd.DataFrame],
    start: str = config.STUDY_START,
    end: str = config.STUDY_END,
) -> Dict[str, pd.DataFrame]:
    """Clean every dataset and return them under the same keys."""
    return {
        name: clean_dataset(df, name, start, end) for name, df in datasets.items()
    }
