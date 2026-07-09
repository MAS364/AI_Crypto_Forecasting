"""Merge the four cleaned datasets into a single daily time series."""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Merge order. Bitcoin leads because it is the widest dataset.
MERGE_ORDER = ["bitcoin", "ethereum", "google_trends", "fear_greed"]


def merge_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine the cleaned datasets on their shared 'Date' column.

    Use inner joins so every row has all required signals. Dates missing from
    any one source are therefore dropped, which is why the merged dataset can
    end earlier than the study period.

    Args:
        datasets: The four cleaned DataFrames, keyed by dataset name.

    Returns:
        The merged dataset, sorted chronologically.

    Raises:
        KeyError: If an expected dataset is missing.
        ValueError: If the datasets do not overlap at all.
    """
    missing = [name for name in MERGE_ORDER if name not in datasets]
    if missing:
        raise KeyError(f"Cannot merge, missing dataset(s): {missing}")

    merged = datasets[MERGE_ORDER[0]]
    for name in MERGE_ORDER[1:]:
        merged = merged.merge(datasets[name], on="Date", how="inner")

    if merged.empty:
        raise ValueError("Merge produced no rows - the datasets do not overlap")

    merged = merged.sort_values("Date").reset_index(drop=True)

    logger.info(
        "Merged dataset: %d rows, %d columns, %s to %s",
        len(merged),
        merged.shape[1],
        merged["Date"].min().date(),
        merged["Date"].max().date(),
    )
    return merged
