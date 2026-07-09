"""Encode the categorical market sentiment variable."""

import logging
from typing import Dict, Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def encode_sentiment(
    df: pd.DataFrame, mapping: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """Add an ordinal encoding of the Fear & Greed sentiment categories.

    The original label is kept alongside the code: models need numbers, but
    plots and tables read better with the wording.

    Args:
        df: Merged dataset containing 'Market_Sentiment'.
        mapping: Category to integer map. Defaults to the project encoding.

    Returns:
        A copy with a 'Market_Sentiment_Encoded' column added.

    Raises:
        KeyError: If 'Market_Sentiment' is missing.
        ValueError: If a category falls outside the mapping. Without this
            check, pandas would quietly produce NaNs.
    """
    if "Market_Sentiment" not in df.columns:
        raise KeyError("Merged dataset has no 'Market_Sentiment' column")

    mapping = mapping or config.SENTIMENT_MAPPING

    unknown = set(df["Market_Sentiment"].dropna()) - set(mapping)
    if unknown:
        raise ValueError(f"Unmapped sentiment categories: {sorted(unknown)}")

    df = df.copy()
    df["Market_Sentiment_Encoded"] = df["Market_Sentiment"].map(mapping)

    logger.info("Encoded market sentiment into %d ordinal levels", len(mapping))
    return df


def decode_sentiment(
    values: pd.Series, mapping: Optional[Dict[str, int]] = None
) -> pd.Series:
    """Turn encoded sentiment values back into their category labels.

    Useful in Streamlit, where model inputs are numeric but the interface
    should show the original wording.
    """
    mapping = mapping or config.SENTIMENT_MAPPING
    return values.map({code: label for label, code in mapping.items()})
