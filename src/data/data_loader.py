"""
Load the raw datasets used throughout the project.

This module centralises all file loading so that notebooks,
the preprocessing pipeline, and the Streamlit application
use the same data source.
"""

from pathlib import Path
import logging

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Loading %s", file_path.name)

    return pd.read_csv(file_path)


def load_raw_data() -> dict[str, pd.DataFrame]:
    """
    Load all raw datasets required for the project.

    Returns
    -------
    dict
        Dictionary containing the raw datasets.
    """

    logger.info("Loading raw datasets...")

    datasets = {
        "bitcoin": load_csv(config.BITCOIN_FILE),
        "ethereum": load_csv(config.ETHEREUM_FILE),
        "google_trends": load_csv(config.GOOGLE_TRENDS_FILE),
        "fear_greed": load_csv(config.FEAR_GREED_FILE),
    }

    logger.info("Successfully loaded %d datasets.", len(datasets))

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data = load_raw_data()

    for name, df in data.items():
        print(f"\n{name.upper()}")
        print(df.head())
        print(df.shape)