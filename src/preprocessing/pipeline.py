"""Preprocessing pipeline for the cryptocurrency forecasting project.

This module is the main entry point for converting the four raw datasets into
one clean, analysis-ready dataset.

Run from the project root:

    python -m src.preprocessing.pipeline
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config
from src.data.data_loader import load_raw_data
from src.preprocessing.cleaning import clean_all
from src.preprocessing.encoding import encode_sentiment
from src.preprocessing.integration import merge_datasets
from src.preprocessing.validation import summary_table, validate_merged_dataset

logger = logging.getLogger(__name__)


def run_preprocessing(
    raw_datasets: Optional[dict[str, pd.DataFrame]] = None,
    start: str = config.STUDY_START,
    end: str = config.STUDY_END,
    output_path: Optional[Path] = config.MERGED_FILE,
) -> tuple[pd.DataFrame, dict]:
    """Run the full preprocessing workflow.

    Args:
        raw_datasets: Optional pre-loaded datasets. If not provided, the raw CSV
            files are loaded from the paths defined in `config.py`.
        start: Start date of the study period.
        end: End date of the study period.
        output_path: Location where the processed dataset should be saved.
            If None, the dataset is not saved.

    Returns:
        A tuple containing:
            - the final processed DataFrame
            - a validation report dictionary
    """

    logger.info("Starting preprocessing pipeline: %s to %s", start, end)

    # Load the four raw datasets if they were not passed in directly.
    if raw_datasets is None:
        raw_datasets = load_raw_data()

    # Clean dates, remove timezone information, standardise columns,
    # filter the study period, and sort each dataset.
    cleaned_datasets = clean_all(raw_datasets, start=start, end=end)

    # Merge Bitcoin, Ethereum, Google Trends, and Fear & Greed data on Date.
    merged_data = merge_datasets(cleaned_datasets)

    # Convert sentiment labels into ordered numeric values.
    merged_data = encode_sentiment(merged_data)

    # Keep the final dataset in a consistent column order.
    missing_columns = [
        col for col in config.FINAL_COLUMNS if col not in merged_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The merged dataset is missing expected columns: "
            f"{missing_columns}"
        )

    merged_data = merged_data[config.FINAL_COLUMNS]

    # Validate the final dataset before saving.
    report = validate_merged_dataset(merged_data, cleaned_datasets)

    # Save the final dataset for EDA, modelling, and deployment.
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_data.to_csv(output_path, index=False)
        logger.info("Processed dataset saved to %s", output_path)

    logger.info("Preprocessing completed successfully.")

    return merged_data, report


def load_processed_data(path: Path = config.MERGED_FILE) -> pd.DataFrame:
    """Load the saved processed dataset.

    Args:
        path: Path to the processed CSV file.

    Returns:
        Processed dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the processed dataset has not been created yet.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {path}. "
            "Run `python -m src.preprocessing.pipeline` first."
        )

    return pd.read_csv(path, parse_dates=["Date"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )

    final_dataset, validation_report = run_preprocessing()

    print(summary_table(validation_report).to_string(index=False))