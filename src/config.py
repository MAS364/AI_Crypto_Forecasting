"""Project paths, study period, and column mappings.

Everything that the preprocessing modules need to agree on lives here, so
there is one place to change a path, date range, or column name.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------
# Raw data files
# ---------------------------------------------------------------------

BITCOIN_FILE = RAW_DATA_DIR / "btc-usd-max.csv"
ETHEREUM_FILE = RAW_DATA_DIR / "eth-usd-max.csv"
GOOGLE_TRENDS_FILE = RAW_DATA_DIR / "google_trends.csv"
FEAR_GREED_FILE = RAW_DATA_DIR / "fear_greed.csv"

MERGED_FILE = PROCESSED_DATA_DIR / "merged_dataset.csv"

# ---------------------------------------------------------------------
# Study period
# ---------------------------------------------------------------------

# The effective final date is controlled by the available Google Trends data.
STUDY_START = "2021-06-01"
STUDY_END = "2026-05-06"

# ---------------------------------------------------------------------
# Source date columns
# ---------------------------------------------------------------------

DATE_COLUMNS = {
    "bitcoin": "event_date",
    "ethereum": "event_date",
    "google_trends": "Date",
    "fear_greed": "Date",
}

# CoinGecko exports dates in UTC; the other sources are timezone-naive.
TIMEZONE_AWARE = {"bitcoin", "ethereum"}

# ---------------------------------------------------------------------
# Column renaming
# ---------------------------------------------------------------------

RENAME_COLUMNS = {
    "bitcoin": {
        "close_price_usd": "Bitcoin_Price",
        "market_cap_usd": "Bitcoin_Market_Cap",
        "volume_usd": "Bitcoin_Volume",
    },
    "ethereum": {
        "close_price_usd": "Ethereum_Price",
        "market_cap_usd": "Ethereum_Market_Cap",
        "volume_usd": "Ethereum_Volume",
    },
    "google_trends": {
        "Bitcoin": "Bitcoin_Search_Interest",
        "Ethereum": "Ethereum_Search_Interest",
    },
    "fear_greed": {
        "Classification": "Market_Sentiment",
    },
}

# ---------------------------------------------------------------------
# Sentiment encoding
# ---------------------------------------------------------------------

SENTIMENT_MAPPING = {
    "Extreme Fear": 0,
    "Fear": 1,
    "Neutral": 2,
    "Greed": 3,
    "Extreme Greed": 4,
}

# ---------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------

NON_NEGATIVE_COLUMNS = [
    "Bitcoin_Price",
    "Bitcoin_Market_Cap",
    "Bitcoin_Volume",
    "Ethereum_Price",
    "Ethereum_Market_Cap",
    "Ethereum_Volume",
    "Bitcoin_Search_Interest",
    "Ethereum_Search_Interest",
]



VALID_RANGES = {
    "Fear_Greed_Index": (0, 100),
}

# ---------------------------------------------------------------------
# Final dataset column order
# ---------------------------------------------------------------------

FINAL_COLUMNS = [
    "Date",
    "Bitcoin_Price",
    "Bitcoin_Market_Cap",
    "Bitcoin_Volume",
    "Ethereum_Price",
    "Ethereum_Market_Cap",
    "Ethereum_Volume",
    "Bitcoin_Search_Interest",
    "Ethereum_Search_Interest",
    "Fear_Greed_Index",
    "Market_Sentiment",
    "Market_Sentiment_Encoded",
]