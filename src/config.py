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


# ---------------------------------------------------------------------
# Statistical analysis (Chapter 4.7): stationarity, correlation, causality
# ---------------------------------------------------------------------

CAUSALITY_RESULTS_DIR = RESULTS_DIR / "causality_analysis"

# All nine variables tested for stationarity.
STATIONARITY_VARIABLES = [
    "Bitcoin_Price",
    "Ethereum_Price",
    "Bitcoin_Volume",
    "Ethereum_Volume",
    "Bitcoin_Market_Cap",
    "Ethereum_Market_Cap",
    "Bitcoin_Search_Interest",
    "Ethereum_Search_Interest",
    "Fear_Greed_Index",
]

# Google Trends search interest and the Fear & Greed Index are the only
# variables treated as external factors in this dissertation (Section 3.3.3).
# Market capitalisation and trading volume are market variables, not
# external ones, and market capitalisation is additionally excluded from
# correlation and Granger causality because it is calculated directly from
# price (r ~ 0.9997), which would add no genuine information.
EXTERNAL_FACTORS = [
    "Bitcoin_Search_Interest",
    "Ethereum_Search_Interest",
    "Fear_Greed_Index",
]

PRICE_VARIABLES = ["Bitcoin_Price", "Ethereum_Price"]

# Variables used in the correlation analysis (market cap excluded, as above).
CORRELATION_VARIABLES = [
    "Bitcoin_Price",
    "Ethereum_Price",
    "Bitcoin_Volume",
    "Ethereum_Volume",
] + EXTERNAL_FACTORS

# Granger causality: each coin's return is a target, tested against the
# external factors. Search interest is tested only against its own asset;
# the Fear & Greed Index, being market-wide, is tested against both.
GRANGER_TARGET_VARIABLES = PRICE_VARIABLES
GRANGER_PREDICTOR_VARIABLES = EXTERNAL_FACTORS

GRANGER_PAIRS = [
    ("Bitcoin_Search_Interest", "Bitcoin_Price"),
    ("Ethereum_Search_Interest", "Ethereum_Price"),
    ("Fear_Greed_Index", "Bitcoin_Price"),
    ("Fear_Greed_Index", "Ethereum_Price"),
]

# Reader-friendly labels used only when plotting, never for calculations.
DISPLAY_LABELS = {
    "Bitcoin_Price": "Bitcoin Return",
    "Ethereum_Price": "Ethereum Return",
    "Bitcoin_Volume": "Bitcoin Volume Change",
    "Ethereum_Volume": "Ethereum Volume Change",
    "Bitcoin_Search_Interest": "Bitcoin Search Interest Change",
    "Ethereum_Search_Interest": "Ethereum Search Interest Change",
    "Fear_Greed_Index": "Fear and Greed Change",
}


# ---------------------------------------------------------------------
# Feature selection and feature engineering (Chapter 4.8 / Section 06)
# ---------------------------------------------------------------------

FEATURE_RESULTS_DIR = RESULTS_DIR / "feature_engineering"
FEATURES_DIR = PROCESSED_DATA_DIR / "features"

# The lag order established by the VAR/BIC analysis in the causality stage
# (05_Correlation_Causality_analysis.ipynb). Reused here as the single
# evidence-selected lag for any predictor carried forward as a candidate
# feature, rather than re-deriving or re-choosing a lag in this notebook.
EVIDENCE_SELECTED_LAG = 2

# Conventional comparison lags with no direct statistical test behind them
# in this project (no ACF/PACF analysis exists in 04_EDA.ipynb). Lag 1 is
# the standard one-step autoregressive/naive-baseline lag. Lag 7 is a
# conventional short-horizon comparison lag, included as standard
# time-series practice — NOT because a weekly pattern was established for
# this data. Cryptocurrency markets trade continuously with no weekend
# close, and no day-of-week or calendar effect was found anywhere in
# 04_EDA.ipynb, so "weekly market structure" is not a valid justification
# here and lag 7 should be read as arbitrary-but-conventional, on the same
# footing as lag 1, not as evidence-based.
CONVENTIONAL_LAGS = [1, 7]

# The only rolling window with a documented precedent in this project:
# 04_EDA.ipynb, Section "Temporal Behaviour", explicitly uses a 30-day
# window for rolling correlation and rolling price volatility. Reused
# directly here for return rolling statistics. For volume rolling
# statistics specifically, this is an extension by analogy rather than a
# literal precedent — the EDA's 30-day window was applied to price-based
# correlation and volatility, not to volume — and this distinction is
# kept explicit in the feature metadata rationale.
ROLLING_WINDOW_EDA_PRECEDENT = 30

# A short trailing window with no EDA precedent, used only for the
# optional external-candidate and ablation feature sets, and labelled as
# a conventional (not evidence-based) choice wherever it is used.
ROLLING_WINDOW_SHORT = 7

# Per-asset variable groupings used to build asset-specific feature sets.
ASSET_VARIABLES = {
    "Bitcoin": {
        "price": "Bitcoin_Price",
        "volume": "Bitcoin_Volume",
        "search_interest": "Bitcoin_Search_Interest",
    },
    "Ethereum": {
        "price": "Ethereum_Price",
        "volume": "Ethereum_Volume",
        "search_interest": "Ethereum_Search_Interest",
    },
}

# Sentiment variables available for the ablation-only feature set.
# Market_Sentiment_Encoded is the ordinal encoding already created during
# preprocessing (03_Data-preprocessing.ipynb) and is reused as-is rather
# than re-encoded here.
SENTIMENT_SCORE_VARIABLE = "Fear_Greed_Index"
SENTIMENT_ORDINAL_VARIABLE = "Market_Sentiment_Encoded"