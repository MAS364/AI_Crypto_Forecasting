"""About page -- project context, methodology, and honest scope notes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from styles import inject

st.set_page_config(page_title="About", layout="wide")
inject()
st.title("About this project")

st.markdown(
    """
**AI for Cryptocurrency Time Series Analysis: Correlation, Causality, and Predictive Modelling**

MSc Data Science dissertation, University of Surrey. Forecasts Bitcoin and Ethereum prices
using market data, Google Trends search interest, and the Crypto Fear & Greed Index, across
three forecast horizons (1, 7, and 14 days ahead).
"""
)

st.markdown("---")
st.subheader("Pipeline")
st.markdown(
    """
Historical market data and external factors flow through data preparation, EDA, correlation
and Granger causality analysis, feature engineering (four configurations), then split into
two parallel branches -- original engineered features and 8-dimensional autoencoder-encoded
features -- both feeding five forecasting models (Naive, ARIMA, Random Forest, XGBoost, LSTM),
model evaluation, best-model selection, permutation feature importance, and finally this inference layer.
"""
)

with st.expander("Research objectives"):
    st.markdown(
        "**Research questions**\n"
        "- RQ1 -- Can ML/DL models beat classical statistical models?\n"
        "- RQ2 -- Does adding Google Trends and Fear & Greed improve forecasts?\n"
        "- RQ3 -- Which model is most accurate overall?\n"
        "- RQ4 -- How do prices relate to these external signals?\n"
        "- RQ5 -- Do these signals actually predict prices (Granger causality)?\n\n"
        "**Also included**\n"
        "- A naive baseline -- the simple reference point every model must beat\n"
        "- Autoencoder-encoded features -- an exploratory extra, not one of the 5 core questions\n\n"
        "See the Model comparison and Research evidence pages for the actual evidence "
        "behind each answer."
    )

st.markdown("---")
st.subheader("What this deployment does, and doesn't, do")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        "**Does**\n"
        "- Loads saved, already-trained models -- never retrains\n"
        "- Runs the exact same feature engineering used in training\n"
        "- Supports Random Forest, XGBoost, LSTM, and ARIMA (as a fresh fit, not a loaded model) for CSV-upload inference\n"
        "- Shows real, saved evaluation and permutation feature importance results"
    )
with col2:
    st.markdown(
        "**Doesn't**\n"
        "- Support Naive for CSV upload (no fitted parameters to run -- see Model comparison for its role as a baseline)\n"
        "- Compute permutation importance for LSTM, ARIMA, or Naive -- only Random Forest and XGBoost are covered (see Feature importance comparison)\n"
        "- Fetch live market data automatically -- you provide the CSV"
    )

st.markdown("---")
st.subheader("Architecture")
st.markdown(
    """
Two layers: a FastAPI backend (`deployment/app.py`) that wraps the project's existing
`src/` modules, and this Streamlit frontend, which talks to the backend exclusively over
HTTP -- never importing `src/` directly. This keeps the two genuinely decoupled: the API
can be used standalone (see `/docs` for interactive documentation), and the frontend
never needs to know how a prediction is actually produced.
"""
)

st.caption("Source: github.com -- see the project README for installation, training, and deployment instructions.")
