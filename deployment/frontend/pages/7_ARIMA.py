"""ARIMA page -- dedicated, since ARIMA doesn't fit the saved-model
pattern the other three models use (no fitted object is saved for it;
it refits on a growing history window at each step, using the order
already selected during model development). It IS available on the
Forecast page's CSV upload -- as a fresh fit, not a loaded model. Shows
its real configuration and role here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from api_client import get_arima_info, get_decision_summary

st.set_page_config(page_title="ARIMA", layout="wide")
st.title("ARIMA")
st.caption("Classical statistical baseline -- available for CSV upload inference as a fresh fit, explained below")

st.info(
    "ARIMA has no saved fitted model like the other three -- it refits on a "
    "growing history window at each forecasting step (see `run_arima()` / "
    "`arima_rolling()` in `src/models/arima_model.py`), using the order "
    "already selected during model development. It's still available on the "
    "Forecast page's CSV upload, just as a fresh fit rather than a loaded "
    "model. This page shows its real configuration and historical performance."
)

col1, col2 = st.columns(2)
with col1:
    asset = st.selectbox("Cryptocurrency", ["Bitcoin", "Ethereum"], key="arima_asset")
with col2:
    horizon = st.selectbox("Forecast horizon (days)", [1, 7, 14], key="arima_horizon")

try:
    info = get_arima_info(asset, horizon)
except Exception:
    st.stop()

st.markdown("---")
p, d, q = info["selected_order"]
c1, c2, c3 = st.columns(3)
c1.metric("Order (p, d, q)", f"({p}, {d}, {q})")
with c2:
    st.caption("Training period")
    st.markdown(f"**{info['training_period'][0]} to {info['training_period'][1]}**")
with c3:
    st.caption("Test period")
    st.markdown(f"**{info['test_period'][0]} to {info['test_period'][1]}**")

st.caption(info["limitation"])

st.markdown("---")
st.subheader("How ARIMA compares against the other models")
try:
    decision_df = get_decision_summary()
    for row in decision_df:
        if row["Asset"] == asset and row["Horizon (days)"] == horizon:
            if row["Best Model"] == "ARIMA":
                st.success(f"ARIMA is the best-performing model for {asset} at {horizon} day(s): {row['Practical Recommendation']}")
            else:
                st.info(f"For {asset} at {horizon} day(s), {row['Best Model']} performed best -- see Model comparison for the full picture.")
except Exception:
    pass

st.markdown("---")
st.subheader("Strengths and limitations")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        "**Strengths**\n"
        "- No feature engineering required -- works directly on the price series\n"
        "- Computationally cheap, no training infrastructure needed\n"
        "- A properly specified order is a genuinely strong baseline for near-random-walk series"
    )
with col2:
    st.markdown(
        "**Limitations**\n"
        "- Univariate -- cannot use Google Trends, Fear & Greed, or any external signal\n"
        "- No saved fitted model -- refits fresh at each step (including on the Forecast page), unlike the other four models\n"
        "- Order selection is fixed per (horizon, asset) combination, not adaptive within a forecast run"
    )
