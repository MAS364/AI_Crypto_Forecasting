"""Home page -- product-style dashboard landing page. KPIs and quick
actions first; methodology and research framing live on the About page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from styles import inject

import streamlit as st
from api_client import health_check, get_decision_summary_optional, get_last_training_date

st.set_page_config(page_title="Crypto Forecasting Platform", layout="wide")
inject()

api_up = health_check()

header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("AI-Driven Bitcoin & Ethereum Forecasting")
    st.caption("Statistical, ML, and deep learning models -- compared, evaluated, and explained with explainable AI via permutation feature importance")
with header_col2:
    st.write("")
    if st.button("Start forecasting", type="primary", use_container_width=True):
        st.switch_page("pages/1_Forecast.py")

if "selected_asset" not in st.session_state:
    st.session_state["selected_asset"] = "Bitcoin"

t1, t2 = st.columns(2)
with t1:
    if st.button("Bitcoin", type="primary" if st.session_state["selected_asset"] == "Bitcoin" else "secondary", use_container_width=True):
        st.session_state["selected_asset"] = "Bitcoin"
        st.rerun()
with t2:
    if st.button("Ethereum", type="primary" if st.session_state["selected_asset"] == "Ethereum" else "secondary", use_container_width=True):
        st.session_state["selected_asset"] = "Ethereum"
        st.rerun()

selected_asset = st.session_state["selected_asset"]
st.markdown("---")

summary = get_decision_summary_optional() if api_up else []

asset_rows = [r for r in summary if r["Asset"] == selected_asset]
best_hit_rate_row = max(asset_rows, key=lambda r: r["Hit Rate (%)"], default=None)

last_trained = get_last_training_date() if api_up else None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Models trained", "5", "Naive to LSTM")
k2.metric("Forecast horizons", "3", "1, 7, 14 days")
if best_hit_rate_row:
    k3.metric(f"{selected_asset} best model", best_hit_rate_row["Best Model"], f"{best_hit_rate_row['Hit Rate (%)']:.1f}% hit rate")
else:
    k3.metric(f"{selected_asset} best model", "--")
k4.metric("Last training date", last_trained or "Not available")

st.markdown("---")

if asset_rows:
    st.subheader(f"Best model by horizon -- {selected_asset}")
    cols = st.columns(len(asset_rows))
    for col, row in zip(cols, asset_rows):
        with col:
            st.markdown(
                f"""<div style="border:0.5px solid rgba(128,128,128,0.3);border-radius:8px;padding:10px 12px;">
                <p style="font-size:12px;color:gray;margin:0;">{row['Horizon (days)']}-day</p>
                <p style="font-size:14px;font-weight:600;margin:2px 0 0;">{row['Best Model']}</p>
                <p style="font-size:11px;color:gray;margin:4px 0 0;">{row.get('Feature Configuration', 'N/A')}</p>
                </div>""",
                unsafe_allow_html=True,
            )
else:
    st.info("Connect to the API to see the current best-model summary.")

st.markdown("---")
st.subheader("Explore")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Run a forecast\n\nUpload data, get a prediction", use_container_width=True):
        st.switch_page("pages/1_Forecast.py")
with c2:
    if st.button("See how it thinks\n\nFeature importance comparison", use_container_width=True):
        st.switch_page("pages/6_Feature_Importance_Comparison.py")
with c3:
    if st.button("Compare models\n\nFull performance breakdown", use_container_width=True):
        st.switch_page("pages/3_Model_Comparison.py")

st.markdown("---")
r1, r2 = st.columns(2)
with r1:
    st.link_button("Repository", "https://github.com", use_container_width=True)
with r2:
    st.link_button("Documentation", "https://github.com", use_container_width=True)

if not api_up:
    st.markdown("---")
    st.error("API not reachable. Run `uvicorn deployment.app:app --reload` in a separate terminal.")
