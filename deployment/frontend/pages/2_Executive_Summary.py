"""Executive Summary page -- plain-language forecast summary for a
supervisor, examiner, or non-technical stakeholder. Reads the same
st.session_state result the Forecast page already populates -- no
duplicated prediction logic, no second inference path.

Note on "confidence": this project's models don't produce a calibrated
prediction interval or ensemble variance -- there's no real uncertainty
quantification to report. Rather than invent a number, this page reports
the model's real, saved historical hit rate (how often its directional
calls have been correct on the test set) and labels it honestly as
that, not as confidence in this specific prediction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styles import inject

import pandas as pd
import streamlit as st
from api_client import get_decision_summary
import reports

st.set_page_config(page_title="Executive summary", layout="wide")
inject()
st.title("Executive summary")
st.caption("Plain-language forecast summary -- for a supervisor, examiner, or non-technical reader")

if "last_result" not in st.session_state:
    st.info("No forecast generated yet. Go to the Forecast page, upload a CSV, and generate a prediction first.")
    st.stop()

result = st.session_state["last_result"]
preview_df = st.session_state.get("last_preview")

direction = "up" if result["predicted_return"] > 0 else "down"
pct = abs(result["predicted_return"]) * 100

st.markdown(f"### {result['asset']} is forecast to move **{direction}** by **{pct:.2f}%** over the next {result['horizon']} day(s)")

k1, k2, k3, k4 = st.columns(4)
if preview_df is not None:
    price_col = f"{result['asset']}_Price"
    if price_col in preview_df.columns:
        k1.metric("Current price", f"${preview_df[price_col].iloc[-1]:,.2f}")
k2.metric("Predicted price", f"${result['predicted_price']:,.2f}")
k3.metric("Model used", result["model"])
with k4:
    st.caption("Feature configuration")
    st.markdown(f"**{reports.display_feature_set(result['feature_set'])}**")

st.markdown("---")
st.subheader("How reliable is this model, historically")
try:
    decision_df = pd.DataFrame(get_decision_summary())
    match = decision_df[(decision_df["Asset"] == result["asset"]) & (decision_df["Horizon (days)"] == result["horizon"])]
    if len(match):
        row = match.iloc[0]
        st.metric("Historical hit rate (this asset/horizon's best model)", f"{row['Hit Rate (%)']:.1f}%")
        st.caption(
            "Hit rate = how often the recommended model correctly predicted the direction of "
            "price movement on real, held-out test data -- not a confidence score for this "
            "specific prediction. This project's models don't produce a calibrated prediction "
            "interval, so no such number is reported here."
        )
except Exception:
    pass

st.markdown("---")
st.subheader("Plain-language interpretation")
st.markdown(
    f"The **{result['model']}** model, trained on the **{reports.display_feature_set(result['feature_set'])}** feature "
    f"configuration using **{result['representation'].lower()}** features, predicts "
    f"**{result['asset']}** will move {direction} by approximately **{pct:.2f}%** over the "
    f"next **{result['horizon']} day(s)**. This is based on {result.get('n_rows_uploaded', 'the uploaded')} "
    f"rows of historical data ending {result.get('date_range', ['', 'the most recent date'])[1]}."
)

st.markdown("---")
st.subheader("Research evidence summary")
st.markdown(
    "Correlation and Granger causality analysis (see Research evidence) found only weak "
    "statistical relationships between price and external variables (Google Trends, Fear & "
    "Greed Index) -- consistent with this project's broader finding that external signals "
    "provide limited, inconsistent forecasting improvement over market data alone."
)

st.markdown("---")
st.subheader("Limitations")
st.markdown(
    "- This forecast reflects patterns in historical data and is not investment advice\n"
    "- No calibrated confidence interval is available for this specific prediction\n"
    "- Cryptocurrency markets are highly volatile; past patterns may not hold during regime changes\n"
    "- Naive baseline is excluded from this CSV-upload workflow (no trainable parameters); ARIMA is supported via a fresh statistical fit, not a saved model -- see Model comparison for details"
)

st.markdown("---")
try:
    decision_row_dict = match.iloc[0].to_dict() if len(match) else None
except (NameError, IndexError):
    decision_row_dict = None
st.download_button(
    "Download executive report (HTML)",
    reports.build_executive_report_html(result, preview_df, decision_row_dict),
    "executive_forecast_report.html", "text/html",
)
