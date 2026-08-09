"""Forecast page -- the core CSV upload inference workflow.

Upload -> validate -> preprocess -> feature engineer -> predict, all via
HTTP calls to the FastAPI backend (deployment/app.py). This page never
imports src/ directly.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styles import inject

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from api_client import (
    get_csv_requirements, predict_csv, ValidationErrors,
    get_decision_summary_optional, get_full_metrics_optional, get_feature_importance_for_model_optional,
)
import reports

# EDIT THIS once the dataset is uploaded to GitHub -- the raw file URL,
# not the GitHub page URL (raw.githubusercontent.com/<user>/<repo>/<branch>/<path>).
# Example: "https://raw.githubusercontent.com/mas364/AI_Crypto_Forecasting/main/data/processed/merged_dataset.csv"
SAMPLE_DATASET_GITHUB_URL = "https://raw.githubusercontent.com/MAS364/AI_Crypto_Forecasting_Data/main/merged_dataset.csv"

st.set_page_config(page_title="Forecast", layout="wide")
inject()
st.title("Forecast")
st.caption("Upload historical data and generate a forecast from a saved model")

col1, col2, col3, col4 = st.columns(4)
with col1:
    asset = st.selectbox("Cryptocurrency", ["Bitcoin", "Ethereum"])
with col2:
    MODEL_CATEGORIES = {
        "ARIMA": "Statistical",
        "Random Forest": "Machine learning",
        "XGBoost": "Machine learning",
        "LSTM": "Deep learning",
    }
    model_options = ["ARIMA", "Random Forest", "XGBoost", "LSTM"]  # Statistical -> ML -> DL
    model = st.selectbox(
        "Model", model_options,
        index=1,  # default to Random Forest, matching what wins most real Decision Summary entries
        format_func=lambda m: f"{MODEL_CATEGORIES[m]} \u2014 {m}",
    )

if model == "ARIMA":
    st.info(
        "ARIMA has no saved fitted model like the other three -- it fits a fresh "
        "statistical model on your uploaded data, using the order already selected "
        "during model development. One real fit per request, not model training or tuning."
    )
with col3:
    horizon = st.selectbox("Forecast horizon (days)", [1, 7, 14])
with col4:
    # Same 4 configurations used everywhere else in the app (Model
    # comparison's filters, the metrics table's feature_set column).
    # "Auto-detect" preserves the old behavior (richest configuration the
    # upload's columns support) as the default -- an explicit choice here
    # is validated against the upload server-side (csv_inference.py's
    # _validate_requested_feature_set) and 422s with a clear message if
    # the required columns aren't present, rather than silently
    # substituting a different configuration.
    FEATURE_SET_DISPLAY_TO_KEY = {
        "Market Only": "primary",
        "Market + Google Trends": "candidate",
        "Market + Fear & Greed Index": "ablation",
        "Market + Google Trends + Fear & Greed Index": "combined",
    }
    feature_set_choice = st.selectbox(
        "Feature configuration", ["Auto-detect (recommended)"] + list(FEATURE_SET_DISPLAY_TO_KEY.keys()),
        disabled=(model == "ARIMA"),
        help="ARIMA is univariate and always uses Market Only, regardless of this setting." if model == "ARIMA" else
             "Auto-detect picks the richest configuration your upload's columns support. Choose one explicitly to force a specific configuration instead.",
    )
requested_feature_set = None if (model == "ARIMA" or feature_set_choice == "Auto-detect (recommended)") else FEATURE_SET_DISPLAY_TO_KEY[feature_set_choice]

# Representation was previously hardcoded to "Original" on every request
# -- the backend (run_predict -> src/models/autoencoder.py's
# predict_encode()) already supports Autoencoder-Encoded end to end, the
# page just never exposed it. ARIMA and Naive were never run through the
# autoencoder branch during model development (see Model comparison's
# "Original vs. autoencoder-encoded features" section), so it's forced
# to Original for ARIMA here, matching that same exclusion.
representation_choice = st.selectbox(
    "Representation", ["Original", "Autoencoder-Encoded"],
    disabled=(model == "ARIMA"),
    help=(
        "ARIMA was never run through the autoencoder branch during model development -- always Original."
        if model == "ARIMA" else
        "Original: the 15-25 named engineered features (price lags, volume, Google Trends, "
        "Fear & Greed, etc.) -- each one interpretable on its own. Autoencoder-Encoded: those "
        "same features compressed into 8 unnamed numeric dimensions by a trained autoencoder, "
        "an exploratory extra tried during model development (RQ addition, not one of the 5 "
        "core research questions)."
    ),
)
representation = "Original" if model == "ARIMA" else representation_choice
if representation == "Original":
    st.caption(
        "Using the original, named engineered features. This is the representation the "
        "Feature Importance Comparison page covers."
    )
else:
    st.caption(
        "Using the 8 autoencoder-encoded dimensions instead of the named features. Feature "
        "importance analysis (Permutation Feature Importance) is not available for this "
        "representation -- the encoded dimensions have no individual real-world meaning to "
        "attribute importance to, only the original named features do. See the Feature "
        "Importance Comparison page for the Original-representation results instead."
    )

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_dataset_csv(url: str) -> bytes:
    """Fetches the public dataset CSV from GitHub. Cached for an hour so a
    normal Streamlit rerun (any widget interaction reruns this whole
    script top to bottom) doesn't refetch the file every time."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.content


try:
    dataset_bytes = _fetch_dataset_csv(SAMPLE_DATASET_GITHUB_URL)
    st.download_button(
        "Download project dataset (CSV)", data=dataset_bytes,
        file_name="merged_dataset.csv", mime="text/csv",
    )
except Exception:
    st.warning("Dataset download is temporarily unavailable -- please try again shortly.")

with st.expander("CSV requirements", expanded=False):
    try:
        reqs = get_csv_requirements(asset)
        st.markdown(f"**Required columns:** {', '.join(reqs['required_columns'])}")
        st.markdown(f"**Optional columns:** {', '.join(reqs['optional_columns'])}")
        st.markdown(f"**Minimum rows:** {reqs['min_rows']} (30-day rolling features need real history)")
        st.markdown(f"**Note:** {reqs['note']}")
        st.markdown(f"**Not supported here:** {', '.join(f'{k} ({v})' for k, v in reqs['unsupported_models'].items())}")
    except Exception:
        st.caption("Connect to the API to see live requirements.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    preview_df = pd.read_csv(uploaded)
    st.caption(f"{len(preview_df)} rows, {len(preview_df.columns)} columns")
    st.dataframe(preview_df.head(5), use_container_width=True)
    uploaded.seek(0)

    if st.button("Generate forecast", type="primary"):
        with st.spinner("Validating, engineering features, and predicting..."):
            try:
                t0 = time.time()
                result = predict_csv(uploaded.read(), uploaded.name, asset, model, horizon, representation=representation, feature_set=requested_feature_set)
                st.session_state["last_inference_time"] = time.time() - t0
            except ValidationErrors as e:
                st.error("Upload didn't pass validation:")
                for problem in e.problems:
                    st.markdown(f"- {problem}")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.session_state["last_result"] = result
        st.session_state["last_preview"] = preview_df
        st.session_state["_just_generated"] = True

if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    preview_df = st.session_state["last_preview"]

    just_generated = st.session_state.pop("_just_generated", False)

    st.markdown('<div id="forecast-results-anchor"></div>', unsafe_allow_html=True)

    if just_generated:
        st.toast("Forecast generated -- see results below", icon="\u2705")
        # Scroll the main app viewport (not this component's iframe) down to
        # the results anchor. Only fires on the run that just generated a
        # result, so navigating elsewhere on the page later doesn't yank
        # the user back down.
        components.html(
            """
            <script>
                setTimeout(function() {
                    var doc = window.parent.document;
                    var el = doc.getElementById('forecast-results-anchor');
                    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
                }, 200);
            </script>
            """,
            height=0,
        )

    st.success("\u2705 Forecast generated")
    if result.get("inference_note"):
        st.info(result["inference_note"])
    if result.get("data_quality_note"):
        st.warning(f"\u26a0\ufe0f {result['data_quality_note']}")
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted price", f"${result['predicted_price']:,.2f}")
    k2.metric("Model", result["model"])
    with k3:
        st.caption("Feature configuration")
        st.markdown(f"**{reports.display_feature_set(result['feature_set'])}**")
    k4.metric("Representation", result["representation"])

    price_col = f"{asset}_Price"
    if price_col in preview_df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(preview_df["Date"]), y=preview_df[price_col],
            mode="lines", name="Historical price", line=dict(color="#4c72b0"),
        ))
        last_date = pd.to_datetime(preview_df["Date"]).max()
        forecast_date = last_date + pd.Timedelta(days=horizon)
        fig.add_trace(go.Scatter(
            x=[last_date, forecast_date], y=[preview_df[price_col].iloc[-1], result["predicted_price"]],
            mode="lines+markers", name="Forecast", line=dict(color="#d62728", dash="dash"),
        ))
        fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), template="plotly_white",
                           xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast summary")
    last_observation = result.get("date_range", ["", ""])[1]
    parsed_last_observation = pd.to_datetime(last_observation, errors="coerce")
    if pd.isna(parsed_last_observation):
        forecast_target = "N/A"
    else:
        forecast_target = (parsed_last_observation + pd.Timedelta(days=horizon)).date().isoformat()
    summary_df = pd.DataFrame([{
        "Last observation": last_observation, "Forecast horizon": f"{horizon} day{'s' if horizon != 1 else ''}",
        "Forecast target": forecast_target, "Predicted return": f"{result['predicted_return']:.4%}",
        "Predicted price": f"${result['predicted_price']:,.2f}",
    }])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Gather optional context for the reports -- each is fetched
    # best-effort; a missing one (e.g. feature importance not yet
    # generated for this exact combination) just means that section is
    # omitted from the reports, not faked. Using the _optional API
    # client variants here, not try/except alone -- a try/except around
    # the hard-stopping versions doesn't actually achieve "best-effort":
    # _get()'s st.stop() halts the rest of this page regardless of a
    # surrounding try/except (confirmed directly -- a 404 here was
    # silently wiping out the entire Downloads section below, with no
    # way to tell without checking).
    decision_df = get_decision_summary_optional()
    decision_row = None
    matches = [r for r in decision_df if r["Asset"] == asset and r["Horizon (days)"] == horizon]
    decision_row = matches[0] if matches else None

    full_metrics = get_full_metrics_optional()
    metrics_row = None
    matches = [r for r in full_metrics if r["asset"] == asset and r["horizon"] == horizon
               and r["model"] == result["model"] and r["split"] == "test"]
    metrics_row = matches[0] if matches else None

    feature_importance = get_feature_importance_for_model_optional(asset, horizon, result["model"], result["feature_set"])

    inference_time = st.session_state.get("last_inference_time")

    st.markdown("---")
    st.subheader("Downloads")
    dl1, dl2, dl3, dl4 = st.columns(4)
    with dl1:
        st.download_button("Forecast CSV", reports.build_forecast_csv(result, preview_df), "forecast_results.csv", "text/csv")
    with dl2:
        st.download_button("Executive report", reports.build_executive_report_html(result, preview_df, decision_row),
                            "executive_forecast_report.html", "text/html")
    with dl3:
        st.download_button("Technical report", reports.build_technical_report_html(result, metrics_row, feature_importance),
                            "technical_forecast_report.html", "text/html")
    with dl4:
        zip_bytes = reports.build_forecast_package_zip(
            result, preview_df, decision_row, metrics_row, feature_importance,
            inference_time_seconds=inference_time,
        )
        st.download_button("Full package (.zip)", zip_bytes, "forecast_package.zip", "application/zip")

    if not feature_importance:
        st.caption(
            "Feature importance is not available for this forecast. Permutation feature "
            "importance is reported for Random Forest and XGBoost models using the "
            "Original feature representation."
        )
