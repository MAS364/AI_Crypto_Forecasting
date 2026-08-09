"""Model comparison page -- interactive, filterable comparison across all
5 models, using the real, saved metrics table (384 rows) -- nothing
recomputed here. Answers: "which model should I trust for this asset and
horizon, and why?" """

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styles import inject

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api_client import get_full_metrics, get_predictions, get_decision_summary

st.set_page_config(page_title="Model comparison", layout="wide")
inject()
st.title("Model comparison")
st.caption("Which forecasting model should you trust for this asset and horizon, and why -- from 07_model_development.ipynb's real, saved metrics")

try:
    metrics = pd.DataFrame(get_full_metrics())
except Exception:
    st.stop()

metrics = metrics[metrics["split"] == "test"]

FSET_KEY_MAP = {"Market Only": "primary", "Market + Google Trends": "candidate",
                 "Market + Fear & Greed Index": "ablation", "Market + Google Trends + Fear & Greed Index": "combined"}

MODEL_CATEGORIES = {
    "Naive": "Baseline",
    "ARIMA": "Statistical",
    "Random Forest": "Machine learning",
    "XGBoost": "Machine learning",
    "LSTM": "Deep learning",
}

# Display-only rename so "Naive" is self-explanatory wherever it appears on
# its own (table rows, chart axes/legends, KPI values) -- never used for
# filtering or matching against the real saved data, which must keep using
# the exact "Naive" string everywhere else in this file.
MODEL_DISPLAY_NAMES = {"Naive": "Naive (Persistence)"}


def _display_model_name(m):
    return MODEL_DISPLAY_NAMES.get(m, m)
MODEL_ORDER = ["Naive", "ARIMA", "Random Forest", "XGBoost", "LSTM"]  # Baseline -> Statistical -> ML -> DL

# Fixed per-model styling used across every chart on this page. Plotly's
# default qualitative palette put Naive/ARIMA in two shades of blue and gave
# Random Forest/XGBoost no visual distinction beyond color -- a real problem
# when two models have near-identical metric values and their lines or bars
# sit on top of each other. Distinct hues plus distinct dash/marker per model
# means overlapping values are still separable without relying on color alone.
MODEL_COLOR_MAP = {
    "Naive": "#7f7f7f",          # gray -- baseline
    "Naive (Persistence)": "#7f7f7f",
    "ARIMA": "#1f77b4",          # blue -- statistical
    "Random Forest": "#ff7f0e",  # orange -- ML
    "XGBoost": "#d62728",        # red -- ML
    "LSTM": "#2ca02c",           # green -- DL
}
MODEL_DASH_MAP = {
    "Naive": "dot",
    "Naive (Persistence)": "dot",
    "ARIMA": "solid",
    "Random Forest": "solid",
    "XGBoost": "dash",
    "LSTM": "solid",
}
MODEL_MARKER_MAP = {
    "Naive": "circle",
    "Naive (Persistence)": "circle",
    "ARIMA": "circle",
    "Random Forest": "square",
    "XGBoost": "diamond",
    "LSTM": "triangle-up",
}

def _model_format(m):
    return f"{MODEL_CATEGORIES.get(m, '')} \u2014 {_display_model_name(m)}"

# --- Section A: Filters ---
st.subheader("Filters")
f1, f2, f3, f4 = st.columns(4)
with f1:
    asset = st.selectbox("Asset", sorted(metrics["asset"].unique()))
with f2:
    horizon = st.selectbox("Horizon (days)", sorted(metrics["horizon"].unique()))
with f3:
    feature_set = st.selectbox("Feature configuration", sorted(metrics["feature_set"].unique()))
with f4:
    representation = st.selectbox("Representation", sorted(metrics["representation"].unique()))

filtered = metrics[
    (metrics["asset"] == asset) & (metrics["horizon"] == horizon)
    & (metrics["feature_set"] == feature_set) & (metrics["representation"] == representation)
]
if representation == "Autoencoder-Encoded" and len(filtered) < 5:
    st.caption(
        "Naive and ARIMA don't appear here because they were never run through the "
        "autoencoder branch -- a deliberate, structural decision from Notebook 07 "
        "(both are excluded there by design). Switch Representation to \"Original\" to see all 5 models."
    )
best_row = filtered.sort_values("RMSE_returns").iloc[0] if len(filtered) else None

# --- Section B: Best Model Summary KPIs ---
st.markdown("---")
st.subheader("Lowest-RMSE model for this selection")
st.caption("Primary ranking criterion: lowest test RMSE.")
if best_row is not None:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Lowest-RMSE model", _display_model_name(best_row["model"]))
    k2.metric("RMSE", f"{best_row['RMSE_returns']:.4f}")
    k3.metric("MAE", f"{best_row['MAE_returns']:.4f}")
    k4.metric("R\u00b2", f"{best_row['R2_returns']:.4f}")
    k5.metric("Hit rate", f"{best_row['Hit_Rate_%']:.1f}%")
    if best_row["R2_returns"] < 0:
        st.caption(
            "A negative R\u00b2 means this model performs worse than simply "
            "predicting the historical average return for this selection."
        )

# --- Section L: Automatic natural-language interpretation ---
if best_row is not None and len(filtered) > 1:
    rank_by_hitrate = filtered.sort_values("Hit_Rate_%", ascending=False)
    best_hitrate_model = rank_by_hitrate.iloc[0]["model"]
    best_hitrate_model_display = _display_model_name(best_hitrate_model)
    also_best_hitrate = " It also achieved the highest hit rate among all evaluated models." if best_hitrate_model == best_row["model"] else f" {best_hitrate_model_display} achieved the highest hit rate instead, worth weighing against RMSE depending on what matters more for your use case."
    st.info(
        f"The **{_display_model_name(best_row['model'])}** model achieved the lowest RMSE ({best_row['RMSE_returns']:.4f}) "
        f"for **{asset}** at the **{horizon}-day** forecast horizon, using the **{feature_set}** "
        f"feature configuration with the **{representation}** representation, among "
        f"{len(filtered)} models evaluated on this exact combination.{also_best_hitrate}"
    )
    st.caption(
        "ARIMA and Naive are univariate/feature-independent baselines. Their results are "
        "repeated across feature-configuration views for comparison and do not use Google "
        "Trends or Fear & Greed predictors."
    )

# --- Section C: Performance table ---
st.markdown("---")
st.subheader("Performance table")
display_cols = ["model", "RMSE_returns", "MAE_returns", "R2_returns", "Hit_Rate_%"]
table_df = filtered[display_cols].sort_values("RMSE_returns").reset_index(drop=True).copy()
table_df["model"] = table_df["model"].map(_display_model_name)
table_df["Rank"] = pd.Series(range(1, len(table_df) + 1), dtype="int64")
table_df = table_df[["Rank", "model", "RMSE_returns", "MAE_returns", "R2_returns", "Hit_Rate_%"]]
search = st.text_input("Search model", "", placeholder="Type to filter rows, e.g. 'Random'")
display_table = table_df[table_df["model"].str.contains(search, case=False)].reset_index(drop=True) if search else table_df
st.dataframe(
    display_table, use_container_width=True, hide_index=True,
    column_order=["Rank", "model", "RMSE_returns", "MAE_returns", "R2_returns", "Hit_Rate_%"],
)
st.caption("Download this table from the Downloads section at the bottom of the page.")

# --- Section D: Switchable metric chart ---
st.markdown("---")
st.subheader("Metric comparison")
metric_choice = st.radio("Metric", ["RMSE", "MAE", "R\u00b2", "Hit rate"], horizontal=True)
metric_col_map = {"RMSE": "RMSE_returns", "MAE": "MAE_returns", "R\u00b2": "R2_returns", "Hit rate": "Hit_Rate_%"}
metric_col = metric_col_map[metric_choice]
ascending = metric_choice not in ("R\u00b2", "Hit rate")  # lower is better for RMSE/MAE, higher is better for R2/Hit rate
chart_df = filtered.assign(model=filtered["model"].map(_display_model_name))
fig = px.bar(chart_df.sort_values(metric_col, ascending=ascending), x="model", y=metric_col, color="model",
             color_discrete_map=MODEL_COLOR_MAP)
if metric_choice == "Hit rate":
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Chance")
fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=30, b=20), showlegend=False, title=metric_choice)
st.plotly_chart(fig, use_container_width=True)
if metric_choice == "Hit rate":
    st.caption(
        "The 50% line is a simple reference, not necessarily this exact selection's actual "
        "balance of positive vs. negative returns -- a mild class imbalance can shift the "
        "true chance level slightly above or below 50%."
    )

# --- Section E: Horizon comparison ---
st.markdown("---")
st.subheader("Performance across forecast horizons")
horizon_data = metrics[(metrics["asset"] == asset) & (metrics["feature_set"] == feature_set) & (metrics["representation"] == representation)]
if len(horizon_data):
    horizon_chart_data = horizon_data.assign(model=horizon_data["model"].map(_display_model_name))
    fig = px.line(
        horizon_chart_data.sort_values("horizon"), x="horizon", y=metric_col, color="model", markers=True,
        color_discrete_map=MODEL_COLOR_MAP,
    )
    for trace in fig.data:
        model_name = trace.name
        trace.line.dash = MODEL_DASH_MAP.get(model_name, "solid")
        trace.marker.symbol = MODEL_MARKER_MAP.get(model_name, "circle")
        trace.marker.size = 9
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=30, b=20),
                       xaxis_title="Forecast horizon (days)", yaxis_title=metric_choice,
                       xaxis=dict(tickmode="array", tickvals=[1, 7, 14]),
                       hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A line chart makes the trend across horizons easy to read -- whether a model's performance degrades, holds steady, or improves as the forecast reaches further ahead. Lines are distinguished by color, dash style, and marker shape, and hovering shows every model's exact value at that horizon -- useful when two models' lines sit close together (e.g. Random Forest and XGBoost often have very similar MAE).")

# --- Section F: Actual vs predicted ---
st.markdown("---")
st.subheader("Actual vs. predicted -- one model")
model_choice = st.selectbox(
    "Model", [m for m in MODEL_ORDER if m in filtered["model"].unique()],
    format_func=_model_format, key="scatter_model",
)
try:
    preds = pd.DataFrame(get_predictions(asset, int(horizon), model_choice, FSET_KEY_MAP.get(feature_set, "primary"), representation))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds["actual_price"], y=preds["predicted_price"], mode="markers", marker=dict(size=6, opacity=0.6)))
    min_v, max_v = preds[["actual_price", "predicted_price"]].min().min(), preds[["actual_price", "predicted_price"]].max().max()
    fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect prediction"))
    fig.update_layout(template="plotly_white", height=400, margin=dict(l=20, r=20, t=30, b=20),
                       xaxis_title="Actual price", yaxis_title="Predicted price", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points on the dashed line would be perfect predictions -- the closer the cloud hugs it, the better calibrated the model.")
except Exception:
    st.info("Predictions not available for this exact combination.")

# --- Section H: Autoencoder comparison ---
st.markdown("---")
st.subheader("Original vs. autoencoder-encoded features")
ae_compare = metrics[(metrics["asset"] == asset) & (metrics["horizon"] == horizon) & (metrics["feature_set"] == feature_set)]
if len(ae_compare["representation"].unique()) > 1:
    fig = px.bar(ae_compare.sort_values(["model", "representation"]), x="model", y="RMSE_returns", color="representation", barmode="group")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=30, b=20), title="RMSE: original vs. autoencoder-encoded features")
    st.plotly_chart(fig, use_container_width=True)

    ae_improvement_rows = []
    for m in ae_compare["model"].unique():
        orig = ae_compare[(ae_compare["model"] == m) & (ae_compare["representation"] == "Original")]
        enc = ae_compare[(ae_compare["model"] == m) & (ae_compare["representation"] == "Autoencoder-Encoded")]
        if len(orig) and len(enc):
            pct = (orig.iloc[0]["RMSE_returns"] - enc.iloc[0]["RMSE_returns"]) / orig.iloc[0]["RMSE_returns"] * 100
            ae_improvement_rows.append({"Model": m, "Improvement % (encoded vs original)": round(pct, 2)})
    if ae_improvement_rows:
        st.dataframe(pd.DataFrame(ae_improvement_rows), use_container_width=True, hide_index=True)
else:
    st.caption("Only one representation available for this feature configuration.")

# --- Section I: Feature configuration comparison ---
st.markdown("---")
st.subheader("Feature configuration comparison")
fc_data = metrics[(metrics["asset"] == asset) & (metrics["horizon"] == horizon) & (metrics["representation"] == representation)]
if len(fc_data):
    pivot = fc_data.assign(model=fc_data["model"].map(_display_model_name)).pivot_table(index="model", columns="feature_set", values="RMSE_returns", aggfunc="first")
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="RdYlGn_r",
        text=pivot.round(4).values, texttemplate="%{text}", colorbar=dict(title="RMSE"),
    ))
    fig.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=30, b=20),
                       title="RMSE by model and feature configuration (lower/greener = better)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A heatmap makes it easy to scan all 4 configurations against all models at once -- a grouped bar chart with this many categories would be harder to compare at a glance.")

# --- Section J: Baseline (Naive) comparison ---
st.markdown("---")
st.subheader("Improvement over Naive baseline")
naive_row = metrics[(metrics["asset"] == asset) & (metrics["horizon"] == horizon) & (metrics["model"] == "Naive")
                     & (metrics["feature_set"] == feature_set) & (metrics["representation"] == "Original")]
if len(naive_row):
    naive_rmse = naive_row.iloc[0]["RMSE_returns"]
    naive_mae = naive_row.iloc[0]["MAE_returns"]
    naive_r2 = naive_row.iloc[0]["R2_returns"]
    naive_hr = naive_row.iloc[0]["Hit_Rate_%"]
    improvement = filtered.copy()
    improvement["RMSE improvement %"] = (naive_rmse - improvement["RMSE_returns"]) / naive_rmse * 100
    improvement["MAE improvement %"] = (naive_mae - improvement["MAE_returns"]) / naive_mae * 100
    improvement["R\u00b2 improvement (pts)"] = improvement["R2_returns"] - naive_r2
    improvement["Hit rate improvement (pts)"] = improvement["Hit_Rate_%"] - naive_hr
    fig = px.bar(improvement.sort_values("RMSE improvement %", ascending=False), x="model", y="RMSE improvement %", color="model",
                 color_discrete_map=MODEL_COLOR_MAP)
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=30, b=20), showlegend=False,
                       title="% RMSE improvement over Naive (positive = better)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(improvement[["model", "RMSE improvement %", "MAE improvement %", "R\u00b2 improvement (pts)", "Hit rate improvement (pts)"]].round(2),
                 use_container_width=True, hide_index=True)
else:
    st.caption("Naive baseline not available for this exact configuration.")

# --- Section K: Model details (ARIMA/Naive absorbed here, not separate pages) ---
st.markdown("---")
st.subheader("Model details")
model_detail_choice = st.selectbox(
    "Select a model for details", MODEL_ORDER,
    format_func=_model_format, key="detail_model",
)

MODEL_INFO = {
    "Naive": {
        "description": "A parameterless persistence rule -- predicts the next period's return will equal the last known period's return. No fitted model, no training.",
        "strengths": "Zero training cost. No feature engineering or external data needed. A genuinely hard baseline to beat for a near-random-walk series.",
        "weaknesses": "Cannot use any external signal, by construction. Assumes the immediate past is the best predictor of the immediate future, which breaks down around regime changes.",
    },
    "ARIMA": {
        "description": "A classical statistical model that refits on a growing history window at each forecasting step, rather than using one saved fitted object.",
        "strengths": "No feature engineering required -- works directly on the price series. Computationally cheap. A properly specified order is a strong baseline for near-random-walk series.",
        "weaknesses": "Univariate -- cannot use Google Trends, Fear & Greed, or any external signal. No saved fitted model, unlike the other four.",
    },
    "Random Forest": {
        "description": "An ensemble of decision trees trained on engineered lag, rolling-statistic, and (optionally) external features.",
        "strengths": "Handles non-linear relationships and feature interactions. Feature importance directly measurable via Permutation Feature Importance. Robust to noisy features.",
        "weaknesses": "Can overfit to spurious patterns in a relatively small training window. No native sense of time ordering beyond what's encoded in lag features.",
    },
    "XGBoost": {
        "description": "A gradient-boosted tree ensemble, same feature inputs as Random Forest, different training approach (sequential error correction vs. independent trees).",
        "strengths": "Often more sample-efficient than Random Forest. Also covered by Permutation Feature Importance, on the same real test set.",
        "weaknesses": "More hyperparameter-sensitive than Random Forest -- some configurations in this project showed near-constant predictions, a real, documented finding from Notebook 07.",
    },
    "LSTM": {
        "description": "A recurrent neural network trained on windowed sequences of the same engineered features, capturing temporal structure directly.",
        "strengths": "Only model that genuinely uses sequence structure across timesteps, not just point-in-time lag features.",
        "weaknesses": "Requires the most data and compute to train. Not covered by the feature importance analysis -- Permutation Feature Importance in this project is scoped to Random Forest and XGBoost, since LSTM has no single engineered feature matrix to attribute individual importance to the same way (it reads windowed sequences, not one row of named features).",
    },
}
info = MODEL_INFO[model_detail_choice]
st.markdown(f"**{info['description']}**")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**Strengths**\n\n{info['strengths']}")
with c2:
    st.markdown(f"**Weaknesses**\n\n{info['weaknesses']}")

decision_row_for_report = None
try:
    decision_df = pd.DataFrame(get_decision_summary())
    match = decision_df[(decision_df["Asset"] == asset) & (decision_df["Horizon (days)"] == horizon)]
    if len(match):
        row = match.iloc[0]
        decision_row_for_report = row.to_dict()
        if row["Best Model"] == model_detail_choice:
            st.success(f"Won for {asset} at {horizon} day(s): {row['Practical Recommendation']}")
        else:
            st.info(f"Lost to {row['Best Model']} for {asset} at {horizon} day(s).")
except Exception:
    pass

if model_detail_choice == "ARIMA":
    try:
        from api_client import get_arima_info
        arima_meta = get_arima_info(asset, int(horizon))
        p, d, q = arima_meta["selected_order"]
        st.caption(f"Real saved order for {asset}, {horizon}d: ({p}, {d}, {q}). {arima_meta['limitation']}")
    except Exception:
        pass
elif model_detail_choice == "Naive":
    try:
        from api_client import get_naive_info
        naive_meta = get_naive_info(asset, int(horizon))
        st.code(naive_meta["rule"], language=None)
        st.caption(naive_meta["limitation"])
    except Exception:
        pass

# --- Section M: Downloads ---
st.markdown("---")
st.subheader("Downloads")
dl1, dl2 = st.columns(2)
with dl1:
    st.download_button("Model comparison CSV", table_df.to_csv(index=False), "model_comparison.csv", "text/csv")
with dl2:
    interpretation_text = (
        f"The {best_row['model']} model achieved the lowest RMSE ({best_row['RMSE_returns']:.4f}) for {asset} "
        f"at the {horizon}-day forecast horizon, using the {feature_set} feature configuration with the "
        f"{representation} representation."
    ) if best_row is not None else "No data available for this selection."
    report_html = f"""<html><head><meta charset="utf-8"><style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 760px; margin: 40px auto; color: #1a1a1a; line-height: 1.6; padding: 0 20px; }}
h1 {{ font-size: 22px; font-weight: 600; border-bottom: 2px solid #1D9E75; padding-bottom: 12px; }}
table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 13px; }}
th {{ color: #666; font-weight: 500; }}
.footer {{ margin-top: 40px; font-size: 11px; color: #999; }}
</style></head><body>
<h1>Model Comparison Report -- {asset}, {horizon}-day horizon</h1>
<p>{interpretation_text}</p>
<table><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R\u00b2</th><th>Hit Rate</th></tr>
{"".join(f"<tr><td>{r['model']}</td><td>{r['RMSE_returns']:.4f}</td><td>{r['MAE_returns']:.4f}</td><td>{r['R2_returns']:.4f}</td><td>{r['Hit_Rate_%']:.1f}%</td></tr>" for _, r in table_df.iterrows())}
</table>
<div class="footer">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} -- Feature configuration: {feature_set} -- Representation: {representation}</div>
</body></html>"""
    st.download_button("Comparison report (HTML)", report_html, "model_comparison_report.html", "text/html")

st.caption("Chart PNGs: use Plotly's built-in camera icon in the top-right corner of any chart above for a publication-quality export.")
