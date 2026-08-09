"""Feature importance comparison page -- the full RF-vs-XGBoost
Permutation Feature Importance (PFI) grid across every real trained
(asset, horizon, feature configuration) combination. This is the
genuine "feature importance analysis for the engineered features,
compared across different ML methods" view.

Replaces an earlier SHAP-based version of this page. SHAP was removed
from this project entirely after a real, reproducible kernel
instability was traced to TreeExplainer in this environment. PFI is
model-agnostic (only ever calls model.predict()) and never showed that
instability -- see run_feature_importance_grid.py's docstring.

Real, saved data from run_feature_importance_grid.py -- not computed
live (48 permutation-importance experiments is not a page-load-sized
amount of work, even though each one is fast).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styles import inject

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api_client import get_feature_importance_comparison, get_feature_importance_agreement_summary

st.set_page_config(page_title="Feature importance comparison", layout="wide")
inject()
st.title("Feature importance comparison")
st.caption(
    "Engineered-feature importance compared across Random Forest and XGBoost -- every real "
    "trained asset/horizon/feature-configuration combination, using Permutation Feature "
    "Importance (PFI)."
)
st.info(
    "This covers every (asset, horizon, feature configuration) this project trains a Random "
    "Forest and an XGBoost model for -- 2 assets x 3 horizons x 4 feature configurations = 24 "
    "configurations, both models each. LSTM isn't included -- there is no engineered feature "
    "matrix to attribute importance to for a model that reads encoded/sequence input the way "
    "LSTM does here. Importance is Permuted RMSE minus Baseline RMSE: positive means the model "
    "actually depends on that feature; negative values are kept, not hidden -- they mean "
    "shuffling that feature happened to slightly help RMSE on this test set."
)

try:
    grid = pd.DataFrame(get_feature_importance_comparison())
    agreement = pd.DataFrame(get_feature_importance_agreement_summary())
except Exception:
    st.warning(
        "The feature importance grid hasn't been generated yet. Run `run_feature_importance_grid.py` "
        "from the project root, then reload this page."
    )
    st.stop()

FSET_DISPLAY = {
    "primary": "Market Only", "candidate": "Market + Google Trends",
    "ablation": "Market + Fear & Greed Index", "combined": "Market + Google Trends + Fear & Greed Index",
}
grid["feature_set_display"] = grid["feature_set_key"].map(FSET_DISPLAY)
agreement["feature_set_display"] = agreement["feature_set_key"].map(FSET_DISPLAY)

RF_COL, XGB_COL = "mean_importance_Random Forest", "mean_importance_XGBoost"

st.markdown("---")
st.subheader("Where do Random Forest and XGBoost agree most / least?")
st.caption(
    "Spearman rank correlation between the two methods' full feature rankings, for every real "
    "configuration -- higher means the two methods ranked engineered features more similarly, "
    "not just picked the same handful."
)

heatmap_df = agreement.copy()
heatmap_df["config_label"] = heatmap_df["asset"] + " " + heatmap_df["horizon"].astype(str) + "d"
pivot = heatmap_df.pivot(index="feature_set_display", columns="config_label", values="spearman_rho")
pivot = pivot.reindex(["Market Only", "Market + Google Trends", "Market + Fear & Greed Index",
                        "Market + Google Trends + Fear & Greed Index"])

fig_heatmap = go.Figure(data=go.Heatmap(
    z=pivot.values, x=pivot.columns, y=pivot.index,
    colorscale="RdYlGn", zmin=-1, zmax=1,
    text=pivot.values.round(2), texttemplate="%{text}",
    colorbar=dict(title="Spearman ρ"),
))
fig_heatmap.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig_heatmap, use_container_width=True)

mean_rho = agreement["spearman_rho"].mean()
mean_top10 = agreement["top_10_overlap"].mean()
st.caption(
    f"Mean Spearman rank correlation across all 24 configurations: **{mean_rho:.3f}**. "
    f"Mean top-10 feature overlap: **{mean_top10:.1f} / 10**. Agreement is real but modest -- "
    "the two models capture similar aggregate signal without ranking individual features the "
    "same way, which is expected given how differently the two algorithms are regularized."
)

st.markdown("---")
st.subheader("Compare a specific configuration")

col1, col2, col3 = st.columns(3)
with col1:
    asset = st.selectbox("Asset", sorted(grid["asset"].unique()))
with col2:
    horizon = st.selectbox("Horizon (days)", sorted(grid["horizon"].unique()))
with col3:
    fset_display = st.selectbox("Feature configuration", list(FSET_DISPLAY.values()))
fset_key = {v: k for k, v in FSET_DISPLAY.items()}[fset_display]

subset = grid[(grid["asset"] == asset) & (grid["horizon"] == horizon) & (grid["feature_set_key"] == fset_key)]

if subset.empty:
    st.info("No data for this combination.")
else:
    subset = subset.sort_values(RF_COL, ascending=False, key=abs)
    top_n = st.slider("Show top N features", 5, min(25, len(subset)), 10)
    plot_df = subset.head(top_n)

    melted = plot_df.melt(
        id_vars=["feature"], value_vars=[RF_COL, XGB_COL],
        var_name="model", value_name="mean_importance",
    )
    melted["model"] = melted["model"].str.replace("mean_importance_", "")
    fig = px.bar(melted, x="mean_importance", y="feature", color="model", orientation="h", barmode="group",
                 color_discrete_map={"Random Forest": "#2ca02c", "XGBoost": "#ff7f0e"})
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(template="plotly_white", height=max(360, top_n * 32),
                       margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    row = agreement[(agreement["asset"] == asset) & (agreement["horizon"] == horizon)
                     & (agreement["feature_set_key"] == fset_key)]
    if len(row):
        rho = row.iloc[0]["spearman_rho"]
        rho_display = f"{rho:.3f}" if pd.notna(rho) else "not computed (too few comparable features)"
        st.caption(f"Spearman rank correlation for this configuration: **{rho_display}** "
                   f"({int(row.iloc[0]['top_10_overlap'])} of 10 top features shared).")

    st.markdown("**Biggest disagreements between the two methods for this configuration**")
    disagreement = subset.sort_values("abs_difference", ascending=False).head(5)
    st.dataframe(
        disagreement[["feature", RF_COL, XGB_COL, "abs_difference", "rank_Random Forest", "rank_XGBoost"]]
        .rename(columns={RF_COL: "Random Forest", XGB_COL: "XGBoost"}),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")
st.subheader("Full grid")
st.caption("Every real (asset, horizon, feature configuration, feature) row -- 24 configurations, both models.")
display_cols = ["asset", "horizon", "feature_set_display", "feature", RF_COL, XGB_COL, "abs_difference"]
st.dataframe(
    grid[display_cols].rename(columns={RF_COL: "Random Forest", XGB_COL: "XGBoost"}),
    use_container_width=True, hide_index=True,
)

# Full HTML report -- heatmap + selected-configuration chart + disagreement
# table + full grid, self-contained (Plotly via CDN, no external files
# needed to open it later). Same pattern as the other pages' reports.
_disagreement_html = ""
if not subset.empty:
    _disagreement_html = f"""
<h2>Selected configuration: {asset}, {horizon}-day, {fset_display}</h2>
{fig.to_html(include_plotlyjs="cdn", full_html=False)}
<h3>Biggest disagreements for this configuration</h3>
{disagreement[["feature", RF_COL, XGB_COL, "abs_difference", "rank_Random Forest", "rank_XGBoost"]].rename(columns={RF_COL: "Random Forest", XGB_COL: "XGBoost"}).to_html(index=False, na_rep="")}
"""

_nan_note_html = ""
if agreement["spearman_rho"].isna().any():
    n_nan = agreement["spearman_rho"].isna().sum()
    _nan_note_html = f"""
<div class="note"><p><em>{n_nan} configuration(s) show no Spearman value (blank cell in the heatmap) --
this happens when one model's importance ranks have no variation to correlate against (e.g. every
feature tied at rank 1, meaning that model showed zero measurable dependence on any individual
feature for that configuration). This is retained as a real result, not hidden or forced to a
number.</em></p></div>
"""

_html_report = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Feature Importance Comparison Report</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }}
h1, h2, h3 {{ color: #111; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
th {{ background: #f2f2f2; }}
.note {{ background: #eef4fb; padding: 12px 16px; border-radius: 6px; }}
</style></head><body>
<h1>Feature Importance Comparison Report</h1>
<p>Engineered-feature importance compared across Random Forest and XGBoost -- every real trained
asset/horizon/feature-configuration combination (24 configurations, both models), using
Permutation Feature Importance.</p>
<div class="note"><p>Importance is Permuted RMSE minus Baseline RMSE: positive means the model
depends on that feature; negative values are kept, not hidden.</p></div>

<h2>Where do Random Forest and XGBoost agree most / least?</h2>
{fig_heatmap.to_html(include_plotlyjs="cdn", full_html=False)}
<p>Mean Spearman rank correlation across all 24 configurations: <b>{mean_rho:.3f}</b>. Mean
top-10 feature overlap: <b>{mean_top10:.1f} / 10</b>.</p>
{_nan_note_html}
{_disagreement_html}

<h2>Full grid</h2>
{grid[display_cols].rename(columns={RF_COL: "Random Forest", XGB_COL: "XGBoost"}).to_html(index=False, na_rep="")}

</body></html>"""

st.download_button("Download full report (HTML)", _html_report, "feature_importance_comparison_report.html", "text/html")
