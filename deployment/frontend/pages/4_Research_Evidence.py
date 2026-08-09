"""Research evidence page -- combines correlation, Granger causality, and
(where available) permutation feature importance results into one
evidence view. All from real, saved analysis -- nothing recomputed here."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styles import inject

import pandas as pd
import streamlit as st
from api_client import get_research_evidence, get_feature_importance_comparison

st.set_page_config(page_title="Research evidence", layout="wide")
inject()
st.title("Research evidence")
st.caption("Correlation, Granger causality, and permutation feature importance -- from 05_correlation_and_granger.ipynb and 09_feature_importance_analysis.ipynb")

try:
    evidence = get_research_evidence()
except Exception:
    st.stop()

corr_df = pd.DataFrame(evidence["correlation"])
granger_df = pd.DataFrame(evidence["granger_causality"])
try:
    fi_df = pd.DataFrame(get_feature_importance_comparison())
except Exception:
    fi_df = None

# Display-friendly configuration names -- feature_set_key holds this
# project's internal keys (primary/candidate/ablation/combined), not
# meant for direct display. Applied once, here, before anything renders
# or gets downloaded, so the table and the CSV always agree.
FSET_DISPLAY = {
    "primary": "Market Only", "candidate": "Market + Google Trends",
    "ablation": "Market + Fear & Greed Index", "combined": "Market + Google Trends + Fear & Greed Index",
}
if fi_df is not None and "feature_set_key" in fi_df.columns:
    fi_df["feature_set_key"] = fi_df["feature_set_key"].map(FSET_DISPLAY).fillna(fi_df["feature_set_key"])

tab1, tab2, tab3 = st.tabs(["Correlation", "Granger causality", "Feature evidence matrix"])

with tab1:
    significant = corr_df[corr_df.get("Significant (FDR-corrected)", False) == True]
    st.metric("Statistically significant relationships (FDR-corrected)", len(significant))
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

with tab2:
    n_significant = (granger_df.get("Significant (FDR-corrected)", pd.Series(dtype=bool)) == True).sum()
    st.metric("Significant Granger-causal relationships", int(n_significant))
    st.dataframe(granger_df, use_container_width=True, hide_index=True)
    if n_significant == 0:
        st.info(
            "No significant Granger-causal relationships were found for either external "
            "variable at any lag tested -- consistent with the weak correlation results above."
        )

with tab3:
    if fi_df is not None:
        st.dataframe(fi_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Feature importance results not available yet -- run run_feature_importance_grid.py "
            "to generate the full feature evidence matrix (Random Forest vs XGBoost permutation "
            "importance, per configuration). The download below still includes correlation and "
            "Granger causality, which are already available."
        )


# Combined download -- correlation and Granger causality are always included;
# feature importance is appended only if it's actually been generated, so
# the file's contents always match what "research evidence" claims to
# cover, not just whichever tab happened to render successfully.
combined_frames = [
    corr_df.assign(**{"Evidence type": "Correlation"}),
    granger_df.assign(**{"Evidence type": "Granger causality"}),
]
if fi_df is not None:
    combined_frames.append(fi_df.assign(**{"Evidence type": "Permutation Feature Importance"}))
combined_df = pd.concat(combined_frames, ignore_index=True, sort=False)

st.markdown("---")
st.download_button(
    "Download research evidence CSV", combined_df.to_csv(index=False),
    "research_evidence.csv", "text/csv",
)
