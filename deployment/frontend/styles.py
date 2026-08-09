"""Shared custom styling -- injected at the top of every page. Without
this, Streamlit renders with its plain default look (generic buttons,
default metric cards, default spacing) regardless of the dark theme set
in .streamlit/config.toml. This is what actually closes the gap between
"looks like default Streamlit" and "looks like a professional fintech
platform" -- the theme alone isn't enough.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
/* Tighter, more deliberate spacing than Streamlit's defaults */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Metric cards: bordered, flat, no default Streamlit box-shadow */
div[data-testid="stMetric"] {
    background-color: #171B22;
    border: 1px solid #262B34;
    border-radius: 8px;
    padding: 16px 18px;
}
div[data-testid="stMetricLabel"] {
    font-size: 12px;
    color: #8A8F98;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
div[data-testid="stMetricValue"] {
    font-size: 26px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}

/* Buttons: flatter, more deliberate than Streamlit default */
.stButton > button {
    border-radius: 6px;
    border: 1px solid #262B34;
    font-weight: 500;
    transition: border-color 0.15s ease;
}
.stButton > button:hover {
    border-color: #1D9E75;
    color: #1D9E75;
}
.stButton > button[kind="primary"] {
    background-color: #1D9E75;
    border: none;
}

/* Tables: tighter row height, tabular numbers for real data alignment */
div[data-testid="stDataFrame"] {
    font-variant-numeric: tabular-nums;
}

/* Headings: tighter tracking, no default Streamlit anchor-link icons cluttering the view */
h1, h2, h3 {
    letter-spacing: -0.01em;
}
[data-testid="stHeaderActionElements"] {
    display: none;
}

/* Sidebar: slightly darker than main content, clear separation */
section[data-testid="stSidebar"] {
    background-color: #0B0D11;
    border-right: 1px solid #262B34;
}

/* Expanders: flatter, bordered instead of Streamlit's default shadow */
div[data-testid="stExpander"] {
    border: 1px solid #262B34;
    border-radius: 8px;
}
</style>
"""


def inject():
    """Call once at the top of every page, right after st.set_page_config()."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
