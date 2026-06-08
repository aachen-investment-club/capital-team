import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st

from lib.theme import inject_css, FAVICON

st.set_page_config(
    page_title="Portfolio Dashboard · AIC",
    page_icon=FAVICON,
    layout="wide",
)
inject_css()

st.title("AIC Capital Dashboard")
st.caption("Aachen Investment Club e.V. · Capital Team · read-only · data refreshes daily")
st.divider()

PAGES = [
    ("01", "Performance",        "Returns, attribution, weightings, and trade log."),
    ("02", "Equities",           "Factor screening across all holdings."),
    ("03", "Weighting",          "Position and basket weights over time."),
    ("04", "Volatility",         "Realised volatility per position and portfolio."),
    ("05", "Barra",              "Multi-factor risk decomposition."),
    ("06", "Correlation",        "Rolling and partial correlations across key market pairs."),
    ("07", "Credit & Liquidity", "HY spread stress and market liquidity monitor."),
    ("08", "Trend Detection",    "Kalman and SMA trend filters with GJR-GARCH adaptive noise."),
    ("09", "Structural Break",   "Regime shift detection via CUSUM and BOCPD."),
]

st.divider()
st.markdown("**Work in progress**")
WIP = [
    "Structural Break — CUSUM & BOCPD regime detection",
    "Regime Detection — hidden Markov model overlay",
    "DCC-GARCH — dynamic correlations for intra-portfolio positions",
]
for item in WIP:
    st.markdown(
        f"<span style='color:#F59E0B'>⚙</span>"
        f"<span style='color:#94A3B8'>  {item}</span>",
        unsafe_allow_html=True,
    )
st.divider()

for num, name, desc in PAGES:
    col_name, col_desc = st.columns([1, 4])
    with col_name:
        st.markdown(f"**{name}**")
    with col_desc:
        st.markdown(f"<span style='color:#94A3B8'>{desc}</span>", unsafe_allow_html=True)
    st.markdown("<hr style='margin:6px 0;border-color:#1E293B'>", unsafe_allow_html=True)
