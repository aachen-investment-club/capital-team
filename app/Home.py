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

st.title("Portfolio Dashboard")
st.caption("Aachen Investment Club e.V. · Capital Team · read-only · data refreshes daily")
st.divider()

st.markdown("""
| Page | What it shows |
|---|---|
| **Performance** | Portfolio vs benchmarks · single-position returns · daily weightings by basket · trade log |

| **Risk KPIs** | Portfolio risk · drawdown · rolling volatility · rolling Sharpe |
| **What-if Simulator** | Portfolio + 1 new asset in question · KPI comparison · graph comparison |
""")
