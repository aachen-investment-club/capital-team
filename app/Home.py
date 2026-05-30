import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import streamlit as st

from lib.theme import inject_css
from lib.data import get_positions, get_returns

st.set_page_config(
    page_title="Portfolio Analytics · AIC",
    page_icon="📊",
    layout="wide",
)
inject_css()

positions = get_positions()
returns = get_returns()

total_nav = positions["market_value"].sum()
n_holdings = len(positions)
latest_date = returns["date"].max()

cum = (
    returns.groupby("ticker")["daily_return"]
    .apply(lambda s: (1 + s).prod() - 1)
    .reset_index(name="total_return")
)
weighted = cum.merge(positions[["ticker", "weight"]], on="ticker")
portfolio_return = (weighted["total_return"] * weighted["weight"]).sum()
best = cum.loc[cum["total_return"].idxmax()]
worst = cum.loc[cum["total_return"].idxmin()]

st.title("Portfolio Analytics")
st.caption("Aachen Investment Club · Capital Team · read-only · data refreshes hourly")
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net Asset Value", f"${total_nav:,.0f}")
c2.metric("Holdings", n_holdings)
c3.metric("Portfolio Return", f"{portfolio_return:+.1%}")
c4.metric("Data Through", str(latest_date))

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Top Performer")
    st.metric(best["ticker"], f"{best['total_return']:+.1%}")
with col_b:
    st.subheader("Worst Performer")
    st.metric(worst["ticker"], f"{worst['total_return']:+.1%}")

st.divider()
st.subheader("Pages")
st.markdown("""
| Page | What it shows |
|---|---|
| **Portfolio Visualizer** | Weights · cumulative returns · rolling vol · factor betas · trade log |

**Extending:** add `app/pages/NN_name.py`, import from `lib.data` and `lib.metrics`, chart with `lib.theme`.
""")
