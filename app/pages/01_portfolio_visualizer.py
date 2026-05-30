import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from lib.theme import inject_css, PNG_CONFIG
from lib.data import (
    get_positions,
    get_returns,
    get_factor_betas,
    get_cumulative_returns,
    get_rolling_vol,
    get_trade_log,
)

st.set_page_config(page_title="Portfolio Visualizer · AIC", layout="wide")
inject_css()
st.title("Portfolio Visualizer")

# ── Sidebar filters ──────────────────────────────────────────────────────────
positions = get_positions()
all_tickers = sorted(positions["ticker"].tolist())

with st.sidebar:
    st.header("Filters")
    selected_tickers = st.multiselect(
        "Tickers", all_tickers, default=all_tickers, key="ticker_filter"
    )
    if not selected_tickers:
        st.warning("Select at least one ticker.")
        st.stop()

# ── Current weights ──────────────────────────────────────────────────────────
st.subheader("Current Weights")

pos_filtered = positions[positions["ticker"].isin(selected_tickers)]
fig_pie = px.pie(
    pos_filtered,
    names="ticker",
    values="weight",
    title="Portfolio Weights",
    template="capital",
    hole=0.3,
)
fig_pie.update_traces(textposition="inside", textinfo="percent+label")
st.plotly_chart(fig_pie, use_container_width=True, config=PNG_CONFIG)

# ── Cumulative returns ───────────────────────────────────────────────────────
st.subheader("Cumulative Returns")

cum_returns = get_cumulative_returns()
cum_filtered = cum_returns[cum_returns["ticker"].isin(selected_tickers)]

fig_line = px.line(
    cum_filtered,
    x="date",
    y="cumulative_return",
    color="ticker",
    title="Cumulative Returns",
    labels={"cumulative_return": "Return", "date": "Date"},
    template="capital",
)
fig_line.update_layout(yaxis_tickformat=".0%")
fig_line.add_hline(y=0, line_dash="dot", line_color="#BFDBFE")
st.plotly_chart(fig_line, use_container_width=True, config=PNG_CONFIG)

# ── Rolling volatility ───────────────────────────────────────────────────────
st.subheader("21-Day Rolling Volatility (annualised)")

roll_vol = get_rolling_vol()
roll_filtered = roll_vol[roll_vol["ticker"].isin(selected_tickers)]

fig_vol = px.line(
    roll_filtered,
    x="date",
    y="rolling_vol_21d",
    color="ticker",
    title="Rolling Volatility",
    labels={"rolling_vol_21d": "Ann. Vol.", "date": "Date"},
    template="capital",
)
fig_vol.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig_vol, use_container_width=True, config=PNG_CONFIG)

# ── Factor exposures ─────────────────────────────────────────────────────────
st.subheader("Factor Exposures")

betas = get_factor_betas()
betas_filtered = betas[betas["ticker"].isin(selected_tickers)]
factor_cols = ["market_beta", "value_beta", "momentum_beta", "quality_beta"]
heat_data = betas_filtered.set_index("ticker")[factor_cols].T

fig_heat = go.Figure(
    go.Heatmap(
        z=heat_data.values,
        x=heat_data.columns.tolist(),
        y=heat_data.index.tolist(),
        colorscale=[[0, "#EF4444"], [0.5, "#F8FAFC"], [1, "#2563EB"]],
        zmid=0,
        text=heat_data.values.round(2),
        texttemplate="%{text}",
        showscale=True,
    )
)
fig_heat.update_layout(title="Factor Betas", template="capital", height=280)
st.plotly_chart(fig_heat, use_container_width=True, config=PNG_CONFIG)

# ── Holdings table ───────────────────────────────────────────────────────────
st.subheader("Holdings")

display = pos_filtered.copy()
display["weight"] = display["weight"].map("{:.2%}".format)
display["price"] = display["price"].map("${:,.2f}".format)
display["market_value"] = display["market_value"].map("${:,.0f}".format)
display["shares"] = display["shares"].map("{:,.0f}".format)
st.dataframe(display, use_container_width=True, hide_index=True)

# ── Trade log ────────────────────────────────────────────────────────────────
st.subheader("Trade Log")

trades = get_trade_log()
trades_filtered = trades[trades["ticker"].isin(selected_tickers)]

trades_display = trades_filtered.copy()
trades_display["price"] = trades_display["price"].map("${:,.2f}".format)
trades_display["value"] = trades_display["value"].map("${:,.0f}".format)
trades_display["shares"] = trades_display["shares"].map("{:,.0f}".format)

st.dataframe(
    trades_display,
    use_container_width=True,
    hide_index=True,
    column_order=["date", "ticker", "action", "shares", "price", "value"],
)
