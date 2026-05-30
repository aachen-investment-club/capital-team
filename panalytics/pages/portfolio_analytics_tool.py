import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>""",
    unsafe_allow_html=True
)


# ===== DATA LOADING =====

@st.cache_data
def load_data():
    trades = pd.read_csv("data/transactions.csv", parse_dates=["date"])
    trades["ticker"] = trades["ticker"].str.strip()
    close_prices = pd.read_csv("data/timeseries.csv", index_col=0, parse_dates=True)
    close_prices.index.name = "date"
    return trades, close_prices

trades, close_prices = load_data()


# ===== BENCHMARKS =====

BENCHMARK_COL = ".MIWO00000PUS"
# Drop US market holidays (all stock prices NaN, but MSCI World may still have a value)
stock_prices = close_prices.drop(columns=[BENCHMARK_COL]).dropna(how="all")
# Align benchmark to the same trading days as the stocks
benchmark_prices = close_prices[BENCHMARK_COL].reindex(stock_prices.index)

# Snap trade dates that fall on weekends/holidays to the next available trading day
valid_dates = stock_prices.index
def snap_to_trading_day(d: pd.Timestamp) -> pd.Timestamp:
    future = valid_dates[valid_dates >= d]
    return future[0] if len(future) > 0 else valid_dates[-1]

trades["date"] = trades["date"].apply(snap_to_trading_day)

prices_long = stock_prices.stack().reset_index()
prices_long.columns = ["date", "ticker", "close"]


# ===== TRADES =====

trades = trades.merge(prices_long, on=["date", "ticker"])
trades["cashflow"] = trades["units"] * trades["close"]
total_invested = trades[trades["units"] > 0]["cashflow"].sum()
# st.write(trades)


# ===== CURRENT HOLDINGS =====

holdings = trades.groupby("ticker")["units"].sum().reset_index()
holdings = holdings[holdings["units"] > 0].copy()
holdings["current_price"] = holdings["ticker"].map(stock_prices.iloc[-1])
holdings["current_value"] = holdings["units"] * holdings["current_price"]
# st.write(holdings)

cost_basis = (
    trades[trades["units"] > 0]
    .groupby("ticker")
    .apply(lambda x: (x["units"] * x["close"]).sum(), include_groups=False)
    .reset_index(name="cost_basis")
)
holdings = holdings.merge(cost_basis, on="ticker")
holdings["pnl"] = holdings["current_value"] - holdings["cost_basis"]
holdings["pnl_pct"] = holdings["pnl"] / holdings["cost_basis"]
holdings["weight"] = holdings["current_value"] / holdings["current_value"].sum()

current_value = holdings["current_value"].sum()
simple_return = current_value / total_invested - 1


# ===== PORTFOLIO VALUE OVER TIME =====

first_trade_date = trades["date"].min()

trade_shares = (
    trades
    .pivot_table(index="date", columns="ticker", values="units", aggfunc="sum")
    .reindex(stock_prices.index)
    .fillna(0)
    .cumsum()
)

trade_cashflows = (
    trades
    .groupby("date")["cashflow"]
    .sum()
    .reindex(stock_prices.index)
    .fillna(0)
)

portfolio_value = (stock_prices * trade_shares).sum(axis=1)
portfolio_value = portfolio_value[portfolio_value.index >= first_trade_date]
trade_cashflows = trade_cashflows[trade_cashflows.index >= first_trade_date]


# ===== TWR & RISK METRICS =====

def twr_returns(pv: pd.Series, cf: pd.Series) -> pd.Series:
    rets = []
    for i in range(1, len(pv)):
        prev = pv.iloc[i - 1]
        r = (pv.iloc[i] - cf.iloc[i]) / prev - 1 if prev != 0 else 0.0
        rets.append(r)
    return pd.Series(rets, index=pv.index[1:])

def xirr(cashflows: pd.Series) -> float:
    """Newton-Raphson solver for XIRR (annualised money-weighted return)."""
    dates = cashflows.index
    values = cashflows.values
    days = np.array([(d - dates[0]).days for d in dates], dtype=float)

    def npv(r: float) -> float:
        return np.sum(values / (1 + r) ** (days / 365.0))

    r = 0.1
    for _ in range(200):
        f = npv(r)
        df = np.sum(-days / 365.0 * values / (1 + r) ** (days / 365.0 + 1))
        if abs(df) < 1e-12:
            break
        r_new = r - f / df
        r_new = max(r_new, -0.999)
        if abs(r_new - r) < 1e-8:
            r = r_new
            break
        r = r_new
    return r

port_returns = twr_returns(portfolio_value, trade_cashflows)

rf = 0.035
ann_vol = port_returns.std() * np.sqrt(252)
ann_ret = port_returns.mean() * 252
sharpe = (ann_ret - rf) / ann_vol
cum = (1 + port_returns).cumprod()
mdd = ((cum - cum.cummax()) / cum.cummax()).min()
twr_total = cum.iloc[-1] - 1
cagr = (cum.iloc[-1] ** (252 / len(cum))) - 1
downside_vol = port_returns[port_returns < 0].std() * np.sqrt(252)
sortino = (ann_ret - rf) / downside_vol
calmar = cagr / abs(mdd)

# MWR (XIRR): negative cashflow = money invested, positive = proceeds + current value
xirr_cf = -trades.groupby("date")["cashflow"].sum()
xirr_cf[stock_prices.index[-1]] = current_value
xirr_cf = xirr_cf.sort_index()
mwr = xirr(xirr_cf)


# ===== BENCHMARK =====

bm = benchmark_prices[benchmark_prices.index >= first_trade_date]
bm_returns = bm.pct_change().dropna()
bm_cum = (1 + bm_returns).cumprod()

common_idx = cum.index.intersection(bm_cum.index)
cum_port = cum.reindex(common_idx)
cum_bm = bm_cum.reindex(common_idx)


# ===== LAYOUT =====

st.title("Portfolio Analytics")
st.caption(f"Data as of {stock_prices.index[-1].date()}")

# --- KPI Row 1 ---
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Total Invested", f"€{total_invested:,.0f}")
with c2:
    delta_eur = current_value - total_invested
    st.metric("Current Value", f"€{current_value:,.0f}", delta=f"€{delta_eur:,.0f}")
with c3:
    st.metric("Simple Return", f"{simple_return:.2%}")
with c4:
    st.metric("TWR", f"{twr_total:.2%}")
with c5:
    st.metric("MWR (XIRR)", f"{mwr:.2%}")
with c6:
    st.metric("CAGR", f"{cagr:.2%}")

# --- KPI Row 2 ---
c7, c8, c9, c10, c11 = st.columns(5)
with c7:
    st.metric("Volatility (ann.)", f"{ann_vol:.2%}")
with c8:
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
with c9:
    st.metric("Sortino Ratio", f"{sortino:.2f}")
with c10:
    st.metric("Max Drawdown", f"{mdd:.2%}")
with c11:
    st.metric("Calmar Ratio", f"{calmar:.2f}")

st.markdown("---")

# --- Portfolio Value & Allocation ---
col_chart, col_pie = st.columns([2, 1])

with col_chart:
    st.subheader("Portfolio Value Over Time")
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(
        x=portfolio_value.index, y=portfolio_value.values,
        mode="lines", name="Portfolio",
        line=dict(color="#1f77b4", width=2),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.08)",
    ))
    fig_val.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Value (€)", tickformat=",.0f"),
        xaxis_title=None, showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_val, use_container_width=True)

with col_pie:
    st.subheader("Allocation")
    fig_pie = px.pie(
        holdings, values="current_value", names="ticker",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Cumulative Returns vs Benchmark ---
st.subheader("Cumulative Returns vs MSCI World")
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=cum_port.index, y=(cum_port - 1) * 100,
    mode="lines", name="Portfolio",
    line=dict(color="#1f77b4", width=2),
))
fig_cum.add_trace(go.Scatter(
    x=cum_bm.index, y=(cum_bm - 1) * 100,
    mode="lines", name="MSCI World",
    line=dict(color="#ff7f0e", width=2, dash="dash"),
))
fig_cum.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
fig_cum.update_layout(
    height=300, margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="Return (%)",
    xaxis_title=None,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_cum, use_container_width=True)

# --- Drawdown ---
st.subheader("Drawdown")
drawdown = (cum - cum.cummax()) / cum.cummax() * 100
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown.values,
    mode="lines",
    line=dict(color="#d62728", width=1.5),
    fill="tozeroy", fillcolor="rgba(214,39,40,0.12)",
    showlegend=False,
))
fig_dd.update_layout(
    height=200, margin=dict(l=0, r=0, t=10, b=0),
    yaxis_title="Drawdown (%)",
    xaxis_title=None,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("---")

# --- Holdings & Transactions ---
col_h, col_t = st.columns([3, 2])

with col_h:
    st.subheader("Current Holdings")
    display_holdings = holdings[["ticker", "units", "current_price", "current_value", "pnl", "pnl_pct", "weight"]].copy()
    display_holdings.columns = ["Ticker", "Units", "Price (€)", "Value (€)", "P&L (€)", "P&L %", "Weight"]
    st.dataframe(
        display_holdings.style.format({
            "Price (€)": "€{:.2f}",
            "Value (€)": "€{:,.2f}",
            "P&L (€)": "€{:,.2f}",
            "P&L %": "{:.2%}",
            "Weight": "{:.1%}",
        }).map(
            lambda v: "color: #2ecc71" if v > 0 else ("color: #e74c3c" if v < 0 else ""),
            subset=["P&L (€)", "P&L %"],
        ),
        use_container_width=True,
        hide_index=True,
    )

with col_t:
    st.subheader("Transactions")
    display_trades = trades[["date", "ticker", "units", "close", "cashflow"]].copy()
    display_trades.columns = ["Date", "Ticker", "Units", "Price (€)", "Cashflow (€)"]
    display_trades["Date"] = display_trades["Date"].dt.strftime("%Y-%m-%d")
    display_trades = display_trades.sort_values("Date", ascending=False)
    st.dataframe(
        display_trades.style.format({
            "Price (€)": "€{:.2f}",
            "Cashflow (€)": "€{:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
