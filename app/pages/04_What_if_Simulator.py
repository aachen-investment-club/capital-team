import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.theme import inject_css, FAVICON, NAVY
from lib.data import get_portfolio_and_benchmarks
from lib.whatif_data import get_ticker_returns

st.set_page_config(page_title="What-If Simulator · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("What-If Simulator")
st.caption("How would adding a new position change the portfolio's risk profile?")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_kpis(r: pd.Series, rf_daily: float, bm_r: pd.Series) -> dict:
    if len(r) < 5:
        return {}

    aligned = pd.DataFrame({"r": r, "bm": bm_r}).dropna()
    r    = aligned["r"]
    bm_r = aligned["bm"]

    vol     = float(r.std() * np.sqrt(252))
    ann_ret = float((1 + r).prod() ** (252.0 / len(r)) - 1)
    excess  = r - rf_daily
    sharpe  = float(excess.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")

    downside = r[r < rf_daily] - rf_daily
    ds_std   = float(downside.std()) if len(downside) > 1 else float("nan")
    sortino  = float(excess.mean() / ds_std * np.sqrt(252)) if (ds_std and ds_std > 0) else float("nan")

    cum    = (1 + r).cumprod()
    dd     = cum / cum.cummax() - 1
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")

    cov_m = np.cov(r.values, bm_r.values)
    beta  = float(cov_m[0, 1] / cov_m[1, 1]) if cov_m[1, 1] > 0 else float("nan")
    alpha = (
        float((excess.mean() - beta * (bm_r - rf_daily).mean()) * 252)
        if not np.isnan(beta) else float("nan")
    )

    return {
        "vol": vol, "ann_ret": ann_ret, "sharpe": sharpe,
        "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
        "beta": beta, "alpha": alpha,
        "r": r, "dd": dd, "cum": cum, "n": len(r),
    }


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}" if not np.isnan(v) else "–"

def _fmt_x(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if not np.isnan(v) else "–"

def _delta(new: float, old: float, higher_is_better: bool = True) -> str:
    if np.isnan(new) or np.isnan(old):
        return ""
    diff = new - old
    sign = "+" if diff > 0 else ""
    better = (diff > 0) == higher_is_better
    arrow  = "▲" if diff > 0 else "▼"
    color  = "green" if better else "red"
    return f'<span style="color:{color}">{arrow} {sign}{diff:.2%}</span>'


# ── Controls ───────────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    ticker_input = st.text_input("Ticker (yfinance)", placeholder="e.g. NVDA, BTC-USD, GLD", key="wi_ticker")
with c2:
    new_weight_pct = st.slider("New position weight (% of portfolio)", 1, 50, 10, 1, key="wi_weight")
with c3:
    rf_pct  = st.slider("Risk-Free Rate (% p.a.)", 0.0, 10.0, 3.5, 0.1, key="wi_rf")

ticker     = ticker_input.strip().upper()
new_weight = new_weight_pct / 100.0
rf_daily   = rf_pct / 100.0 / 252.0

if not ticker:
    st.info("Enter a ticker above to start the simulation.")
    st.stop()


# ── Load data ──────────────────────────────────────────────────────────────────

with st.spinner(f"Fetching {ticker} from yfinance…"):
    try:
        ticker_r = get_ticker_returns(ticker)
    except ValueError as e:
        st.error(str(e))
        st.stop()

port_df  = get_portfolio_and_benchmarks()
port_r   = port_df[port_df["ticker"] == "PORTFOLIO"].set_index("date")["daily_return"].sort_index()
port_r.index = pd.to_datetime(port_r.index).tz_localize(None)

# Benchmark (S&P 500) for beta/alpha
bm_r = port_df[port_df["ticker"] == "SPX"].set_index("date")["daily_return"].sort_index()
bm_r.index = pd.to_datetime(bm_r.index).tz_localize(None)

# Align all three on the portfolio period (portfolio is the shortest series)
aligned = pd.DataFrame({"port": port_r, "new": ticker_r, "bm": bm_r}).dropna()

n_overlap = len(aligned)
if n_overlap < 5:
    st.error(
        f"Only {n_overlap} overlapping trading days between the portfolio and {ticker}. "
        "Cannot compute meaningful KPIs."
    )
    st.stop()

if n_overlap < 30:
    st.warning(
        f"Only {n_overlap} overlapping trading days — annualised figures are statistically noisy. "
        "Results will improve as the portfolio history grows."
    )

port_r_aligned   = aligned["port"]
ticker_r_aligned = aligned["new"]
bm_r_aligned     = aligned["bm"]

# What-if: scale existing portfolio by (1-w), add new position at w
whatif_r = (1 - new_weight) * port_r_aligned + new_weight * ticker_r_aligned


# ── Compute KPIs ───────────────────────────────────────────────────────────────

kpis_before = _compute_kpis(port_r_aligned,   rf_daily, bm_r_aligned)
kpis_after  = _compute_kpis(whatif_r,          rf_daily, bm_r_aligned)

date_lo = aligned.index.min().strftime("%d %b %Y")
date_hi = aligned.index.max().strftime("%d %b %Y")
st.caption(
    f"Simulation over {n_overlap} overlapping trading days · {date_lo} → {date_hi} · "
    f"New weight: {new_weight_pct}% · rf = {rf_pct:.1f}%"
)


# ── KPI comparison table ───────────────────────────────────────────────────────

st.subheader("KPI Comparison")

metrics = [
    ("Volatility (ann.)",  "vol",     _fmt_pct, False),
    ("Ann. Return",        "ann_ret", _fmt_pct, True),
    ("Sharpe Ratio",       "sharpe",  lambda v: _fmt_x(v, 2), True),
    ("Sortino Ratio",      "sortino", lambda v: _fmt_x(v, 2), True),
    ("Max Drawdown",       "max_dd",  _fmt_pct, False),
    ("Calmar Ratio",       "calmar",  lambda v: _fmt_x(v, 2), True),
    ("Beta (vs S&P 500)",  "beta",    lambda v: _fmt_x(v, 2), False),
    ("Alpha (ann.)",       "alpha",   _fmt_pct, True),
]

cols = st.columns([2, 1, 1, 1])
cols[0].markdown("**Metric**")
cols[1].markdown("**Current**")
cols[2].markdown(f"**+ {ticker} ({new_weight_pct}%)**")
cols[3].markdown("**Change**")
st.divider()

for label, key, fmt, higher_is_better in metrics:
    v_before = kpis_before.get(key, float("nan"))
    v_after  = kpis_after.get(key,  float("nan"))
    cols = st.columns([2, 1, 1, 1])
    cols[0].write(label)
    cols[1].write(fmt(v_before))
    cols[2].write(fmt(v_after))
    cols[3].markdown(_delta(v_after, v_before, higher_is_better), unsafe_allow_html=True)

st.divider()


# ── Charts ─────────────────────────────────────────────────────────────────────

tab_dd, tab_cum, tab_ticker, tab_weights = st.tabs(["Drawdown", "Cumulative Return", f"{ticker} Price", "Weight Change"])

with tab_dd:
    dd_before = kpis_before["dd"]
    dd_after  = kpis_after["dd"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_before.index, y=dd_before.values,
        name="Current portfolio",
        line=dict(color=NAVY, width=1.5),
        hovertemplate="%{x|%d %b %Y}: %{y:.1%}<extra>Current</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dd_after.index, y=dd_after.values,
        name=f"+ {ticker} ({new_weight_pct}%)",
        line=dict(color="#EF4444", width=1.5, dash="dash"),
        hovertemplate="%{x|%d %b %Y}: %{y:.1%}<extra>What-If</extra>",
    ))
    fig.update_layout(
        title="Drawdown: Current vs What-If",
        yaxis_tickformat=".0%",
        yaxis_title="Drawdown",
        xaxis_title="",
        template="capital",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, width="stretch")

with tab_cum:
    cum_before = kpis_before["cum"]
    cum_after  = kpis_after["cum"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_before.index, y=cum_before.values,
        name="Current portfolio",
        line=dict(color=NAVY, width=2),
        hovertemplate="%{x|%d %b %Y}: %{y:.4f}<extra>Current</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=cum_after.index, y=cum_after.values,
        name=f"+ {ticker} ({new_weight_pct}%)",
        line=dict(color="#B8962E", width=2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}: %{y:.4f}<extra>What-If</extra>",
    ))
    fig.update_layout(
        title="Cumulative Return: Current vs What-If",
        yaxis_title="Growth of 1",
        xaxis_title="",
        template="capital",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, width="stretch")

with tab_ticker:
    ticker_cum = (1 + ticker_r_aligned).cumprod()
    fig = go.Figure(go.Scatter(
        x=ticker_cum.index, y=ticker_cum.values,
        line=dict(color="#10B981", width=2),
        hovertemplate="%{x|%d %b %Y}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{ticker} — Cumulative Return (portfolio overlap period)",
        yaxis_title="Growth of 1",
        xaxis_title="",
        showlegend=False,
        template="capital",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, width="stretch")

with tab_weights:
    from lib.data import get_daily_weightings_history

    dw       = get_daily_weightings_history()
    entry_dt = pd.Timestamp(aligned.index.min())
    snap     = dw[dw["date"] == entry_dt].copy()

    if snap.empty:
        snap = dw[dw["date"] == dw["date"].max()].copy()
        entry_label = dw["date"].max().strftime("%d %b %Y") + " (latest)"
    else:
        entry_label = entry_dt.strftime("%d %b %Y")

    snap = snap[["symbol", "name", "pct_nav"]].copy()
    snap["pct_nav_new"] = snap["pct_nav"] * (1 - new_weight)

    new_row = pd.DataFrame([{
        "symbol":      ticker,
        "name":        ticker,
        "pct_nav":     0.0,
        "pct_nav_new": new_weight * 100,
    }])
    snap = pd.concat([snap, new_row], ignore_index=True)
    snap = snap.sort_values("pct_nav_new", ascending=True)
    labels = snap.apply(
        lambda r: f"{r['symbol']} – {r['name']}" if r["symbol"] != ticker else ticker,
        axis=1,
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=snap["pct_nav"],
        name="Current",
        orientation="h",
        marker_color=NAVY,
        hovertemplate="%{y}: %{x:.1f}%<extra>Current</extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels,
        x=snap["pct_nav_new"],
        name=f"+ {ticker} ({new_weight_pct}%)",
        orientation="h",
        marker_color="#B8962E",
        hovertemplate="%{y}: %{x:.1f}%<extra>What-If</extra>",
    ))
    fig.update_layout(
        title=f"Portfolio Weights on Entry Date ({entry_label})",
        barmode="group",
        xaxis_title="% of NAV",
        yaxis_title="",
        xaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="capital",
        height=max(400, 40 * len(snap)),
    )
    st.plotly_chart(fig, width="stretch")
