import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date

from lib.theme import inject_css, FAVICON
from lib.data import (
    get_security_master,
    get_eod_prices,
    get_portfolio_and_benchmarks,
    _eod_data_version,
)
from lib.weighting import build_basket_stats, get_weights, efficient_frontier, portfolio_stats

st.set_page_config(page_title="Weighting · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Portfolio Optimiser")

_TODAY    = date.today() - pd.Timedelta(days=1)
_DATE_MIN = date(2024, 1, 1)


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Optimiser Settings")

    backtest_from = st.date_input(
        "Backtest data from",
        value=date(2025, 1, 1),
        min_value=_DATE_MIN,
        max_value=_TODAY,
        help="Use EOD price history starting from this date to estimate returns, vol, and beta.",
    )

    risk_aversion = st.slider(
        "Risk aversion (λ)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Higher λ penalises volatility more and pushes weights toward lower-risk assets.",
    )

    max_beta = st.slider(
        "Max portfolio beta",
        min_value=0.3,
        max_value=2.0,
        value=1.0,
        step=0.05,
        help="Upper bound on the weighted-average beta of the optimised portfolio vs SPX.",
    )

    run_btn = st.button("Run Optimiser", type="primary", use_container_width=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_all_eod(cache_version: str) -> dict:
    """Load EOD prices for every active security, keyed by security_id."""
    sm = get_security_master()
    return {row["security_id"]: get_eod_prices(row["security_id"], cache_version)
            for _, row in sm.iterrows()}


@st.cache_data(ttl=3600)
def _spx_returns() -> pd.Series:
    """Daily returns for SPX from the portfolio benchmarks table."""
    df = get_portfolio_and_benchmarks()
    spx = df[df["ticker"] == "SPX"][["date", "daily_return"]].copy()
    spx = spx.dropna(subset=["daily_return"]).set_index("date")["daily_return"]
    return spx


@st.cache_data(ttl=3600)
def _portfolio_returns() -> pd.Series:
    """Daily returns for the live portfolio."""
    df = get_portfolio_and_benchmarks()
    port = df[df["ticker"] == "PORTFOLIO"][["date", "daily_return"]].copy()
    port = port.dropna(subset=["daily_return"]).set_index("date")["daily_return"]
    return port


def _current_portfolio_stats(backtest_from) -> dict | None:
    """Annualised return and vol for the live portfolio from backtest_from onwards."""
    rets = _portfolio_returns()
    rets = rets[rets.index >= pd.Timestamp(backtest_from)]
    if len(rets) < 5:
        return None
    return {
        "Ann Return": float(rets.mean() * 252),
        "Ann Vol":    float(rets.std() * np.sqrt(252)),
    }


sm = get_security_master()
sm = sm[sm["asset_type"] != "INDEX"].reset_index(drop=True)  # exclude benchmark indices

if sm.empty:
    st.info("No active securities found in security_master.csv.")
    st.stop()

eod_cache    = _load_all_eod(_eod_data_version())
spx_rets     = _spx_returns()
current_port = _current_portfolio_stats(backtest_from)

basket_stats = build_basket_stats(sm, eod_cache, spx_rets, backtest_from)

if not basket_stats:
    st.warning(
        f"No securities have sufficient price history from {backtest_from}. "
        "Try an earlier backtest start date or ingest more EOD data."
    )
    st.stop()


# ── Efficient frontier (always shown, recomputes when settings change) ────────

st.subheader("Efficient Frontier")

with st.spinner("Computing efficient frontier…"):
    try:
        f_vols, f_rets = efficient_frontier(basket_stats)
        frontier_ok = len(f_vols) > 1
    except Exception as e:
        frontier_ok = False
        st.error(f"Frontier computation failed: {e}")
        f_vols, f_rets = [], []

fig = go.Figure()

if frontier_ok:
    fig.add_trace(go.Scatter(
        x=f_vols, y=f_rets,
        mode="lines",
        name="Efficient Frontier",
        line=dict(color="#2563EB", width=2, dash="dash"),
    ))

# Current portfolio anchor
if current_port is not None:
    fig.add_trace(go.Scatter(
        x=[current_port["Ann Vol"]],
        y=[current_port["Ann Return"]],
        mode="markers",
        name="Current Portfolio",
        marker=dict(symbol="diamond", size=14, color="#F59E0B"),
        hovertemplate=(
            "<b>Current Portfolio</b><br>"
            "Ann Return: %{y:.1%}<br>"
            "Ann Vol: %{x:.1%}<extra></extra>"
        ),
    ))

# Individual securities
for b in basket_stats:
    fig.add_trace(go.Scatter(
        x=[b["Ann Vol"]], y=[b["Ann Return"]],
        mode="markers+text",
        name=b["Symbol"],
        text=[b["Symbol"]],
        textposition="top center",
        marker=dict(size=9, opacity=0.75),
        hovertemplate=(
            f"<b>{b['Symbol']}</b><br>"
            f"Ann Return: %{{y:.1%}}<br>"
            f"Ann Vol: %{{x:.1%}}<br>"
            f"Beta: {b['Beta']:.2f}<extra></extra>"
        ),
    ))

fig.update_layout(
    template="capital",
    xaxis_title="Annualised Volatility",
    yaxis_title="Annualised Return",
    xaxis_tickformat=".0%",
    yaxis_tickformat=".0%",
    legend=dict(orientation="v"),
    height=520,
    margin=dict(l=60, r=20, t=40, b=60),
)
fig.update_xaxes(rangemode="tozero")

_frontier_placeholder = st.empty()
_frontier_placeholder.plotly_chart(fig, width="stretch", key="frontier_base")


# ── Optimised weights (triggered by button or rerun) ─────────────────────────

if "opt_result" not in st.session_state:
    st.session_state["opt_result"] = None

if run_btn:
    with st.spinner("Optimising…"):
        try:
            weights = get_weights(basket_stats, max_beta=max_beta, risk_aversion=risk_aversion)
            stats   = portfolio_stats(basket_stats, weights)
            st.session_state["opt_result"] = {
                "weights":  weights,
                "stats":    stats,
                "symbols":  [b["Symbol"] for b in basket_stats],
                "names":    [b["Name"]   for b in basket_stats],
                "ind_vols": [b["Ann Vol"]    for b in basket_stats],
                "ind_rets": [b["Ann Return"] for b in basket_stats],
                "betas":    [b["Beta"]       for b in basket_stats],
            }
        except Exception as e:
            st.error(str(e))
            st.session_state["opt_result"] = None

result = st.session_state["opt_result"]

if result is not None:
    # Re-draw frontier chart with optimised portfolio overlaid
    fig2 = go.Figure(fig)
    fig2.add_trace(go.Scatter(
        x=[result["stats"]["Ann Vol"]],
        y=[result["stats"]["Ann Return"]],
        mode="markers",
        name="Optimised Portfolio",
        marker=dict(symbol="star", size=18, color="#EF4444"),
        hovertemplate=(
            "<b>Optimised Portfolio</b><br>"
            f"Ann Return: %{{y:.1%}}<br>"
            f"Ann Vol: %{{x:.1%}}<br>"
            f"Beta: {result['stats']['Beta']:.2f}<extra></extra>"
        ),
    ))
    _frontier_placeholder.plotly_chart(fig2, width="stretch", key="frontier_opt")

    # ── Metrics row ──────────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Ann Return",  f"{result['stats']['Ann Return']:+.2%}")
    m2.metric("Ann Vol",     f"{result['stats']['Ann Vol']:.2%}")
    m3.metric("Portfolio β", f"{result['stats']['Beta']:.2f}")

    # ── Weights table ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Optimal Weights")

    df_w = pd.DataFrame({
        "Symbol":      result["symbols"],
        "Name":        result["names"],
        "Weight":      result["weights"],
        "Ann Return":  result["ind_rets"],
        "Ann Vol":     result["ind_vols"],
        "Beta":        result["betas"],
    })
    df_w = df_w.sort_values("Weight", ascending=False).reset_index(drop=True)

    # Bar chart of weights
    bar_fig = go.Figure(go.Bar(
        x=df_w["Symbol"],
        y=df_w["Weight"],
        marker_color="#2563EB",
        text=df_w["Weight"].map("{:.1%}".format),
        textposition="outside",
    ))
    bar_fig.update_layout(
        template="capital",
        yaxis_tickformat=".0%",
        yaxis_title="Weight",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(bar_fig, width="stretch")

    # Formatted table
    disp = df_w.copy()
    disp["Weight"]     = disp["Weight"].map("{:.1%}".format)
    disp["Ann Return"] = disp["Ann Return"].map("{:+.1%}".format)
    disp["Ann Vol"]    = disp["Ann Vol"].map("{:.1%}".format)
    disp["Beta"]       = disp["Beta"].map("{:.2f}".format)
    st.dataframe(
        disp.rename(columns={
            "Symbol": "Symbol", "Name": "Name", "Weight": "Weight",
            "Ann Return": "Ann Return", "Ann Vol": "Ann Vol", "Beta": "Beta",
        }),
        width="stretch",
        hide_index=True,
    )

else:
    st.info(
        "Configure the settings in the sidebar and click **Run Optimiser** "
        "to generate optimal weights."
    )
