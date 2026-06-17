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
    get_daily_weightings_history,
    _eod_data_version,
)
from lib.weighting import (
    build_basket_stats,
    get_weights,
    efficient_frontier,
    portfolio_stats,
    build_price_matrix,
    scheme_nav,
    scheme_nav_fixed,
    scheme_stats,
)

st.set_page_config(page_title="Weighting · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Portfolio Optimiser")
with st.popover("ℹ"):
    st.markdown("""Shows current and historical allocation by position and basket, and runs a quadratic optimiser to suggest better weights given your risk aversion and beta constraints.""")

_TODAY    = date.today() - pd.Timedelta(days=1)
_DATE_MIN = date(2024, 1, 1)


# ── Controls ──────────────────────────────────────────────────────────────────

_c1, _c2, _c3, _c4 = st.columns([1.4, 1.4, 1.4, 0.8])
backtest_from = _c1.date_input(
    "Backtest data from", value=date(2025, 1, 1),
    min_value=_DATE_MIN, max_value=_TODAY,
    help="Use EOD price history starting from this date to estimate returns, vol, and beta.",
)
risk_aversion = _c2.slider(
    "Risk aversion (λ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
    help="Higher λ penalises volatility more and pushes weights toward lower-risk assets.",
)
max_beta = _c3.slider(
    "Max portfolio beta", min_value=0.3, max_value=2.0, value=1.0, step=0.05,
    help="Upper bound on the weighted-average beta of the optimised portfolio vs SPX.",
)
run_btn = _c4.button("Run Optimiser", type="primary", use_container_width=True)


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _load_all_eod(cache_version: str) -> dict:
    sm = get_security_master()
    return {row["security_id"]: get_eod_prices(row["security_id"], cache_version)
            for _, row in sm.iterrows()}


@st.cache_data(ttl=3600)
def _spx_returns() -> pd.Series:
    df  = get_portfolio_and_benchmarks()
    spx = df[df["ticker"] == "SPX"][["date", "daily_return"]].copy()
    return spx.dropna(subset=["daily_return"]).set_index("date")["daily_return"]


@st.cache_data(ttl=3600)
def _portfolio_returns() -> pd.Series:
    df   = get_portfolio_and_benchmarks()
    port = df[df["ticker"] == "PORTFOLIO"][["date", "daily_return"]].copy()
    return port.dropna(subset=["daily_return"]).set_index("date")["daily_return"]


@st.cache_data(ttl=3600)
def _cached_price_matrix(cache_version: str) -> pd.DataFrame:
    sm_all = get_security_master()
    sm_all = sm_all[sm_all["asset_type"] != "INDEX"].reset_index(drop=True)
    return build_price_matrix(sm_all, get_eod_prices, cache_version)


@st.cache_data(ttl=3600)
def _cached_scheme_nav(prices: pd.DataFrame, s: str, freq: str | None,
                        spx: pd.Series | None = None) -> pd.Series:
    return scheme_nav(prices, s, freq, spx)


@st.cache_data(ttl=3600)
def _cached_scheme_nav_fixed(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return scheme_nav_fixed(prices, weights)


# ── Load data ─────────────────────────────────────────────────────────────────

sm = get_security_master()
sm = sm[sm["asset_type"] != "INDEX"].reset_index(drop=True)

if sm.empty:
    st.info("No active securities found in security_master.csv.")
    st.stop()

version      = _eod_data_version()
eod_cache    = _load_all_eod(version)
spx_rets     = _spx_returns()

port_rets = _portfolio_returns()
port_rets = port_rets[port_rets.index >= pd.Timestamp(backtest_from)]
current_port = (
    {"Ann Return": float(port_rets.mean() * 252), "Ann Vol": float(port_rets.std() * np.sqrt(252))}
    if len(port_rets) >= 5 else None
)

basket_stats = build_basket_stats(sm, eod_cache, spx_rets, backtest_from)

if not basket_stats:
    st.warning(
        f"No securities have sufficient price history from {backtest_from}. "
        "Try an earlier backtest start date or ingest more EOD data."
    )
    st.stop()


# ── Efficient Frontier ────────────────────────────────────────────────────────

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
        x=f_vols, y=f_rets, mode="lines",
        name="Efficient Frontier",
        line=dict(color="#2563EB", width=2, dash="dash"),
    ))

if current_port is not None:
    fig.add_trace(go.Scatter(
        x=[current_port["Ann Vol"]], y=[current_port["Ann Return"]],
        mode="markers", name="Current Portfolio",
        marker=dict(symbol="diamond", size=14, color="#F59E0B"),
        hovertemplate="<b>Current Portfolio</b><br>Ann Return: %{y:.1%}<br>Ann Vol: %{x:.1%}<extra></extra>",
    ))

for b in basket_stats:
    fig.add_trace(go.Scatter(
        x=[b["Ann Vol"]], y=[b["Ann Return"]],
        mode="markers+text", name=b["Symbol"],
        text=[b["Symbol"]], textposition="top center",
        marker=dict(size=9, opacity=0.75),
        hovertemplate=(
            f"<b>{b['Symbol']}</b><br>Ann Return: %{{y:.1%}}<br>"
            f"Ann Vol: %{{x:.1%}}<br>Beta: {b['Beta']:.2f}<extra></extra>"
        ),
    ))

fig.update_layout(
    template="capital",
    xaxis_title="Annualised Volatility", yaxis_title="Annualised Return",
    xaxis_tickformat=".0%", yaxis_tickformat=".0%",
    legend=dict(orientation="v"), height=520,
    margin=dict(l=60, r=20, t=40, b=60),
)
fig.update_xaxes(rangemode="tozero")

_frontier_ph = st.empty()
_frontier_ph.plotly_chart(fig, width="stretch", key="frontier_base")


# ── Optimiser ─────────────────────────────────────────────────────────────────

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
                "symbols":  [b["Symbol"]     for b in basket_stats],
                "names":    [b["Name"]       for b in basket_stats],
                "ind_vols": [b["Ann Vol"]    for b in basket_stats],
                "ind_rets": [b["Ann Return"] for b in basket_stats],
                "betas":    [b["Beta"]       for b in basket_stats],
            }
        except Exception as e:
            st.error(str(e))
            st.session_state["opt_result"] = None

result = st.session_state["opt_result"]

if result is not None:
    fig2 = go.Figure(fig)
    fig2.add_trace(go.Scatter(
        x=[result["stats"]["Ann Vol"]], y=[result["stats"]["Ann Return"]],
        mode="markers", name="Optimised Portfolio",
        marker=dict(symbol="star", size=18, color="#EF4444"),
        hovertemplate=(
            "<b>Optimised Portfolio</b><br>"
            f"Ann Return: %{{y:.1%}}<br>Ann Vol: %{{x:.1%}}<br>"
            f"Beta: {result['stats']['Beta']:.2f}<extra></extra>"
        ),
    ))
    _frontier_ph.plotly_chart(fig2, width="stretch", key="frontier_opt")

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Ann Return",  f"{result['stats']['Ann Return']:+.2%}")
    m2.metric("Ann Vol",     f"{result['stats']['Ann Vol']:.2%}")
    m3.metric("Portfolio β", f"{result['stats']['Beta']:.2f}")

    st.divider()
    st.subheader("Optimal Weights")

    df_w = pd.DataFrame({
        "Symbol":     result["symbols"],
        "Name":       result["names"],
        "Weight":     result["weights"],
        "Ann Return": result["ind_rets"],
        "Ann Vol":    result["ind_vols"],
        "Beta":       result["betas"],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    bar_fig = go.Figure(go.Bar(
        x=df_w["Symbol"], y=df_w["Weight"],
        marker_color="#2563EB",
        text=df_w["Weight"].map("{:.1%}".format),
        textposition="outside",
    ))
    bar_fig.update_layout(
        template="capital", yaxis_tickformat=".0%", yaxis_title="Weight",
        height=320, margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(bar_fig, width="stretch")

    disp = df_w.copy()
    disp["Weight"]     = disp["Weight"].map("{:.1%}".format)
    disp["Ann Return"] = disp["Ann Return"].map("{:+.1%}".format)
    disp["Ann Vol"]    = disp["Ann Vol"].map("{:.1%}".format)
    disp["Beta"]       = disp["Beta"].map("{:.2f}".format)
    st.dataframe(disp, width="stretch", hide_index=True)

else:
    st.info("Click **Run Optimiser** to generate optimal weights.")


# ── Weighting Scheme Comparison ───────────────────────────────────────────────

st.divider()
st.subheader("Weighting Scheme Comparison")

_REBAL_OPTIONS = {"Daily": "B", "Weekly": "W-FRI", "Monthly": "MS", "Never": None}
_rc1, _rc2 = st.columns([1.5, 4])
rebal_freq_label = _rc1.selectbox("Rebalance frequency", list(_REBAL_OPTIONS.keys()), index=2)
rebal_freq = _REBAL_OPTIONS[rebal_freq_label]
st.caption("1-year backtest of weighting schemes applied to current holdings.")

SCHEME_COLORS = {
    "Equal weight":    "#3B82F6",
    "Inverse vol":     "#10B981",
    "Momentum":        "#F59E0B",
    "Min beta":        "#06B6D4",
    "Current weights": "#A855F7",
    "Optimised":       "#EF4444",
}

with st.spinner("Building comparison…"):
    prices = _cached_price_matrix(version)

    if prices.empty:
        st.warning("Not enough price history to build comparison.")
    else:
        nav_equal = _cached_scheme_nav(prices, "equal",    rebal_freq)
        nav_vol   = _cached_scheme_nav(prices, "inv_vol",  rebal_freq)
        nav_mom   = _cached_scheme_nav(prices, "momentum", rebal_freq)
        nav_beta  = _cached_scheme_nav(prices, "beta",     rebal_freq, spx_rets)

        # Current portfolio weights as static fixed-weight simulation
        nav_current = None
        wgt_hist = get_daily_weightings_history()
        if not wgt_hist.empty:
            latest_w = wgt_hist[wgt_hist["date"] == wgt_hist["date"].max()]
            latest_w = latest_w[~latest_w["symbol"].str.startswith("CASH_")]
            sym_to_idx = {col: i for i, col in enumerate(prices.columns)}
            w_cur = np.zeros(len(prices.columns))
            for _, row in latest_w.iterrows():
                if row["symbol"] in sym_to_idx:
                    w_cur[sym_to_idx[row["symbol"]]] = float(row["pct_nav"])
            if w_cur.sum() > 0:
                w_cur /= w_cur.sum()
                nav_current = _cached_scheme_nav_fixed(prices, w_cur)

        # Optimised weights simulation (available after clicking Run Optimiser)
        nav_opt = None
        opt = st.session_state.get("opt_result")
        if opt is not None:
            sym_to_idx = {col: i for i, col in enumerate(prices.columns)}
            w_opt = np.zeros(len(prices.columns))
            for sym, wt in zip(opt["symbols"], opt["weights"]):
                if sym in sym_to_idx:
                    w_opt[sym_to_idx[sym]] = wt
            if w_opt.sum() > 0:
                w_opt /= w_opt.sum()
                nav_opt = _cached_scheme_nav_fixed(prices, w_opt)

        # ── NAV chart ─────────────────────────────────────────────────────────
        fig_cmp = go.Figure()

        for label, nav, color in [
            ("Equal weight", nav_equal, SCHEME_COLORS["Equal weight"]),
            ("Inverse vol",  nav_vol,   SCHEME_COLORS["Inverse vol"]),
            ("Momentum",     nav_mom,   SCHEME_COLORS["Momentum"]),
            ("Min beta",     nav_beta,  SCHEME_COLORS["Min beta"]),
        ]:
            s = nav.dropna()
            fig_cmp.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=label,
                line=dict(width=2, color=color),
                hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
            ))

        if nav_current is not None:
            s = nav_current.dropna()
            fig_cmp.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name="Current weights",
                line=dict(width=2, color=SCHEME_COLORS["Current weights"]),
                hovertemplate="Current weights: %{y:.3f}<extra></extra>",
            ))

        if nav_opt is not None:
            s = nav_opt.dropna()
            fig_cmp.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name="Optimised",
                line=dict(width=2.5, color=SCHEME_COLORS["Optimised"], dash="dash"),
                hovertemplate="Optimised: %{y:.3f}<extra></extra>",
            ))

        fig_cmp.add_hline(y=1.0, line_width=1, line_color="rgba(255,255,255,0.15)", line_dash="dot")
        fig_cmp.update_layout(
            template="capital",
            title=dict(text="<b>NAV comparison</b> — start at 1.0", x=0, xanchor="left"),
            height=420,
            yaxis=dict(tickformat=".2f", zeroline=False),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=50, r=20, t=48, b=60),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Stats table ───────────────────────────────────────────────────────
        rows = [
            scheme_stats(nav_equal, "Equal weight"),
            scheme_stats(nav_vol,   "Inverse vol"),
            scheme_stats(nav_mom,   "Momentum"),
            scheme_stats(nav_beta,  "Min beta"),
        ]
        if nav_current is not None:
            rows.append(scheme_stats(nav_current, "Current weights"))
        if nav_opt is not None:
            rows.append(scheme_stats(nav_opt, "Optimised"))

        df_stats = pd.DataFrame(rows)
        for col in ("Total return", "Ann return", "Ann vol", "Max drawdown"):
            df_stats[col] = df_stats[col].map("{:+.2%}".format)
        df_stats["Sharpe"] = df_stats["Sharpe"].map("{:.2f}".format)
        st.dataframe(df_stats, hide_index=True, use_container_width=True)
