import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import date

from lib.theme import inject_css, FAVICON, NAVY
from lib.data import get_daily_weightings_history, get_portfolio_and_benchmarks

st.set_page_config(page_title="Risk KPIs · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Risk KPIs")

_TODAY    = date.today() - pd.Timedelta(days=1)
_DATE_MIN = date(2026, 5, 6) # Start of trade log

BENCHMARKS = {
    "S&P 500":        "SPX",
    "MSCI World":     "MSCI_WORLD",
    "MSCI Europe":    "MSCI_EUROPE",
    "60/40 Balanced": "60_40",
}

# ── Controls ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
with c1:
    risk_start = st.date_input(
        "From", value=_DATE_MIN, min_value=_DATE_MIN, max_value=_TODAY, key="risk_start"
    )
with c2:
    risk_end = st.date_input(
        "To", value=_TODAY, min_value=_DATE_MIN, max_value=_TODAY, key="risk_end"
    )
with c3:
    bm_label  = st.selectbox("Benchmark (Beta / Alpha)", list(BENCHMARKS.keys()), key="bm_sel")
    bm_ticker = BENCHMARKS[bm_label]
with c4:
    rf_pct  = st.slider("Risk-Free Rate (% p.a.)", 0.0, 10.0, 3.5, 0.1, key="rf_rate")
    rf_rate = rf_pct / 100.0

if risk_start >= risk_end:
    st.warning("'From' date must be before 'To' date.")
    st.stop()


# ── Data helpers ───────────────────────────────────────────────────────────────

@st.cache_data
def _portfolio_daily_returns(start: date, end: date) -> pd.Series:
    df = get_portfolio_and_benchmarks()
    mask = (df["ticker"] == "PORTFOLIO") & df["date"].between(
        pd.Timestamp(start), pd.Timestamp(end)
    )
    return df[mask].set_index("date")["daily_return"].sort_index()


@st.cache_data
def _stock_daily_returns(symbol: str, start: date, end: date) -> pd.Series:
    df = get_daily_weightings_history()
    mask = (df["symbol"] == symbol) & df["date"].between(
        pd.Timestamp(start), pd.Timestamp(end)
    )
    return df[mask].set_index("date")["daily_return"].sort_index()


@st.cache_data
def _benchmark_daily_returns(ticker: str, start: date, end: date) -> pd.Series:
    df = get_portfolio_and_benchmarks()
    mask = (df["ticker"] == ticker) & df["date"].between(
        pd.Timestamp(start), pd.Timestamp(end)
    )
    return df[mask].set_index("date")["daily_return"].sort_index()


def _compute_kpis(r: pd.Series, bm_r: pd.Series, rf: float) -> dict:
    aligned = pd.DataFrame({"r": r, "bm": bm_r}).dropna()
    if len(aligned) < 5:
        return {}

    r    = aligned["r"]
    bm_r = aligned["bm"]
    rf_d = rf / 252.0

    vol     = float(r.std() * np.sqrt(252))
    ann_ret = float((1 + r).prod() ** (252.0 / len(r)) - 1)

    excess  = r - rf_d
    sharpe  = float(excess.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")

    downside = r[r < rf_d] - rf_d
    ds_std   = float(downside.std()) if len(downside) > 1 else float("nan")
    sortino  = float(excess.mean() / ds_std * np.sqrt(252)) if (ds_std and ds_std > 0) else float("nan")

    cum    = (1 + r).cumprod()
    dd     = cum / cum.cummax() - 1
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else float("nan")

    cov_m = np.cov(r.values, bm_r.values)
    beta  = float(cov_m[0, 1] / cov_m[1, 1]) if cov_m[1, 1] > 0 else float("nan")
    alpha = (
        float((excess.mean() - beta * (bm_r - rf_d).mean()) * 252)
        if not np.isnan(beta) else float("nan")
    )

    return {
        "vol": vol, "ann_ret": ann_ret, "sharpe": sharpe,
        "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
        "beta": beta, "alpha": alpha,
        "r": r, "dd": dd, "cum": cum, "n": len(r),
    }


@st.cache_data
def _compute_portfolio_kpis(start: date, end: date, bm: str, rf: float) -> dict:
    r    = _portfolio_daily_returns(start, end)
    bm_r = _benchmark_daily_returns(bm, start, end)
    return _compute_kpis(r, bm_r, rf)


@st.cache_data
def _compute_stock_kpis(symbol: str, start: date, end: date, bm: str, rf: float) -> dict:
    r    = _stock_daily_returns(symbol, start, end)
    bm_r = _benchmark_daily_returns(bm, start, end)
    return _compute_kpis(r, bm_r, rf)


@st.cache_data
def _equity_symbols() -> list[tuple[str, str]]:
    df = get_daily_weightings_history()
    stocks = df[df["category"] != "Cash"][["symbol", "name"]].drop_duplicates()
    return sorted(stocks.itertuples(index=False, name=None), key=lambda x: x[0])


# ── Shared rendering helpers ───────────────────────────────────────────────────

def _fmt_pct(v: float) -> str:
    return f"{v:.1%}" if not np.isnan(v) else "–"

def _fmt_x(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if not np.isnan(v) else "–"


def _render_kpis(kpis: dict, label: str, rf_pct: float, bm_label: str) -> None:
    st.caption(
        f"{kpis['n']} trading days · "
        f"{risk_start.strftime('%d %b %Y')} → {risk_end.strftime('%d %b %Y')} · "
        f"Benchmark: {bm_label} · rf = {rf_pct:.1f}%"
    )

    cols = st.columns(7)
    cards = [
        ("Volatility (ann.)", _fmt_pct(kpis["vol"])),
        ("Sharpe Ratio",      _fmt_x(kpis["sharpe"])),
        ("Sortino Ratio",     _fmt_x(kpis["sortino"])),
        ("Max Drawdown",      _fmt_pct(kpis["max_dd"])),
        ("Calmar Ratio",      _fmt_x(kpis["calmar"])),
        ("Beta",              _fmt_x(kpis["beta"])),
        ("Alpha (ann.)",      _fmt_pct(kpis["alpha"])),
    ]
    for col, (card_label, value) in zip(cols, cards):
        with col:
            st.metric(card_label, value)

    st.divider()

    r    = kpis["r"]
    dd   = kpis["dd"]
    rf_d = rf_rate / 252.0

    tab_dd, tab_vol, tab_sharpe = st.tabs(["Drawdown", "Rolling Volatility", "Rolling Sharpe"])

    with tab_dd:
        fig = go.Figure(go.Scatter(
            x=dd.index, y=dd.values,
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.12)",
            line=dict(color="#EF4444", width=1.5),
            hovertemplate="%{x|%d %b %Y}: %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{label} Drawdown",
            yaxis_tickformat=".0%",
            yaxis_title="Drawdown",
            xaxis_title="",
            showlegend=False,
            template="capital",
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, width="stretch")

    with tab_vol:
        rv30 = r.rolling(30).std() * np.sqrt(252)
        rv90 = r.rolling(90).std() * np.sqrt(252)
        fig  = go.Figure()
        fig.add_trace(go.Scatter(
            x=rv30.index, y=rv30.values, name="30-day",
            line=dict(color=NAVY, width=2),
            hovertemplate="%{x|%d %b %Y}: %{y:.1%}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=rv90.index, y=rv90.values, name="90-day",
            line=dict(color="#B8962E", width=2),
            hovertemplate="%{x|%d %b %Y}: %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{label} Rolling Volatility (Annualized)",
            yaxis_tickformat=".0%",
            yaxis_title="Volatility",
            xaxis_title="",
            template="capital",
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, width="stretch")

    with tab_sharpe:
        rs90 = (r - rf_d).rolling(90).mean() / r.rolling(90).std() * np.sqrt(252)
        fig  = go.Figure(go.Scatter(
            x=rs90.index, y=rs90.values,
            line=dict(color=NAVY, width=2),
            hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#BFDBFE")
        fig.update_layout(
            title=f"{label} Rolling Sharpe Ratio (90-day)",
            yaxis_title="Sharpe Ratio",
            xaxis_title="",
            showlegend=False,
            template="capital",
        )
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, width="stretch")


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_portfolio, tab_stock = st.tabs(["Portfolio", "Single Stock"])

with tab_portfolio:
    try:
        kpis = _compute_portfolio_kpis(risk_start, risk_end, bm_ticker, rf_rate)
    except Exception as exc:
        st.error(f"Could not compute KPIs: {exc}")
        st.stop()

    if not kpis:
        st.info("Not enough data in the selected range.")
    else:
        _render_kpis(kpis, "Portfolio", rf_pct, bm_label)

with tab_stock:
    symbols = _equity_symbols()
    if not symbols:
        st.info("No equity positions found.")
    else:
        options      = [f"{sym} – {name}" for sym, name in symbols]
        sym_map      = {f"{sym} – {name}": sym for sym, name in symbols}
        selected_opt = st.selectbox("Stock", options, key="stock_sel")
        selected_sym = sym_map[selected_opt]

        try:
            kpis = _compute_stock_kpis(selected_sym, risk_start, risk_end, bm_ticker, rf_rate)
        except Exception as exc:
            st.error(f"Could not compute KPIs: {exc}")
            st.stop()

        if not kpis:
            st.info("Not enough data for this stock in the selected range.")
        else:
            _render_kpis(kpis, selected_sym, rf_pct, bm_label)
