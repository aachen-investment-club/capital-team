import sys
import pathlib
import math
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from lib.theme import inject_css, FAVICON
from lib.data import get_security_master, get_eod_prices, _eod_data_version

st.set_page_config(page_title="Equities · AIC", page_icon=FAVICON, layout="wide")
inject_css()

st.title("Equities")
with st.popover("ℹ"):
    st.markdown("""Ranks each holding by standard investment factors (momentum, valuation, and quality) so you can see at a glance which stocks are looking strong or weak relative to the rest.""")

master = get_security_master()

if master.empty:
    st.warning("Security master is empty. Add securities to config/security_master.csv.")
    st.stop()

labels = (master["ticker"] + "  —  " + master["name"]).tolist()
label_to_id = dict(zip(labels, master["security_id"].tolist()))

# ── Controls ──────────────────────────────────────────────────────────────────

PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "All"]

col_sel, col_period, col_vol = st.columns([3, 5, 1])
with col_sel:
    selected_label = st.selectbox("Security", labels, label_visibility="collapsed")
with col_period:
    period = st.radio("Period", PERIODS, index=4, horizontal=True, label_visibility="collapsed")
with col_vol:
    show_volume = st.checkbox("Volume", value=True)

# ── Data ──────────────────────────────────────────────────────────────────────

security_id = label_to_id[selected_label]
df = get_eod_prices(security_id, cache_version=_eod_data_version())

if df.empty:
    st.info(
        "No EOD data for this security yet. "
        "Run `python scripts/ingest_eod.py` to fetch history."
    )
    st.stop()

df["date"] = pd.to_datetime(df["date"]).dt.date


def _filter_period(df: pd.DataFrame, p: str) -> pd.DataFrame:
    if p == "All" or df.empty:
        return df
    today = date.today()
    cutoff: date = {
        "1M":  today - timedelta(days=30),
        "3M":  today - timedelta(days=91),
        "6M":  today - timedelta(days=182),
        "YTD": date(today.year, 1, 1),
        "1Y":  today - timedelta(days=365),
        "3Y":  today - timedelta(days=3 * 365),
        "5Y":  today - timedelta(days=5 * 365),
    }[p]
    return df[df["date"] >= cutoff].reset_index(drop=True)


view = _filter_period(df, period)

if view.empty:
    st.info(f"No data in the selected period ({period}). Try a longer window.")
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────

latest = view.iloc[-1]
ticker_label = selected_label.split("  —  ")[0]

# Compute return and volatility on the full series, not the period view
all_returns = df["close"].pct_change().dropna()
n_calendar_days = max((df["date"].iloc[-1] - df["date"].iloc[0]).days, 1)
total_ret  = df["close"].iloc[-1] / df["close"].iloc[0] - 1
ann_return = (1 + total_ret) ** (365.25 / n_calendar_days) - 1
ann_vol    = all_returns.std() * math.sqrt(252) if len(all_returns) > 1 else None

st.caption(f"Available history: {df['date'].min()} → {df['date'].max()}")

c1, c2, c3 = st.columns(3)
c1.metric("Latest Close",    f"{latest['close']:,.2f}")
c2.metric("Ann. Return",     f"{ann_return * 100:+.1f}%")
c3.metric("Ann. Volatility", f"{ann_vol * 100:.1f}%" if ann_vol is not None else "—")

# ── Candlestick chart ─────────────────────────────────────────────────────────

if show_volume:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.03,
    )
    price_row, vol_row = 1, 2
else:
    fig = make_subplots(rows=1, cols=1)
    price_row, vol_row = 1, None

fig.add_trace(
    go.Candlestick(
        x=view["date"],
        open=view["open"], high=view["high"],
        low=view["low"],   close=view["close"],
        name=ticker_label,
        increasing_line_color="#2E7D6B",
        decreasing_line_color="#8B2E4A",
        increasing_fillcolor="#2E7D6B",
        decreasing_fillcolor="#8B2E4A",
    ),
    row=price_row, col=1,
)

if show_volume and vol_row is not None:
    bar_colors = [
        "#2E7D6B" if c >= o else "#8B2E4A"
        for c, o in zip(view["close"], view["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=view["date"], y=view["volume"],
            name="Volume",
            marker_color=bar_colors,
            marker_opacity=0.55,
            showlegend=False,
        ),
        row=vol_row, col=1,
    )

fig.update_layout(
    title=ticker_label,
    template="capital",
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    showlegend=False,
    height=520 if show_volume else 400,
)
fig.update_yaxes(title_text="Price", row=price_row, col=1)
if show_volume and vol_row is not None:
    fig.update_yaxes(title_text="", showticklabels=False, row=vol_row, col=1)

st.plotly_chart(fig, width="stretch")
