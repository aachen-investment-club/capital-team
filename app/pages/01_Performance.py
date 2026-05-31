import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import date

from lib.theme import inject_css, PNG_CONFIG, FAVICON
from lib.data import (
    get_portfolio_and_benchmarks,
    get_daily_weightings_history,
    get_theme_mappings,
    get_trade_log,
)

st.set_page_config(page_title="Performance · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Performance")

_TODAY    = date(2026, 5, 29)
_DATE_MIN = date(2026, 5, 6)   # fund inception — first date in nav_history


# ── Holdings table (grouped by theme/category basket) ────────────────────────

def _basket_html(df: pd.DataFrame) -> str:
    NAVY       = "#0C1E40"
    BLUE       = "#2563EB"
    BLUE_LIGHT = "#EFF6FF"
    BORDER     = "#E2E8F0"
    TEXT       = "#0F172A"
    TEXT_MUTED = "#64748B"
    GREEN      = "#10B981"
    RED        = "#EF4444"

    rows_html = ""
    for theme in sorted(df["theme"].dropna().unique()):
        group      = df[df["theme"] == theme].sort_values("pct_nav", ascending=False)
        basket_nav = group["pct_nav"].sum()
        basket_cr  = (
            (group["pct_nav"] * group["cumulative_return"]).sum() / basket_nav
            if basket_nav > 0 else 0.0
        )
        basket_dr  = (
            (group["pct_nav"] * group["daily_return"]).sum() / basket_nav
            if basket_nav > 0 else 0.0
        )
        cr_col = GREEN if basket_cr >= 0 else RED
        dr_col = GREEN if basket_dr >= 0 else RED

        rows_html += f"""
        <tr style="background:{BLUE_LIGHT};border-top:2px solid {BORDER};">
          <td colspan="2" style="padding:9px 14px;font-weight:700;color:{NAVY};font-size:13px;">{theme}</td>
          <td style="padding:9px 14px;text-align:center;color:{TEXT_MUTED};font-size:12px;"></td>
          <td style="padding:9px 14px;font-weight:700;color:{NAVY};text-align:right;">{basket_nav:.2f}%</td>
          <td style="padding:9px 14px;font-weight:700;color:{dr_col};text-align:right;">{basket_dr:+.2%}</td>
          <td style="padding:9px 14px;font-weight:700;color:{cr_col};text-align:right;">{basket_cr:+.2%}</td>
        </tr>"""

        for _, pos in group.iterrows():
            is_cash    = str(pos["symbol"]).startswith("CASH_")
            pos_dr     = pos["daily_return"]
            pos_cr     = pos["cumulative_return"]
            pos_dr_col = GREEN if pos_dr >= 0 else RED
            pos_cr_col = GREEN if pos_cr >= 0 else RED
            dr_str     = "–" if is_cash and pos["ccy"] == "EUR" else f"{pos_dr:+.2%}"
            cr_str     = "–" if is_cash and pos["ccy"] == "EUR" else f"{pos_cr:+.2%}"
            rows_html += f"""
        <tr style="border-bottom:1px solid {BORDER};">
          <td style="padding:6px 14px 6px 30px;color:{TEXT};font-size:12px;font-weight:500;">{pos["symbol"]}</td>
          <td style="padding:6px 14px;color:{TEXT_MUTED};font-size:12px;">{pos["name"]}</td>
          <td style="padding:6px 14px;text-align:center;color:{TEXT_MUTED};font-size:11px;">{pos["ccy"]}</td>
          <td style="padding:6px 14px;text-align:right;color:{TEXT};font-size:12px;">{pos["pct_nav"]:.2f}%</td>
          <td style="padding:6px 14px;text-align:right;color:{pos_dr_col};font-size:12px;">{dr_str}</td>
          <td style="padding:6px 14px;text-align:right;color:{pos_cr_col};font-size:12px;">{cr_str}</td>
        </tr>"""

    return f"""
<div style="border:1px solid {BORDER};border-radius:8px;overflow:hidden;font-family:'Inter',sans-serif;">
<table style="width:100%;border-collapse:collapse;">
<thead>
<tr style="border-bottom:2px solid {BLUE};background:#fff;">
  <th style="padding:10px 14px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Symbol</th>
  <th style="padding:10px 14px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Name</th>
  <th style="padding:10px 14px;text-align:center;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">CCY</th>
  <th style="padding:10px 14px;text-align:right;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">% NAV</th>
  <th style="padding:10px 14px;text-align:right;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Day</th>
  <th style="padding:10px 14px;text-align:right;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Since Inception</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>"""


# ── Shared: join weightings with theme mappings ───────────────────────────────

@st.cache_data
def _weightings_with_themes() -> pd.DataFrame:
    df     = get_daily_weightings_history()
    themes = get_theme_mappings()

    if not themes.empty:
        df = df.merge(themes[["symbol", "theme"]], on="symbol", how="left")
    else:
        df["theme"] = None

    # Fallback: use category as theme when DynamoDB has no mapping
    no_theme = df["theme"].isna()
    df.loc[no_theme, "theme"] = df.loc[no_theme, "category"]

    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Returns vs Benchmark
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Returns vs Benchmark")

try:
    df_bench_all = get_portfolio_and_benchmarks()
    available_bm = sorted(t for t in df_bench_all["ticker"].unique() if t != "PORTFOLIO")
    bench_ok     = not df_bench_all.empty
except Exception:
    bench_ok     = False
    df_bench_all = None

if not bench_ok:
    st.info("Benchmark data not yet available — run the precompute job when S3 data is present.")
else:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        perf_start = st.date_input("From", value=_DATE_MIN,
                                   min_value=_DATE_MIN, max_value=_TODAY, key="perf_start")
    with c2:
        perf_end = st.date_input("To", value=_TODAY,
                                 min_value=_DATE_MIN, max_value=_TODAY, key="perf_end")
    with c3:
        selected_bm = st.multiselect("Benchmarks", available_bm, default=available_bm[:1], key="benchmarks")

    if perf_start < perf_end:
        tickers_to_plot = ["PORTFOLIO"] + selected_bm
        df_perf = (
            df_bench_all[
                df_bench_all["date"].between(pd.Timestamp(perf_start), pd.Timestamp(perf_end))
                & df_bench_all["ticker"].isin(tickers_to_plot)
            ]
            .copy()
            .sort_values(["ticker", "date"])
        )

        tab_idx, tab_ret = st.tabs(["Index", "Returns"])
        with tab_idx:
            fig = px.line(df_perf, x="date", y="index_value", color="ticker",
                          labels={"index_value": "NAV Multiple (inception = 1.0)", "date": "Date", "ticker": ""},
                          template="capital")
            fig.add_hline(y=1.0, line_dash="dot", line_color="#BFDBFE")
            st.plotly_chart(fig, use_container_width=True, config=PNG_CONFIG)
        with tab_ret:
            fig = px.bar(df_perf, x="date", y="daily_return", color="ticker", barmode="group",
                         labels={"daily_return": "Daily Return", "date": "Date", "ticker": ""},
                         template="capital")
            fig.update_layout(yaxis_tickformat=".1%")
            fig.add_hline(y=0, line_dash="dot", line_color="#BFDBFE")
            st.plotly_chart(fig, use_container_width=True, config=PNG_CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single Positions Performance
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Single Positions Performance")

c1, c2 = st.columns(2)
with c1:
    pos_start = st.date_input("From", value=_DATE_MIN,
                              min_value=_DATE_MIN, max_value=_TODAY, key="pos_start")
with c2:
    pos_end = st.date_input("To", value=_TODAY,
                            min_value=_DATE_MIN, max_value=_TODAY, key="pos_end")

if pos_start >= pos_end:
    st.warning("'From' must be before 'To'.")
else:
    df_wh = _weightings_with_themes()
    df_pos = (
        df_wh[
            df_wh["date"].between(pd.Timestamp(pos_start), pd.Timestamp(pos_end))
            & ~df_wh["symbol"].str.startswith("CASH_")
        ]
        .copy()
        .sort_values(["symbol", "date"])
    )

    start_cr          = df_pos.groupby("symbol")["cumulative_return"].transform("first")
    df_pos["period_return"] = (1 + df_pos["cumulative_return"]) / (1 + start_cr) - 1

    tab_line, tab_table = st.tabs(["Line Chart", "Summary Table"])

    with tab_line:
        fig = px.line(df_pos, x="date", y="period_return", color="symbol",
                      hover_data=["name"],
                      labels={"period_return": "Return", "date": "Date", "symbol": ""},
                      template="capital")
        fig.update_layout(yaxis_tickformat=".1%")
        fig.add_hline(y=0, line_dash="dot", line_color="#BFDBFE")
        st.plotly_chart(fig, use_container_width=True, config=PNG_CONFIG)

    with tab_table:
        summary = (
            df_pos.groupby(["symbol", "name", "category", "theme"])
            .agg(period_return=("period_return", "last"),
                 peak=("period_return", "max"),
                 trough=("period_return", "min"))
            .reset_index()
            .sort_values("period_return", ascending=False)
        )
        disp = summary.copy()
        for col in ["period_return", "peak", "trough"]:
            disp[col] = disp[col].map("{:+.2%}".format)
        st.dataframe(
            disp.rename(columns={"symbol": "Symbol", "name": "Name",
                                  "category": "Asset Class", "theme": "Theme",
                                  "period_return": "Period Return",
                                  "peak": "Max", "trough": "Min"}),
            use_container_width=True, hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Portfolio Weightings
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Portfolio Weightings")

weight_date = st.date_input("Date", value=_TODAY,
                            min_value=_DATE_MIN, max_value=_TODAY, key="weight_date")

df_wh_full  = _weightings_with_themes()
avail_dates = df_wh_full["date"].dt.date.unique()
valid_dates = sorted(d for d in avail_dates if d <= weight_date)
snap_date   = valid_dates[-1] if valid_dates else sorted(avail_dates)[-1]
df_snap     = df_wh_full[df_wh_full["date"].dt.date == snap_date].copy()


def _pie(df: pd.DataFrame, names: str, values: str) -> object:
    fig = px.pie(df, names=names, values=values, hole=0.35, template="capital")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=24, b=24, l=24, r=24))
    return fig


tab_w_pos, tab_w_class, tab_w_theme = st.tabs(["By Position", "By Asset Class", "By Theme"])

with tab_w_pos:
    st.plotly_chart(_pie(df_snap, "symbol", "pct_nav"), use_container_width=True, config=PNG_CONFIG)

with tab_w_class:
    by_class = df_snap.groupby("category", as_index=False)["pct_nav"].sum()
    st.plotly_chart(_pie(by_class, "category", "pct_nav"), use_container_width=True, config=PNG_CONFIG)

with tab_w_theme:
    by_theme = df_snap.groupby("theme", as_index=False)["pct_nav"].sum()
    st.plotly_chart(_pie(by_theme, "theme", "pct_nav"), use_container_width=True, config=PNG_CONFIG)

st.subheader(f"Holdings — {snap_date.strftime('%d %b %Y')}")
st.html(_basket_html(df_snap))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trade Log
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Trade Log")

df_trades    = get_trade_log()
periods      = sorted(df_trades["trade_date"].dt.to_period("M").unique(), reverse=True)
month_labels = [p.strftime("%B %Y") for p in periods]
default_label = pd.Timestamp(_TODAY).to_period("M").strftime("%B %Y")
default_idx  = month_labels.index(default_label) if default_label in month_labels else 0

selected_label  = st.selectbox("Month", month_labels, index=default_idx, key="trade_month")
selected_period = periods[month_labels.index(selected_label)]

df_month = df_trades[df_trades["trade_date"].dt.to_period("M") == selected_period].copy()

if df_month.empty:
    st.info("No trades recorded for this month.")
else:
    disp = df_month.copy()
    disp["trade_date"]  = disp["trade_date"].dt.strftime("%Y-%m-%d")
    disp["quantity"]    = disp["quantity"].map("{:,.4f}".format)
    disp["avg_price"]   = disp["avg_price"].map("{:,.4f}".format)
    disp["proceeds"]    = disp["proceeds"].map("{:,.2f}".format)
    disp["commission"]  = disp["commission"].map("{:,.2f}".format)
    disp["asset_type"]  = disp["asset_type"].map({"COMMON": "Stock", "ETF": "ETF", "FX": "FX"})
    st.dataframe(
        disp.rename(columns={
            "trade_date": "Date", "symbol": "Symbol", "name": "Name",
            "currency": "CCY", "asset_type": "Type", "buy_sell": "Side",
            "quantity": "Qty", "avg_price": "Avg Price",
            "proceeds": "Total Cost", "commission": "Commission",
        }),
        use_container_width=True, hide_index=True,
        column_order=["Date", "Type", "Symbol", "Name", "CCY",
                      "Side", "Qty", "Avg Price", "Total Cost", "Commission"],
    )
