import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from lib.theme import inject_css, FAVICON


# Human-readable labels for ticker codes — used in chart legends and exports.
DISPLAY_NAMES: dict[str, str] = {
    "PORTFOLIO":   "AIC Portfolio",
    "60_40":       "60/40 Balanced",
    "SPX":         "S&P 500",
    "MSCI_WORLD":  "MSCI World",
    "MSCI_EUROPE": "MSCI Europe",
}


# Light-on-dark colours for the transparent matplotlib export (navy background).
_RETURNS_COLORS_LIGHT: dict[str, str] = {
    "PORTFOLIO":   "#93C5FD",  # blue-300   — primary, brightest
    "SPX":         "#9CA3AF",  # grey-400
    "MSCI_WORLD":  "#60A5FA",  # blue-400
    "MSCI_EUROPE": "#6B7280",  # grey-500
    "60_40":       "#CBD5E1",  # slate-300  — lightest grey
}

# Dark → light blue-gray gradient
_BLUES_PALETTE = [
    "#0C2A4A",  # very dark navy
    "#1A3F6F",  # dark blue
    "#2E5E9E",  # medium-dark blue
    "#4279BC",  # medium blue
    "#5A94D0",  # medium-light blue
    "#74ADE0",  # light blue
    "#8FBFEC",  # lighter blue
    "#8090A8",  # blue-grey transition
    "#94A3B8",  # slate-blue
    "#A8B8CC",  # light blue-grey
    "#BDC9D8",  # pale blue-grey
    "#D1DCE5",  # very pale
    "#E2E8F0",  # almost white-grey
]

def _palette_sample(palette: list[str], n: int) -> list[str]:
    """Pick n colours equidistantly from palette so any subset spans the full range."""
    if n <= 1:
        return [palette[0]]
    return [palette[round(i * (len(palette) - 1) / (n - 1))] for i in range(n)]


from lib.data import (
    get_portfolio_and_benchmarks,
    get_daily_weightings_history,
    get_theme_mappings,
    get_trade_log,
)

st.set_page_config(page_title="Performance · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Performance")

_TODAY    = date.today()
_DATE_MIN = date(2026, 5, 6)   # fund inception — first date in nav_history


# ── Holdings table (grouped by theme/category basket) ────────────────────────

def _basket_html(df: pd.DataFrame) -> str:
    NAVY       = "#0C1E40"
    BLUE_LIGHT = "#EFF6FF"
    BORDER     = "#E2E8F0"
    TEXT       = "#0F172A"
    TEXT_MUTED = "#64748B"
    GREEN      = "#10B981"
    RED        = "#EF4444"

    rows_html = ""
    all_themes   = sorted(df["theme"].dropna().unique())
    cash_themes  = [t for t in all_themes
                    if df[df["theme"] == t]["symbol"].str.startswith("CASH_").all()]
    other_themes = [t for t in all_themes if t not in set(cash_themes)]
    for theme in other_themes + cash_themes:
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
            is_base_ccy = pos["symbol"] == "CASH_EUR"
            pos_dr     = pos["daily_return"]
            pos_cr     = pos["cumulative_return"]
            pos_dr_col = GREEN if pos_dr >= 0 else RED
            pos_cr_col = GREEN if pos_cr >= 0 else RED
            dr_str     = "–" if is_base_ccy else f"{pos_dr:+.2%}"
            cr_str     = "–" if is_base_ccy else f"{pos_cr:+.2%}"
            rows_html += f"""
        <tr style="border-bottom:1px solid {BORDER};">
          <td style="padding:6px 14px 6px 30px;color:{TEXT};font-size:12px;font-weight:500;">{pos["symbol"]}</td>
          <td style="padding:6px 14px;color:{TEXT_MUTED};font-size:12px;">{pos["name"]}</td>
          <td style="padding:6px 14px;text-align:left;color:{TEXT_MUTED};font-size:11px;">{pos.get("isin", "")}</td>
          <td style="padding:6px 14px;text-align:right;color:{TEXT};font-size:12px;">{pos["pct_nav"]:.2f}%</td>
          <td style="padding:6px 14px;text-align:right;color:{pos_dr_col};font-size:12px;">{dr_str}</td>
          <td style="padding:6px 14px;text-align:right;color:{pos_cr_col};font-size:12px;">{cr_str}</td>
        </tr>"""

    return f"""
<div style="border:1px solid {BORDER};border-radius:8px;overflow:hidden;font-family:'Inter',sans-serif;">
<table style="width:100%;border-collapse:collapse;">
<thead>
<tr style="border-bottom:2px solid {BORDER};background:#fff;">
  <th style="padding:10px 14px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Symbol</th>
  <th style="padding:10px 14px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Name</th>
  <th style="padding:10px 14px;text-align:left;color:{TEXT_MUTED};font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.05em;">ISIN</th>
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
# Matplotlib export for Returns vs Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _export_returns_matplotlib(df_perf: pd.DataFrame) -> list[str]:
    """Save Returns vs Benchmark charts as transparent-background PNGs to ~/Downloads."""
    from pathlib import Path
    from datetime import datetime as _dt

    downloads = Path.home() / "Downloads"
    saved: list[str] = []

    # PORTFOLIO first, then remaining tickers alphabetically
    _all = df_perf["ticker"].unique()
    tickers = (["PORTFOLIO"] if "PORTFOLIO" in _all else []) + sorted(
        t for t in _all if t != "PORTFOLIO"
    )

    # ── Typography & colour constants ─────────────────────────────────────────
    F_LABEL  = 18   # axis labels
    F_TICK   = 15   # tick labels
    F_LEGEND = 15   # legend text
    OFFWHITE = "#CBD5E1"  # labels, ticks — same colour intentionally
    GRID_C   = (1.0, 1.0, 1.0, 0.08)   # barely-visible white grid
    REF_C    = "#93C5FD"                # dotted reference line (light blue)

    _series_colors = [_RETURNS_COLORS_LIGHT.get(t, "#94A3B8") for t in tickers]

    def _style(fig, ax) -> None:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=OFFWHITE, labelsize=F_TICK, length=0)
        ax.xaxis.label.set_color(OFFWHITE)
        ax.yaxis.label.set_color(OFFWHITE)
        ax.grid(True, color=GRID_C, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    def _legend(ax) -> None:
        leg = ax.legend(
            fontsize=F_LEGEND,
            framealpha=0.25,
            facecolor="#0C1E40",
            edgecolor="none",
            labelcolor=OFFWHITE,
        )
        for line in leg.get_lines():
            line.set_linewidth(3)

    def _save(fig, path: str) -> None:
        fig.savefig(path, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        saved.append(path)

    # ── 1. Index — line chart ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axhline(1.0, color=REF_C, linestyle=":", linewidth=1.2, zorder=1)
    for i, ticker in enumerate(tickers):
        sub = df_perf[df_perf["ticker"] == ticker].sort_values("date")
        ax.plot(sub["date"], sub["index_value"],
                label=DISPLAY_NAMES.get(ticker, ticker), color=_series_colors[i],
                linewidth=3, solid_capstyle="round", zorder=2)
    _style(fig, ax)
    ax.set_xlabel("Date", fontsize=F_LABEL, labelpad=10)
    ax.set_ylabel("Index", fontsize=F_LABEL, labelpad=10)
    ax.tick_params(axis="x", rotation=30)
    _legend(ax)
    fig.tight_layout()
    _save(fig, str(downloads / f"returns_index.png"))

    # ── 2. Daily returns — grouped bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axhline(0, color=REF_C, linestyle=":", linewidth=1.2, zorder=1)
    dates = sorted(df_perf["date"].unique())
    n = len(tickers)
    bar_w = 0.72 / max(n, 1)
    date_idx = {d: i for i, d in enumerate(dates)}
    for i, ticker in enumerate(tickers):
        sub = df_perf[df_perf["ticker"] == ticker].sort_values("date")
        xs = [date_idx[d] + (i - n / 2 + 0.5) * bar_w for d in sub["date"]]
        ax.bar(xs, sub["daily_return"].values, width=bar_w,
               label=DISPLAY_NAMES.get(ticker, ticker), color=_series_colors[i],
               alpha=0.9, zorder=2)
    step = max(1, len(dates) // 10)
    x_all = list(range(len(dates)))
    ax.set_xticks(x_all[::step])
    ax.set_xticklabels(
        [pd.Timestamp(dates[j]).strftime("%d %b") for j in x_all[::step]],
        rotation=30, ha="right",
    )
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    _style(fig, ax)
    ax.set_xlabel("Date", fontsize=F_LABEL, labelpad=10)
    ax.set_ylabel("Daily Return", fontsize=F_LABEL, labelpad=10)
    _legend(ax)
    fig.tight_layout()
    _save(fig, str(downloads / f"returns_bar.png"))

    return saved


# ─────────────────────────────────────────────────────────────────────────────
#  Returns vs Benchmark
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
        selected_bm = st.multiselect(
            "Benchmarks", available_bm,
            default=[],
            key="benchmarks",
        )

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

        df_perf_plot = df_perf.copy()
        df_perf_plot["ticker"] = df_perf_plot["ticker"].map(lambda t: DISPLAY_NAMES.get(t, t))
        portfolio_label = DISPLAY_NAMES.get("PORTFOLIO", "PORTFOLIO")
        ticker_order = [portfolio_label] + sorted(
            t for t in df_perf_plot["ticker"].unique() if t != portfolio_label
        )
        bm_colors_display = dict(zip(ticker_order, _palette_sample(_BLUES_PALETTE, len(ticker_order))))

        tab_idx, tab_ret = st.tabs(["Index", "Returns"])
        with tab_idx:
            fig = px.line(df_perf_plot, x="date", y="index_value", color="ticker",
                          color_discrete_map=bm_colors_display,
                          category_orders={"ticker": ticker_order},
                          labels={"index_value": "Index", "date": "Date", "ticker": ""},
                          template="capital")
            fig.add_hline(y=1.0, line_dash="dot", line_color="#BFDBFE")
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, width="stretch")
        with tab_ret:
            fig = px.bar(df_perf_plot, x="date", y="daily_return", color="ticker", barmode="group",
                         color_discrete_map=bm_colors_display,
                         category_orders={"ticker": ticker_order},
                         labels={"daily_return": "Daily Return", "date": "Date", "ticker": ""},
                         template="capital")
            fig.update_layout(yaxis_tickformat=".1%")
            fig.add_hline(y=0, line_dash="dot", line_color="#BFDBFE")
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
        clicked = st.button("Export Figures", key="export_returns_mpl")
        st.markdown('</div>', unsafe_allow_html=True)
        if clicked:
            try:
                paths = _export_returns_matplotlib(df_perf)
                for p in paths:
                    st.success(f"Saved: {p}")
            except Exception as exc:
                st.error(f"Export failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib export for Portfolio Weightings pies
# ─────────────────────────────────────────────────────────────────────────────

_CASH_CCY_RANK = {"EUR": 0, "USD": 1, "GBP": 2}

def _sort_pos_cash_first(df: pd.DataFrame) -> pd.DataFrame:
    """Return df_snap with CASH_* rows first (EUR→USD→GBP→rest), then equities by pct_nav desc."""
    is_cash = df["symbol"].str.startswith("CASH_")
    cash = df[is_cash].copy()
    cash["_r"] = cash["symbol"].str.replace("CASH_", "", regex=False).map(
        lambda c: _CASH_CCY_RANK.get(c, 99)
    )
    cash = cash.sort_values("_r").drop(columns="_r")
    equity = df[~is_cash].sort_values("pct_nav", ascending=False)
    return pd.concat([cash, equity], ignore_index=True)


def _export_pies_matplotlib(df_snap: pd.DataFrame) -> list[str]:
    """Save all three Portfolio Weightings pies as transparent PNGs to ~/Downloads."""
    from pathlib import Path
    from datetime import datetime as _dt

    downloads = Path.home() / "Downloads"
    saved: list[str] = []

    _LEADER_MIN = 1.0   # label slices >= this %

    def _save_pie(labels, values, suffix):
        import math
        total = sum(values)
        pcts = [v / total * 100 for v in values]
        colors = [_BLUES_PALETTE[i % len(_BLUES_PALETTE)] for i in range(len(labels))]

        _HOLE   = 0.4          # inner radius of donut ring
        _R_TEXT = (_HOLE + 1.0) / 2  # ≈ 0.72 — midpoint of the ring

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_alpha(0)
        fig.set_facecolor((0, 0, 0, 0))
        ax.set_facecolor((0, 0, 0, 0))
        wedges, _ = ax.pie(
            values, labels=None, colors=colors,
            autopct=None,
            startangle=0, counterclock=False,
            wedgeprops=dict(width=1 - _HOLE,
                            linewidth=0.5, edgecolor=(1, 1, 1, 0.12)),
        )
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Inside-slice labels
        _RIM_INSET = 0.02  # distance inward from the outer rim for small labels
        for wedge, label, pct in zip(wedges, labels, pcts):
            if pct < _LEADER_MIN:
                continue
            mid_deg = (wedge.theta1 + wedge.theta2) / 2
            mid_rad = math.radians(mid_deg)
            small = pct < 5.0
            r = (1.0 - _RIM_INSET) if small else _R_TEXT
            x = r * math.cos(mid_rad)
            y = r * math.sin(mid_rad)
            fs = 9 if small else max(9, min(13, int(pct ** 0.5 * 3.5)))
            text = f"{label}\n{pct:.1f}%" if not small else f"{label}  {pct:.1f}%"
            if small:
                rot = mid_deg % 360
                flipped = 90 < rot < 270
                if flipped:
                    rot -= 180
                # align the rim-facing end of the text to the anchor point
                ha = "left" if flipped else "right"
            else:
                rot = 0
                ha = "center"
            ax.text(x, y, text,
                    ha=ha, va="center",
                    fontsize=fs, color="white",
                    fontweight="600" if pct >= 5 else "normal",
                    rotation=rot, rotation_mode="anchor",
                    linespacing=1.3)

        path = str(downloads / f"weights_{suffix}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        saved.append(path)

    # By Position — cash first
    df_pos = _sort_pos_cash_first(df_snap)
    _save_pie(df_pos["symbol"].tolist(), df_pos["pct_nav"].tolist(), "by_position")

    # By Asset Class
    by_class = df_snap.groupby("category", as_index=False)["pct_nav"].sum().sort_values("pct_nav", ascending=False)
    _save_pie(by_class["category"].tolist(), by_class["pct_nav"].tolist(), "by_class")

    # By Theme — cash themes first
    _bt = df_snap.groupby("theme", as_index=False)["pct_nav"].sum()
    _cash_t = {t for t in _bt["theme"]
               if df_snap[df_snap["theme"] == t]["symbol"].str.startswith("CASH_").all()}
    by_theme = pd.concat([
        _bt[_bt["theme"].isin(_cash_t)],
        _bt[~_bt["theme"].isin(_cash_t)].sort_values("pct_nav", ascending=False),
    ], ignore_index=True)
    _save_pie(by_theme["theme"].tolist(), by_theme["pct_nav"].tolist(), "by_theme")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Weightings
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
    fig = px.pie(df, names=names, values=values, hole=0.35,
                 template="capital", color_discrete_sequence=_BLUES_PALETTE)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      sort=False, rotation=0, direction="clockwise")
    fig.update_layout(margin=dict(t=24, b=24, l=24, r=24))
    return fig


tab_w_pos, tab_w_class, tab_w_theme = st.tabs(["By Position", "By Asset Class", "By Theme"])

with tab_w_pos:
    fig = _pie(_sort_pos_cash_first(df_snap), "symbol", "pct_nav")
    st.plotly_chart(fig, width="stretch")

with tab_w_class:
    by_class = df_snap.groupby("category", as_index=False)["pct_nav"].sum()
    fig = _pie(by_class, "category", "pct_nav")
    st.plotly_chart(fig, width="stretch")

with tab_w_theme:
    _bt = df_snap.groupby("theme", as_index=False)["pct_nav"].sum()
    _cash_t = {t for t in _bt["theme"]
               if df_snap[df_snap["theme"] == t]["symbol"].str.startswith("CASH_").all()}
    by_theme = pd.concat([
        _bt[_bt["theme"].isin(_cash_t)],
        _bt[~_bt["theme"].isin(_cash_t)].sort_values("pct_nav", ascending=False),
    ], ignore_index=True)
    fig = _pie(by_theme, "theme", "pct_nav")
    st.plotly_chart(fig, width="stretch")

st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
clicked_pies = st.button("Export Figures", key="export_pies_mpl")
st.markdown('</div>', unsafe_allow_html=True)
if clicked_pies:
    try:
        paths = _export_pies_matplotlib(df_snap)
        for p in paths:
            st.success(f"Saved: {p}")
    except Exception as exc:
        st.error(f"Export failed: {exc}")

st.subheader(f"Holdings — {snap_date.strftime('%d %b %Y')}")
st.html(_basket_html(df_snap))


# ─────────────────────────────────────────────────────────────────────────────
# Trade Log
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
    disp["quantity"]         = disp["quantity"].map("{:,.4f}".format)
    disp["entry_exit_price"] = disp["entry_exit_price"].map("{:,.4f}".format)
    disp["effective_price"]  = disp["effective_price"].map("{:,.4f}".format)
    disp["commission"]       = disp["commission"].map("{:,.2f}".format)
    disp["asset_type"]       = disp["asset_type"].map({"COMMON": "Stock", "ETF": "ETF", "FX": "FX"})
    st.dataframe(
        disp.rename(columns={
            "trade_date": "Date", "symbol": "Symbol", "name": "Name", "isin": "ISIN",
            "currency": "CCY", "asset_type": "Type", "buy_sell": "Side",
            "quantity": "Qty", "entry_exit_price": "Entry/Exit Price",
            "effective_price": "Effective Price", "commission": "Commission",
        }),
        width="stretch", hide_index=True,
        column_order=["Date", "Type", "Symbol", "ISIN", "Name", "CCY",
                      "Side", "Qty", "Entry/Exit Price", "Effective Price", "Commission"],
    )
