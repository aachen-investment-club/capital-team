"""Performance, returns vs benchmarks, weightings, holdings, trade log, analytics.

Reference page for the loaders/figures pattern: all data via capital.data.loaders,
callbacks return figures only, matplotlib report exports via dcc.Download.
"""
import io
import zipfile
from datetime import date

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc

from capital.analytics.analysis import compute_metrics, legend_html, metrics_html
from capital.analytics import quarterly_report as qr
from capital.dashboard import components as ui
from capital.data import loaders

dash.register_page(
    __name__, path="/performance", name="Performance", order=1,
    description="Portfolio returns vs benchmarks, weightings, holdings and trade log.",
)

_DATE_MIN = date(2026, 5, 6)   # fund inception, first date in nav_history

DISPLAY_NAMES: dict[str, str] = {
    "PORTFOLIO":   "AIC Portfolio",
    "60_40":       "60/40 Balanced",
    "SPX":         "S&P 500",
    "MSCI_WORLD":  "MSCI World",
    "MSCI_EUROPE": "MSCI Europe",
}

# Light-on-dark colours for the transparent matplotlib export (navy background).
_RETURNS_COLORS_LIGHT: dict[str, str] = {
    "PORTFOLIO":   "#93C5FD",
    "SPX":         "#9CA3AF",
    "MSCI_WORLD":  "#60A5FA",
    "MSCI_EUROPE": "#6B7280",
    "60_40":       "#CBD5E1",
}

# Dark → light blue-gray gradient
_BLUES_PALETTE = [
    "#0C2A4A", "#1A3F6F", "#2E5E9E", "#4279BC", "#5A94D0", "#74ADE0",
    "#8FBFEC", "#8090A8", "#94A3B8", "#A8B8CC", "#BDC9D8", "#D1DCE5", "#E2E8F0",
]


def _palette_sample(palette: list[str], n: int) -> list[str]:
    """Pick n colours equidistantly from palette so any subset spans the full range."""
    if n <= 1:
        return [palette[0]]
    return [palette[round(i * (len(palette) - 1) / (n - 1))] for i in range(n)]


def _yesterday() -> date:
    return date.today() - pd.Timedelta(days=1)


# ── Holdings table (grouped by theme/category basket) ────────────────────────

def _basket_html(df: pd.DataFrame) -> str:
    NAVY, BLUE_LIGHT = "#0C1E40", "#EFF6FF"
    BORDER, TEXT, TEXT_MUTED = "#E2E8F0", "#0F172A", "#64748B"
    GREEN, RED = "#10B981", "#EF4444"

    rows_html = ""
    all_themes = sorted(df["theme"].dropna().unique())
    cash_themes = [t for t in all_themes
                   if df[df["theme"] == t]["symbol"].str.startswith("CASH_").all()]
    other_themes = [t for t in all_themes if t not in set(cash_themes)]
    for theme in other_themes + cash_themes:
        group = df[df["theme"] == theme].sort_values("pct_nav", ascending=False)
        basket_nav = group["pct_nav"].sum()
        basket_cr = ((group["pct_nav"] * group["cumulative_return"]).sum() / basket_nav
                     if basket_nav > 0 else 0.0)
        basket_dr = ((group["pct_nav"] * group["daily_return"]).sum() / basket_nav
                     if basket_nav > 0 else 0.0)
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
            pos_dr, pos_cr = pos["daily_return"], pos["cumulative_return"]
            pos_dr_col = GREEN if pos_dr >= 0 else RED
            pos_cr_col = GREEN if pos_cr >= 0 else RED
            dr_str = ": " if is_base_ccy else f"{pos_dr:+.2%}"
            cr_str = ": " if is_base_ccy else f"{pos_cr:+.2%}"
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


# ── Matplotlib report exports (transparent PNGs, navy-deck styling) ──────────

def _returns_figures(df_perf: pd.DataFrame) -> dict[str, bytes]:
    """Transparent-background PNGs for the Returns vs Benchmark charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    figures: dict[str, bytes] = {}
    _all = df_perf["ticker"].unique()
    tickers = (["PORTFOLIO"] if "PORTFOLIO" in _all else []) + sorted(
        t for t in _all if t != "PORTFOLIO")

    F_LABEL, F_TICK, F_LEGEND = 18, 15, 15
    OFFWHITE = "#CBD5E1"
    GRID_C = (1.0, 1.0, 1.0, 0.08)
    REF_C = "#93C5FD"
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
        leg = ax.legend(fontsize=F_LEGEND, framealpha=0.25, facecolor="#0C1E40",
                        edgecolor="none", labelcolor=OFFWHITE)
        for line in leg.get_lines():
            line.set_linewidth(3)

    def _save(fig, name: str) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        figures[name] = buf.getvalue()

    # 1. Index, line chart
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
    _save(fig, "returns_index.png")

    # 2. Daily returns, grouped bar chart
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
    ax.set_xticklabels([pd.Timestamp(dates[j]).strftime("%d %b") for j in x_all[::step]],
                       rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    _style(fig, ax)
    ax.set_xlabel("Date", fontsize=F_LABEL, labelpad=10)
    ax.set_ylabel("Daily Return", fontsize=F_LABEL, labelpad=10)
    _legend(ax)
    fig.tight_layout()
    _save(fig, "returns_bar.png")

    return figures


_CASH_CCY_RANK = {"EUR": 0, "USD": 1, "GBP": 2}


def _sort_pos_cash_first(df: pd.DataFrame) -> pd.DataFrame:
    """CASH_* rows first (EUR→USD→GBP→rest), then equities by pct_nav desc."""
    is_cash = df["symbol"].str.startswith("CASH_")
    cash = df[is_cash].copy()
    cash["_r"] = cash["symbol"].str.replace("CASH_", "", regex=False).map(
        lambda c: _CASH_CCY_RANK.get(c, 99))
    cash = cash.sort_values("_r").drop(columns="_r")
    equity = df[~is_cash].sort_values("pct_nav", ascending=False)
    return pd.concat([cash, equity], ignore_index=True)


def _theme_groups(df_snap: pd.DataFrame) -> pd.DataFrame:
    """% NAV grouped by theme, cash themes first."""
    _bt = df_snap.groupby("theme", as_index=False)["pct_nav"].sum()
    _cash_t = {t for t in _bt["theme"]
               if df_snap[df_snap["theme"] == t]["symbol"].str.startswith("CASH_").all()}
    return pd.concat([
        _bt[_bt["theme"].isin(_cash_t)],
        _bt[~_bt["theme"].isin(_cash_t)].sort_values("pct_nav", ascending=False),
    ], ignore_index=True)


def _pies_figures(df_snap: pd.DataFrame) -> dict[str, bytes]:
    """Transparent-background PNGs for the weighting pies."""
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, bytes] = {}
    _LEADER_MIN = 1.0

    def _save_pie(labels, values, suffix):
        total = sum(values)
        pcts = [v / total * 100 for v in values]
        colors = [_BLUES_PALETTE[i % len(_BLUES_PALETTE)] for i in range(len(labels))]
        _HOLE = 0.4
        _R_TEXT = (_HOLE + 1.0) / 2

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_alpha(0)
        fig.set_facecolor((0, 0, 0, 0))
        ax.set_facecolor((0, 0, 0, 0))
        wedges, _ = ax.pie(
            values, labels=None, colors=colors, autopct=None,
            startangle=0, counterclock=False,
            wedgeprops=dict(width=1 - _HOLE, linewidth=0.5, edgecolor=(1, 1, 1, 0.12)),
        )
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        _RIM_INSET = 0.02
        for wedge, label, pct in zip(wedges, labels, pcts):
            if pct < _LEADER_MIN:
                continue
            mid_deg = (wedge.theta1 + wedge.theta2) / 2
            mid_rad = math.radians(mid_deg)
            small = pct < 5.0
            r = (1.0 - _RIM_INSET) if small else _R_TEXT
            x, y = r * math.cos(mid_rad), r * math.sin(mid_rad)
            fs = 9 if small else max(9, min(13, int(pct ** 0.5 * 3.5)))
            text = f"{label}\n{pct:.1f}%" if not small else f"{label}  {pct:.1f}%"
            if small:
                rot = mid_deg % 360
                flipped = 90 < rot < 270
                if flipped:
                    rot -= 180
                ha = "left" if flipped else "right"
            else:
                rot, ha = 0, "center"
            ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color="white",
                    fontweight="600" if pct >= 5 else "normal",
                    rotation=rot, rotation_mode="anchor", linespacing=1.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        figures[f"weights_{suffix}.png"] = buf.getvalue()

    df_pos = _sort_pos_cash_first(df_snap)
    _save_pie(df_pos["symbol"].tolist(), df_pos["pct_nav"].tolist(), "by_position")

    by_class = df_snap.groupby("category", as_index=False)["pct_nav"].sum() \
        .sort_values("pct_nav", ascending=False)
    _save_pie(by_class["category"].tolist(), by_class["pct_nav"].tolist(), "by_class")

    by_theme = _theme_groups(df_snap)
    _save_pie(by_theme["theme"].tolist(), by_theme["pct_nav"].tolist(), "by_theme")

    return figures


# ── Section 5: Returns & Attribution (formatted HTML page) ───────────────────
# Standalone, styled report section — Returns table (QTD/YTD/Trailing 12M/
# Since Inception %) and Attribution table (selection vs. allocation by
# theme). Built in isolation from the rest of the report's section numbering;
# the portfolio's QTD row is the figure meant to feed the report's future
# Executive Summary return line, flagged below rather than wired up yet.

_RETURNS_COL_LABELS: dict[str, str] = {
    "portfolio_return_pct": DISPLAY_NAMES["PORTFOLIO"],
    **{qr.BENCHMARK_EXPORT_COLS[t]: DISPLAY_NAMES[t] for t in qr.BENCHMARK_TICKERS},
}

_REPORT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body{margin:0;padding:32px;font-family:'Inter',system-ui,sans-serif;color:#0F172A;background:#fff;}
h1{font-size:22px;color:#0C1E40;margin:0 0 4px;}
h2{font-size:16px;color:#0C1E40;margin:28px 0 4px;}
.sub{color:#64748B;font-size:13px;margin:0 0 20px;}
.note{color:#64748B;font-size:12px;line-height:1.5;margin:8px 0 20px;padding:10px 14px;
      background:#F8FAFC;border-left:3px solid #CBD5E1;border-radius:4px;}
table{width:100%;border-collapse:collapse;margin-bottom:8px;}
th{padding:8px 14px;text-align:right;color:#64748B;font-size:11px;font-weight:500;
   text-transform:uppercase;letter-spacing:.05em;border-bottom:2px solid #E2E8F0;background:#F8FAFC;}
th:first-child,td:first-child{text-align:left;}
td{padding:7px 14px;text-align:right;font-size:13px;border-bottom:1px solid #E2E8F0;}
tr.total td{font-weight:700;border-top:2px solid #E2E8F0;border-bottom:none;background:#EFF6FF;}
.na{color:#94A3B8;}
.pos{color:#10B981;font-weight:600;}
.neg{color:#EF4444;font-weight:600;}
</style>"""


def _sign_class(v: float) -> str:
    return "pos" if v > 0 else "neg" if v < 0 else ""


def _fmt_pct_cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '<span class="na">N/A</span>'
    return f'<span class="{_sign_class(v)}">{v:+.2f}%</span>'


def _fmt_bps_cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '<span class="na">N/A</span>'
    return f'<span class="{_sign_class(v)}">{v:+.1f}</span>'


def _returns_table_html(df: pd.DataFrame) -> str:
    headers = ["Period"] + [_RETURNS_COL_LABELS.get(c, c) for c in df.columns]
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for period, row in df.iterrows():
        cells = "".join(f"<td>{_fmt_pct_cell(row[c])}</td>" for c in df.columns)
        body += f"<tr><td>{period}</td>{cells}</tr>"
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _attribution_table_html(df_bm: pd.DataFrame, benchmark_label: str) -> str:
    rows = df_bm.sort_values("portfolio_weight_pct", ascending=False)
    body = ""
    for _, r in rows.iterrows():
        body += (
            f'<tr><td>{r["theme"]}</td>'
            f'<td>{r["portfolio_weight_pct"]:.2f}%</td>'
            f'<td>{r["benchmark_weight_pct"]:.2f}%</td>'
            f'<td>{_fmt_bps_cell(r["selection_effect_bps"])}</td>'
            f'<td>{_fmt_bps_cell(r["allocation_effect_bps"])}</td>'
            f'<td>{_fmt_bps_cell(r["total_effect_bps"])}</td></tr>'
        )
    totals = rows[["portfolio_weight_pct", "benchmark_weight_pct", "selection_effect_bps",
                   "allocation_effect_bps", "total_effect_bps"]].sum()
    body += (
        f'<tr class="total"><td>Total</td>'
        f'<td>{totals["portfolio_weight_pct"]:.2f}%</td>'
        f'<td>{totals["benchmark_weight_pct"]:.2f}%</td>'
        f'<td>{_fmt_bps_cell(totals["selection_effect_bps"])}</td>'
        f'<td>{_fmt_bps_cell(totals["allocation_effect_bps"])}</td>'
        f'<td>{_fmt_bps_cell(totals["total_effect_bps"])}</td></tr>'
    )
    return (
        f'<h2>vs. {benchmark_label}</h2>'
        f'<table><thead><tr><th>Theme</th><th>Portfolio Wt</th><th>Benchmark Wt</th>'
        f'<th>Selection (bps)</th><th>Allocation (bps)</th><th>Total (bps)</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
    )


def _section5_html(start, end, benchmarks: list[str]) -> str:
    """Section 5 — Returns table + Attribution table, as a standalone styled
    HTML page (built in isolation; not yet wired into a larger report/
    Executive Summary structure — see the note below).
    """
    returns_df = qr.returns_summary_table(end, tuple(benchmarks))
    attr_df = qr.theme_attribution(start, end, tuple(benchmarks))

    qtd_port = returns_df.loc["QTD", "portfolio_return_pct"] if "QTD" in returns_df.index else None
    qtd_str = f"{qtd_port:+.2f}%" if qtd_port is not None and not pd.isna(qtd_port) else "N/A"
    exec_summary_note = f"Executive Summary hook: this section's QTD portfolio return ({qtd_str})"

    attr_sections = "".join(
        _attribution_table_html(attr_df[attr_df["benchmark"] == bm], DISPLAY_NAMES.get(bm, bm))
        for bm in benchmarks if not attr_df[attr_df["benchmark"] == bm].empty
    )

    return f"""<!doctype html><html><head><meta charset="utf-8">{_REPORT_CSS}</head><body>
<h1>Section 5 — Returns &amp; Attribution</h1>
<p class="sub">Period: {start} to {end}</p>

<h2>Returns (QTD / YTD / Trailing 12M / Since Inception)</h2>
{_returns_table_html(returns_df)}
<div class="note">
{exec_summary_note}. This section is built standalone — it is not yet wired into
an Executive Summary or numbered against the report's other sections.
</div>

<h1 style="margin-top:32px;">Attribution by Theme</h1>
<div class="note">
Simplified Brinson-Fachler: the tracked benchmarks have no native per-theme
weight or return, so benchmark_weight = portfolio_weight and each theme's
benchmark return = the benchmark's total return. Both assumptions zero
Allocation by construction — all measurable excess return per theme shows as
Selection. A cash-heavy theme's Selection can also read as distorted if a
currency balance was swept to zero and later refilled during the period (a
cash-management event, not an investment loss) — cross-check against
theme_position_snapshot.csv in this export before reading it at face value.
</div>
{attr_sections}
</body></html>"""


# ── Full report export (charts + all CSVs, bundled) ──────────────────────────

def _metrics_csv(start, end) -> bytes | None:
    """Risk/return metrics for the portfolio over [start, end], vs S&P 500.
    None if there are too few trading days in range to compute them.
    """
    df_all = loaders.get_portfolio_and_benchmarks()
    df_all = df_all[df_all["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    port_rets = (df_all[df_all["ticker"] == "PORTFOLIO"]
                 .sort_values("date").set_index("date")["daily_return"].dropna())
    if len(port_rets) < 5:
        return None
    spx_rets = (df_all[df_all["ticker"] == "SPX"]
                .sort_values("date").set_index("date")["daily_return"].dropna())
    bench = spx_rets if not spx_rets.empty else None
    metrics = compute_metrics(port_rets, bench_daily_returns=bench)
    out = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    return out.to_csv(index=False).encode()


def _report_readme(start, end, snap_date, bm_list: list[str]) -> str:
    bm_names = ", ".join(DISPLAY_NAMES.get(b, b) for b in bm_list)
    return f"""AIC Portfolio — Performance Report Export
Generated from the dashboard's Performance page.

Period                : {start} to {end}
Holdings/theme snapshot as-of : {snap_date}
Benchmarks             : {bm_names}

FILES
-----
returns_index.png, returns_bar.png
    Index & daily-return charts for the selected period and benchmarks.

weights_by_position.png, weights_by_class.png, weights_by_theme.png
    Portfolio weight pies, as of the snapshot date.

daily_returns.csv
    One row per trading day, portfolio inception -> {end}.
    portfolio_return_pct is a TIME-WEIGHTED daily return: it tracks the
    fund's unit price (nav_history.fund_nav), which moves only with
    investment performance — deposits/withdrawals buy or redeem units at
    the prior day's unit price and never affect it. Safe to chain for
    QTD/YTD/trailing-12M/since-inception returns over any date range.

aum_and_flows.csv
    One row per day, portfolio inception -> {end}. aum_eur is end-of-day
    fund NAV; external_flow_eur is net deposits/withdrawals that day.
    Reconciliation: this fund's unit-price accounting applies a flow at
    the PRIOR day's unit price, so new capital participates in that
    day's return. The exact identity is therefore
        aum(t) = (aum(t-1) + external_flow_eur(t)) * (1 + portfolio_return_pct(t)/100)
    rather than aum(t-1)*(1+r) + flow(t) applied strictly after the
    day's return — the two differ only by flow(t) x return(t), which is
    negligible for realistic flow sizes.

theme_position_snapshot.csv
    One row per held position as of {snap_date}, grouped by theme.
    period_return_pct is compounded from {start}; cumulative_return_pct
    is since the position's own inception (cost basis).

theme_attribution.csv
    Brinson-Fachler attribution by theme, {start} -> {end}.
    SIMPLIFICATION: the tracked benchmarks have no native breakdown by
    this portfolio's themes, so benchmark_weight_pct is assumed equal
    to portfolio_weight_pct and each theme's benchmark return is
    assumed equal to the benchmark's total return. Both assumptions
    force allocation_effect_bps to 0 by construction (there's no data
    to measure it from); all measurable excess return per theme is
    reported as selection_effect_bps. Labelled approximation, not a
    true per-theme Brinson-Fachler split.
    CAVEAT: a cash-heavy theme's selection_effect_bps can be distorted
    if a currency balance was fully swept to zero and later refilled
    during the period (reads as a -100% day for that cash symbol, a
    pre-existing quirk of the underlying position data, not a real
    investment loss) — cross-check against theme_position_snapshot.csv.

trades.csv
    Trade log, {start} -> {end}.

metrics.csv
    Risk/return metrics (Sharpe, Sortino, drawdown, VaR, etc.) for the
    portfolio over {start} -> {end}, vs S&P 500 where applicable.
    Omitted if fewer than 5 trading days fall in range.

section5_returns_and_attribution.html
    Formatted "Section 5" report page: a Returns table (QTD / YTD /
    Trailing 12M / Since Inception, portfolio + each benchmark) and an
    Attribution table (selection vs. allocation by theme) per
    benchmark, both computed as of {end}. Built standalone — not yet
    wired into a larger numbered report or an Executive Summary; the
    file notes which figure (QTD portfolio return) is meant to feed
    that Executive Summary's return line once it exists. Same
    methodology and caveats as daily_returns.csv / theme_attribution.csv.
"""


# ── Layout ────────────────────────────────────────────────────────────────────

def _analytics_block():
    try:
        df_all = loaders.get_portfolio_and_benchmarks()
        port_rets = (df_all[df_all["ticker"] == "PORTFOLIO"]
                     .sort_values("date").set_index("date")["daily_return"].dropna())
        spx_rets = (df_all[df_all["ticker"] == "SPX"]
                    .sort_values("date").set_index("date")["daily_return"].dropna())
        if len(port_rets) < 5:
            return dmc.Alert("Not enough portfolio history to compute analytics yet (need ≥ 5 days).",
                             color="blue", variant="light")
        bench = spx_rets if not spx_rets.empty else None
        metrics = compute_metrics(port_rets, bench_daily_returns=bench)
        return dmc.Stack([
            ui.raw_html(metrics_html(metrics), height=340),
            dmc.Accordion([dmc.AccordionItem(
                [dmc.AccordionControl("Colour-scale legend"),
                 dmc.AccordionPanel(ui.raw_html(legend_html(), height=420))],
                value="legend")]),
        ])
    except Exception as e:
        return dmc.Alert(f"Analytics unavailable: {e}", color="blue", variant="light")


def layout():
    today = _yesterday()
    try:
        df_bench = loaders.get_portfolio_and_benchmarks()
        available_bm = sorted(t for t in df_bench["ticker"].unique() if t != "PORTFOLIO")
        bench_ok = not df_bench.empty
    except Exception:
        available_bm, bench_ok = [], False

    if not bench_ok:
        returns_section = dmc.Alert(
            "Benchmark data not yet available: run the ingest job when S3 data is present.",
            color="blue", variant="light")
    else:
        returns_section = dmc.Stack([
            dmc.Group([
                dmc.DatePickerInput(id="perf-start", label="From", value=_DATE_MIN.isoformat(),
                                    minDate=_DATE_MIN.isoformat(), maxDate=today.isoformat(), w=160),
                dmc.DatePickerInput(id="perf-end", label="To", value=today.isoformat(),
                                    minDate=_DATE_MIN.isoformat(), maxDate=today.isoformat(), w=160),
                dmc.MultiSelect(id="perf-benchmarks", label="Benchmarks",
                                data=[{"value": t, "label": DISPLAY_NAMES.get(t, t)}
                                      for t in available_bm],
                                value=[], w=340, clearable=True),
            ], align="end"),
            dmc.Tabs([
                dmc.TabsList([dmc.TabsTab("Index", value="index"),
                              dmc.TabsTab("Returns", value="returns")]),
                dmc.TabsPanel(ui.graph("perf-index-chart"), value="index"),
                dmc.TabsPanel(ui.graph("perf-returns-chart"), value="returns"),
            ], value="index"),
        ])

    return dmc.Stack([
        dmc.Group([
            ui.page_title(
                "Performance",
                "Shows how the portfolio has performed over time, what each position "
                "contributed to returns, how weights have shifted across baskets, and "
                "a full log of every trade made."),
            ui.export_button("perf-export-all", "Export Performance Report"),
        ], justify="space-between", align="flex-start"),

        ui.section("Returns vs Benchmark"),
        returns_section,

        dmc.Divider(mt="lg"),
        ui.section("Portfolio Weightings"),
        dmc.DatePickerInput(id="weight-date", label="Date", value=today.isoformat(),
                            minDate=_DATE_MIN.isoformat(), maxDate=today.isoformat(), w=160),
        dmc.Tabs([
            dmc.TabsList([dmc.TabsTab("By Position", value="pos"),
                          dmc.TabsTab("By Asset Class", value="class"),
                          dmc.TabsTab("By Theme", value="theme")]),
            dmc.TabsPanel(ui.graph("weight-pie-pos"), value="pos"),
            dmc.TabsPanel(ui.graph("weight-pie-class"), value="class"),
            dmc.TabsPanel(ui.graph("weight-pie-theme"), value="theme"),
        ], value="pos"),
        dmc.Title(id="holdings-title", order=3, c="#0C1E40", mt="lg", mb="sm"),
        dcc.Loading(dmc.Box(id="holdings-table")),

        dmc.Divider(mt="lg"),
        ui.section("Trade Log"),
        dmc.Select(id="trade-month", label="Month", w=220, data=[], searchable=False),
        dcc.Loading(dmc.Box(id="trade-table", mt="sm")),

        dmc.Divider(mt="lg"),
        ui.section("Portfolio Analytics"),
        _analytics_block(),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

def _perf_frame(start, end, benchmarks) -> pd.DataFrame:
    df_all = loaders.get_portfolio_and_benchmarks()
    tickers = ["PORTFOLIO"] + (benchmarks or [])
    return (df_all[df_all["date"].between(pd.Timestamp(start), pd.Timestamp(end))
                   & df_all["ticker"].isin(tickers)]
            .copy().sort_values(["ticker", "date"]))


@callback(
    Output("perf-index-chart", "figure"),
    Output("perf-returns-chart", "figure"),
    Input("perf-start", "value"),
    Input("perf-end", "value"),
    Input("perf-benchmarks", "value"),
)
def update_returns(start, end, benchmarks):
    df_perf = _perf_frame(start, end, benchmarks)
    df_plot = df_perf.copy()
    df_plot["ticker"] = df_plot["ticker"].map(lambda t: DISPLAY_NAMES.get(t, t))
    portfolio_label = DISPLAY_NAMES["PORTFOLIO"]
    ticker_order = [portfolio_label] + sorted(
        t for t in df_plot["ticker"].unique() if t != portfolio_label)
    colors = dict(zip(ticker_order, _palette_sample(_BLUES_PALETTE, len(ticker_order))))

    fig_idx = px.line(df_plot, x="date", y="index_value", color="ticker",
                      color_discrete_map=colors, category_orders={"ticker": ticker_order},
                      labels={"index_value": "Index", "date": "Date", "ticker": ""})
    fig_idx.add_hline(y=1.0, line_dash="dot", line_color="#BFDBFE")
    fig_idx.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    fig_ret = px.bar(df_plot, x="date", y="daily_return", color="ticker", barmode="group",
                     color_discrete_map=colors, category_orders={"ticker": ticker_order},
                     labels={"daily_return": "Daily Return", "date": "Date", "ticker": ""})
    fig_ret.update_layout(yaxis_tickformat=".1%")
    fig_ret.add_hline(y=0, line_dash="dot", line_color="#BFDBFE")
    fig_ret.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig_idx, fig_ret


def _snapshot(weight_date) -> tuple[pd.DataFrame, date]:
    df_wh = loaders.get_daily_weightings_with_themes()
    avail = df_wh["date"].dt.date.unique()
    target = pd.Timestamp(weight_date).date()
    valid = sorted(d for d in avail if d <= target)
    snap_date = valid[-1] if valid else sorted(avail)[-1]
    return df_wh[df_wh["date"].dt.date == snap_date].copy(), snap_date


def _pie(df: pd.DataFrame, names: str, values: str):
    fig = px.pie(df, names=names, values=values, hole=0.35,
                 color_discrete_sequence=_BLUES_PALETTE)
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      sort=False, rotation=0, direction="clockwise")
    fig.update_layout(margin=dict(t=24, b=24, l=24, r=24))
    return fig


@callback(
    Output("weight-pie-pos", "figure"),
    Output("weight-pie-class", "figure"),
    Output("weight-pie-theme", "figure"),
    Output("holdings-title", "children"),
    Output("holdings-table", "children"),
    Input("weight-date", "value"),
)
def update_weightings(weight_date):
    df_snap, snap_date = _snapshot(weight_date)
    by_class = df_snap.groupby("category", as_index=False)["pct_nav"].sum()
    holdings = ui.raw_html(_basket_html(df_snap), height=42 * len(df_snap) + 120)
    return (_pie(_sort_pos_cash_first(df_snap), "symbol", "pct_nav"),
            _pie(by_class, "category", "pct_nav"),
            _pie(_theme_groups(df_snap), "theme", "pct_nav"),
            f"Holdings, {snap_date.strftime('%d %b %Y')}",
            holdings)


@callback(
    Output("perf-export-all-download", "data"),
    Input("perf-export-all", "n_clicks"),
    Input("perf-start", "value"),
    Input("perf-end", "value"),
    Input("perf-benchmarks", "value"),
    Input("weight-date", "value"),
    prevent_initial_call=True,
)
def export_all(n_clicks, start, end, benchmarks, weight_date):
    """One button, one ZIP: every chart and table already on this page, plus
    the quarterly-report CSVs (TWR daily returns, AUM & flows, theme/position
    snapshot, theme attribution) — all parameterised by the page's own date
    range and benchmark/snapshot pickers, so it works for any period.
    """
    if dash.ctx.triggered_id != "perf-export-all" or not n_clicks:
        return dash.no_update

    df_perf = _perf_frame(start, end, benchmarks)
    df_snap, snap_date = _snapshot(weight_date)
    bm_list = benchmarks or sorted(
        t for t in loaders.get_portfolio_and_benchmarks()["ticker"].unique() if t != "PORTFOLIO")

    files: dict[str, bytes] = {}
    files.update(_returns_figures(df_perf))
    files.update(_pies_figures(df_snap))

    trades = loaders.get_trade_log()
    trades_period = trades[trades["trade_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    files["trades.csv"] = trades_period.to_csv(index=False).encode()

    metrics_csv = _metrics_csv(start, end)
    if metrics_csv is not None:
        files["metrics.csv"] = metrics_csv

    files["daily_returns.csv"] = qr.daily_returns_timeseries(end).to_csv(index=False).encode()
    files["aum_and_flows.csv"] = qr.aum_and_flows_timeseries(end).to_csv(index=False).encode()
    files["theme_position_snapshot.csv"] = (
        qr.theme_position_snapshot(snap_date, start).to_csv(index=False).encode())
    files["theme_attribution.csv"] = (
        qr.theme_attribution(start, end, tuple(bm_list)).to_csv(index=False).encode())
    files["section5_returns_and_attribution.html"] = (
        _section5_html(start, end, bm_list).encode())

    files["README.txt"] = _report_readme(start, end, snap_date, bm_list).encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    fname = f"performance_report_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}.zip"
    return dcc.send_bytes(lambda f, d=buf.getvalue(): f.write(d), fname)


@callback(
    Output("trade-month", "data"),
    Output("trade-month", "value"),
    Input("shell-url", "pathname"),
)
def init_trade_months(pathname):
    df = loaders.get_trade_log()
    periods = sorted(df["trade_date"].dt.to_period("M").unique(), reverse=True)
    labels = [p.strftime("%B %Y") for p in periods]
    default = pd.Timestamp(_yesterday()).to_period("M").strftime("%B %Y")
    value = default if default in labels else (labels[0] if labels else None)
    return labels, value


@callback(Output("trade-table", "children"), Input("trade-month", "value"))
def update_trades(month_label):
    if not month_label:
        return dmc.Alert("No trades recorded.", color="blue", variant="light")
    df = loaders.get_trade_log()
    period = pd.Period(pd.to_datetime(month_label, format="%B %Y"), freq="M")
    df_month = df[df["trade_date"].dt.to_period("M") == period].copy()
    if df_month.empty:
        return dmc.Alert("No trades recorded for this month.", color="blue", variant="light")

    disp = df_month.copy()
    disp["trade_date"] = disp["trade_date"].dt.strftime("%Y-%m-%d")
    disp["quantity"] = disp["quantity"].map("{:,.4f}".format)
    disp["entry_exit_price"] = disp["entry_exit_price"].map("{:,.4f}".format)
    disp["effective_price"] = disp["effective_price"].map("{:,.4f}".format)
    disp["commission"] = disp["commission"].map("{:,.2f}".format)
    disp["asset_type"] = disp["asset_type"].map({"COMMON": "Stock", "ETF": "ETF", "FX": "FX"})
    disp = disp.rename(columns={
        "trade_date": "Date", "symbol": "Symbol", "name": "Name", "isin": "ISIN",
        "currency": "CCY", "asset_type": "Type", "buy_sell": "Side",
        "quantity": "Qty", "entry_exit_price": "Entry/Exit Price",
        "effective_price": "Effective Price", "commission": "Commission",
    })[["Date", "Type", "Symbol", "ISIN", "Name", "CCY",
        "Side", "Qty", "Entry/Exit Price", "Effective Price", "Commission"]]
    return ui.df_table(disp)
