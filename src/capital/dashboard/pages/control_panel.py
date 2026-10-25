"""Control panel: one-screen overview of market regime, volatility, strategy
leadership, the portfolio's outlook, movers and risk, with alerts.

One box per theme, chart-first: each box leads with a chart and keeps its
numbers to a compact stat strip, and its explanation sits behind an ℹ icon in
the header rather than taking vertical space.

Every box reads through version-keyed caches, so the HMM and GARCH fits run once
per data version (after the nightly ingest) and every later page load is a cache
hit. Each box has its own callback: one failing input (say, S3 being down for the
position weights) costs that box, not the page.
"""
import traceback

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from capital.analytics import control_panel as cp
from capital.analytics import regime as rg
from capital.analytics.factors import regression as reg
from capital.analytics.factors import risk as riskmod
from capital.analytics.factors.risk import periods_per_year
from capital.analytics.factors.spec import STYLES
from capital.analytics.volatility import log_returns_pct
from capital.dashboard import components as ui
from capital.data import factor_store as fstore
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import BARRA_NEG, BARRA_POS, BARRA_SCALE, GRAPH_CONFIG, NAVY, TEXT_MUTED

dash.register_page(
    __name__, path="/control-panel", name="Control Panel", order=0.5,
    description="Market regime (HMM), GARCH volatility, leading strategies and "
                "sectors, portfolio outlook, movers, risk and alerts on one screen.",
)

POS, NEG = "#2E7D6B", "#8B2E4A"
GOLD, SAGE, SLATE, STEEL = "#B8962E", "#2E7D6B", "#94A3B8", "#3A6B9C"
#: Chart colours for the six cross-asset lines, fixed per ticker so a colour
#: always means the same asset.
ASSET_COLORS = {"SPY": NAVY, "QQQ": STEEL, "IWM": GOLD, "TLT": SAGE, "HYG": "#A0522D", "GLD": "#7B6FA0"}
W = cp.WINDOW
H = 170  # default chart height inside a box, so boxes in one row line up


# ── Formatting ────────────────────────────────────────────────────────────────

def _pct(x, digits: int = 1, sign: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def _num(x, fmt: str = "{:.2f}") -> str:
    return "–" if x is None or not np.isfinite(x) else fmt.format(x)


def _tone(x) -> str:
    if x is None or not np.isfinite(x) or x == 0:
        return NAVY
    return POS if x > 0 else NEG


def _date(ts) -> str:
    return "–" if ts is None or pd.isna(ts) else f"{pd.Timestamp(ts):%d %b %Y}"


def _stat(label: str, value: str, hint: str = "", color: str = NAVY) -> html.Div:
    """A small stat: muted label, bold value, optional hint. Much denser than
    ui.metric, so a box can carry eight of them in two short rows."""
    return html.Div([
        dmc.Text(label, size="9px", c=TEXT_MUTED, fw=600, tt="uppercase", lh=1.15),
        dmc.Text(value, size="sm", fw=700, c=color, lh=1.2),
        dmc.Text(hint, size="9px", c=TEXT_MUTED, lh=1.15) if hint else None,
    ], style={"minWidth": 0})


def _stats(items: list, cols: dict | int) -> dmc.SimpleGrid:
    return dmc.SimpleGrid([_stat(*i) for i in items], cols=cols, spacing=6, verticalSpacing=3)


def _info(md: str) -> dmc.HoverCard:
    """ℹ icon whose hover card holds the box's methodology (markdown + maths)."""
    return dmc.HoverCard([
        dmc.HoverCardTarget(dmc.ThemeIcon("ℹ", variant="light", radius="xl", size="sm",
                                          style={"cursor": "help"})),
        dmc.HoverCardDropdown(dcc.Markdown(md, mathjax=True, className="explain-body"),
                              maw=440, fz="sm"),
    ], position="bottom-end", shadow="md", openDelay=150, withinPortal=True)


def _box(title: str, body, subtitle: str = "", info: str | None = None, right=None) -> dmc.Paper:
    """One theme = one box: a single-line header (title, subtitle, controls, ℹ) over the body."""
    head = dmc.Group([
        dmc.Group([dmc.Text(title, fw=700, c=NAVY, size="xs"),
                   dmc.Text(subtitle, size="10px", c=TEXT_MUTED, truncate="end",
                            style={"minWidth": 0, "flex": 1})] if subtitle else
                  [dmc.Text(title, fw=700, c=NAVY, size="xs")],
                  gap=8, wrap="nowrap", style={"minWidth": 0, "flex": 1}),
        dmc.Group([x for x in (right, _info(info) if info else None) if x is not None],
                  gap=6, wrap="nowrap"),
    ], justify="space-between", wrap="nowrap", gap="xs")
    return dmc.Paper(dmc.Stack([head, body], gap=3), withBorder=True, radius="md", p=8, h="100%")


def _fail(title: str) -> dmc.Paper:
    return _box(title, dmc.Alert(dmc.Code(traceback.format_exc(), block=True),
                                 color="red", variant="light", title=f"{title} failed"))


def _graph(fig: go.Figure, height: int = H, legend: bool = False, left: int = 40,
           right: int = 10, bottom: int = 22) -> dcc.Graph:
    """A compact chart for a box. Size `left` with ui.axis_margin on categorical
    y axes so long labels are not clipped."""
    fig.update_layout(
        height=height, margin=dict(l=left, r=right, t=18 if legend else 4, b=bottom),
        showlegend=legend, font=dict(size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=9), borderwidth=0) if legend else None,
    )
    fig.update_xaxes(tickfont=dict(size=9), title_font=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9), title_font=dict(size=9))
    return dcc.Graph(figure=fig, config={**GRAPH_CONFIG, "displayModeBar": False},
                     style={"height": f"{height}px"})


def _hbar(values: pd.Series, labels, custom=None, hover: str = "%{y}: %{x:+.1%}<extra></extra>",
          fmt: str = "{:+.1%}", pos: str = POS, neg: str = NEG) -> go.Figure:
    """Sorted horizontal bar chart coloured by sign, values printed on the bars."""
    v = values.to_numpy(dtype=float)
    fig = go.Figure(go.Bar(
        x=v, y=list(labels), orientation="h", marker_color=[pos if x > 0 else neg for x in v],
        text=[fmt.format(x) if np.isfinite(x) else "" for x in v], textposition="outside",
        textfont=dict(size=9), cliponaxis=False,
        customdata=custom, hovertemplate=hover))
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(zeroline=True, zerolinecolor=SLATE, zerolinewidth=1)
    return fig


# ── Cached inputs ─────────────────────────────────────────────────────────────

@cached_by_version
def _spy_returns() -> pd.Series:
    return log_returns_pct(loaders.get_market_prices("SPY"))


@cached_by_version
def _regime() -> dict:
    """The HMM fit — ~10s, so it runs once per data version, never per page load."""
    return rg.fit_regime_model(_spy_returns())


@cached_by_version
def _nav() -> pd.Series:
    nav = loaders.get_nav_history()
    return nav.set_index("date")["fund_nav"].astype(float).dropna().sort_index()


@cached_by_version
def _bench_eur() -> tuple[pd.Series, str]:
    """SPY translated to EUR (the fund's currency), or USD when no FX is stored."""
    spy = loaders.get_market_prices("SPY")
    fx = loaders.get_fx_rates("EUR")
    if fx.empty or "USD" not in fx.columns:
        return spy, "USD"
    return cp.to_eur(spy, fx), "EUR"


@cached_by_version
def _vol(asset: str) -> dict:
    if asset == "Portfolio":
        r = _nav().pct_change().dropna() * 100
    else:
        r = _spy_returns()
    return cp.vol_perspective(r)


@cached_by_version
def _kpis() -> dict:
    bench, _ = _bench_eur()
    return cp.portfolio_kpis(_nav(), bench)


@cached_by_version
def _positions() -> pd.DataFrame:
    return cp.position_contributions(loaders.get_daily_weightings_history())


@cached_by_version
def _prices(tickers: tuple[str, ...]) -> dict[str, pd.Series]:
    return {tk: loaders.get_market_prices(tk) for tk in tickers}


def _run_id() -> str | None:
    return fstore.latest_run_id()


@cached_by_version
def _factor_view(run_id: str) -> dict:
    """Strategy returns, leaderboard, and the portfolio's exposures vs SPY for one run."""
    manifest = fstore.load_manifest(run_id) or {}
    freq = manifest.get("spec", {}).get("frequency", "W-FRI")
    fr = fstore.load_factor_returns(run_id)
    strat = cp.strategy_returns(fr)
    board = cp.strategy_leaderboard(strat, freq)

    exposures = fstore.load_exposure_matrix(run_id)
    styles = [s for s in manifest.get("styles", []) if s in exposures.columns]

    # Portfolio exposure: latest weights mapped from ticker to security_id.
    port_x, weights = pd.Series(dtype=float), pd.Series(dtype=float)
    try:
        hist = loaders.get_daily_weightings_history()
        latest = hist[hist["date"] == hist["date"].max()]
        latest = latest[~latest["symbol"].str.startswith("CASH_")]
        master = loaders.get_security_master()
        sid = latest["symbol"].map(dict(zip(master["ticker"], master["security_id"])))
        weights = pd.Series(latest["pct_nav"].to_numpy() / 100.0, index=sid).dropna()
        weights = weights[weights.index.notna()].groupby(level=0).sum()
        port_x = riskmod.portfolio_exposure(weights, exposures[styles])
    except Exception as exc:  # noqa: BLE001
        print(f"[control-panel] portfolio exposure unavailable: {exc}")

    # SPY exposure: returns-based, exactly as the model prices ETFs.
    spy = loaders.get_market_prices("SPY")
    spy_period = spy.reindex(spy.index.union(fr.index)).ffill().reindex(fr.index).pct_change()
    rb = reg.returns_based_exposures(spy_period.to_frame("SPY"), fr, styles)
    spy_x = rb.loc["SPY", styles].astype(float) if "SPY" in rb.index else pd.Series(0.0, index=styles)

    active = (port_x.reindex(styles) - spy_x.reindex(styles)) if len(port_x) else pd.Series(dtype=float)
    return {
        "manifest": manifest, "frequency": freq, "strat": strat, "board": board,
        "port_x": port_x, "spy_x": spy_x, "active": active, "weights": weights,
        "exposures": exposures, "styles": styles, "end": fr.index.max() if len(fr) else None,
        "spy_r2": float(rb.loc["SPY", "r2"]) if "SPY" in rb.index else np.nan,
    }


def _strategy_tilt(board: pd.DataFrame, active: pd.Series) -> pd.Series:
    """The fund's active exposure expressed in each strategy's direction
    (long "Low volatility" = short the resvol factor)."""
    sign = {k: s for _, k, s, _ in cp.STRATEGIES}
    if active is None or not len(active):
        return pd.Series(np.nan, index=board["strategy"])
    return pd.Series([active.get(f, np.nan) * sign.get(f, 1) for f in board["factor"]],
                     index=board["strategy"])


def _freshness() -> dict[str, pd.Timestamp | None]:
    def last(fn):
        try:
            s = fn()
            return s.index.max() if len(s) else None
        except Exception:  # noqa: BLE001
            return None
    rid = _run_id()
    fv_end = None
    if rid:
        try:
            fv_end = _factor_view(rid)["end"]
        except Exception:  # noqa: BLE001
            fv_end = None
    return {
        "Market": last(lambda: loaders.get_market_prices("SPY")),
        "NAV": last(_nav),
        "Factor model": fv_end,
    }


# ── Layout ────────────────────────────────────────────────────────────────────

def _slot(id_: str, span: dict) -> dmc.GridCol:
    return dmc.GridCol(dcc.Loading(html.Div(id=id_, style={"height": "100%"}), delay_show=300,
                                   parent_style={"height": "100%"}), span=span)


def layout():
    third = {"base": 12, "md": 6, "xl": 4}
    return dmc.Stack([
        dmc.Group([
            # A compact header rather than ui.page_title: on a one-screen panel the
            # H1 and its margin cost a whole row of charts.
            dmc.Group([
                dmc.Title("Control Panel", order=3, c=NAVY),
                _info("One-screen overview. Every 'this month' number is a rolling 21 trading "
                      "day lookback, judged against the 252-day (one year) equivalent. "
                      "Benchmark: SPY. Regime: 3-state HMM on SPY. Volatility: GJR-GARCH. "
                      "Outlook: Kalman trend on the fund NAV. Hover the ℹ on any box for its method."),
            ], gap=6),
            dcc.Loading(html.Div(id="cp-freshness"), delay_show=300),
        ], justify="space-between", align="center", wrap="wrap", gap="xs"),
        dmc.Grid([
            # Regime detector across the top; alerts and fund vs SPY side by side under it.
            _slot("cp-regime", {"base": 12}),
            _slot("cp-alerts", {"base": 12, "md": 6}),
            _slot("cp-scorecard", {"base": 12, "md": 6}),

            # The volatility box frame (header + asset toggle) is static, and only
            # its body is a callback output: re-rendering the toggle from its own
            # callback would re-fire that callback every time it was inserted.
            dmc.GridCol(_box(
                "Volatility",
                dcc.Loading(html.Div(id="cp-vol"), delay_show=300),
                html.Span(id="cp-vol-sub"), _VOL_TEXT,
                right=dmc.SegmentedControl(id="cp-vol-asset", data=["SPY", "Portfolio"],
                                           value="SPY", size="xs"),
            ), span={"base": 12, "xl": 4}),
            _slot("cp-strategies", {"base": 12, "xl": 8}),

            _slot("cp-movers", third),
            _slot("cp-sectors", third),
            _slot("cp-cross", third),

            # Bottom row ends with the Kalman outlook in the bottom-right corner.
            _slot("cp-risk", third),
            _slot("cp-regime-cond", third),
            _slot("cp-outlook", third),
        ], gutter=6, align="stretch"),
        # Fires every callback once on page load; nothing else drives them.
        dcc.Store(id="cp-init", data=0),
    ], gap=4)


# ── Header: freshness ─────────────────────────────────────────────────────────

@callback(Output("cp-freshness", "children"), Input("cp-init", "data"))
def _freshness_box(_):
    try:
        today = pd.Timestamp.today().normalize()
        badges = []
        for label, ts in _freshness().items():
            if ts is None:
                color = "red"
            elif label == "Factor model":
                color = "orange" if (today - pd.Timestamp(ts)).days > cp.LIMITS["stale_model_days"] else "teal"
            else:
                lag = max(0, len(pd.bdate_range(pd.Timestamp(ts), today)) - 1)
                color = "orange" if lag > cp.LIMITS["stale_bdays"] else "teal"
            badges.append(dmc.Badge(f"{label} {_date(ts)}", color=color, variant="dot", size="sm"))
        ingested = loaders.data_version().split("|")[0]
        if ingested:
            badges.append(dmc.Badge(f"Ingested {ingested[:16].replace('T', ' ')} UTC",
                                    color="gray", variant="dot", size="sm"))
        return dmc.Group(badges, gap=4)
    except Exception:  # noqa: BLE001
        return dmc.Text("Freshness unavailable", size="xs", c="red")


# ── Scorecard: fund vs SPY ────────────────────────────────────────────────────

_PCT_METRICS = {"Return", "Volatility", "Max drawdown", "VaR 95% (1d)", "CVaR 95% (1d)", "Hit rate"}


def _fmt_metric(metric: str, v) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if metric == "Sharpe":
        return f"{v:+.2f}"
    if metric == "Return":
        return f"{v:+.1%}"
    return f"{v:.1%}" if metric != "Max drawdown" else f"{v:.1%}"


def _verdict(better) -> dmc.ThemeIcon | str:
    if better is None or (isinstance(better, float) and np.isnan(better)):
        return dmc.Text("·", c=TEXT_MUTED, size="sm")
    return dmc.ThemeIcon("✓" if better else "✗", color="teal" if better else "red",
                         variant="light", size="xs", radius="xl")


@callback(Output("cp-scorecard", "children"), Input("cp-init", "data"))
def _scorecard_box(_):
    try:
        nav = _nav()
        bench, ccy = _bench_eur()
        k = _kpis()
        long_n = min(cp.LONG, len(nav) - 1)
        windows = [W, long_n]
        sc = cp.scorecard(nav, bench, windows)
        wins = sc.groupby("window")["better"].apply(lambda s: int(s.fillna(False).sum()))
        totals = sc.groupby("window")["better"].apply(lambda s: int(s.notna().sum()))

        def head_cell(text, **kw):
            return dmc.TableTh(text, style={"textAlign": "right", "fontSize": 10, "padding": "2px 4px", **kw})

        head = dmc.TableThead([
            dmc.TableTr([dmc.TableTh("", style={"padding": "2px 4px"}),
                         *[html.Th(f"{n}D", colSpan=3, style={"textAlign": "center", "fontSize": 11,
                                                                 "padding": "2px 4px", "color": NAVY})
                           for n in windows]]),
            dmc.TableTr([dmc.TableTh("", style={"padding": "2px 4px"}),
                         *[c for _ in windows for c in (head_cell("Fund"), head_cell("SPY"), head_cell(""))]]),
        ])
        body = []
        for metric, _higher in cp.SCORECARD_METRICS:
            cells = [dmc.TableTd(metric, style={"fontSize": 10, "padding": "1px 4px", "whiteSpace": "nowrap"})]
            for n in windows:
                r = sc[(sc["window"] == n) & (sc["metric"] == metric)].iloc[0]
                cells += [
                    dmc.TableTd(_fmt_metric(metric, r["fund"]),
                                style={"textAlign": "right", "fontSize": 10, "fontWeight": 600, "padding": "1px 4px"}),
                    dmc.TableTd(_fmt_metric(metric, r["bench"]),
                                style={"textAlign": "right", "fontSize": 10, "color": TEXT_MUTED, "padding": "1px 4px"}),
                    dmc.TableTd(_verdict(r["better"]), style={"padding": "1px 2px", "width": 20}),
                ]
            body.append(dmc.TableTr(cells))
        table = dmc.Table([head, dmc.TableTbody(body)], striped=True, highlightOnHover=True,
                          withRowBorders=False)

        right = dmc.Group([dmc.Badge(f"{n}D {wins[n]}/{totals[n]}",
                                     color="teal" if wins[n] * 2 > totals[n] else "red",
                                     variant="light", size="sm") for n in windows], gap=4)
        foot = _stats([
            (f"Active {W}D", _pct(k["active_window"]), "fund − SPY", _tone(k["active_window"])),
            ("Tracking err.", _pct(k["tracking_error"], 1, False), f"{k['long_label']}"),
            ("Beta", _num(k["beta"]), "to SPY"),
        ], cols=3)
        return _box("Fund vs SPY", dmc.Stack([table, foot], gap=6),
                    f"SPY in {ccy} · {_date(k['as_of'])}", _SCORE_TEXT, right=right)
    except Exception:  # noqa: BLE001
        return _fail("Fund vs SPY")


_SCORE_TEXT = """
The fund against SPY over two windows: the rolling 21 trading days and the
longest window the fund's history allows (252 days once it has a year; until
then, every day since inception). ✓ = the fund did better on that metric,
✗ = SPY did. Lower is better for volatility, VaR and CVaR; a shallower max
drawdown and a higher return, Sharpe and hit rate (share of up days) are better.

Both sides cover the same calendar span, each measured on its own trading days.
SPY is converted to EUR, the fund's currency, when FX rates are in the store.
The badges count the ticks per window.
"""


# ── Portfolio outlook (Kalman) ────────────────────────────────────────────────

@callback(Output("cp-outlook", "children"), Input("cp-init", "data"))
def _outlook_box(_):
    try:
        nav = _nav()
        ko = cp.kalman_outlook(nav)
        bench, ccy = _bench_eur()
        lo, hi = ko["band_ret"]
        stats = _stats([
            ("21d projection", _pct(ko["projected_ret"]), f"{_pct(lo)} … {_pct(hi)}", _tone(ko["projected_ret"])),
            ("Trend speed", _pct(ko["slope_ann"], 0), f"21d ago {_pct(ko['slope_window_ago'], 0)}",
             _tone(ko["slope_ann"])),
            ("NAV vs trend", _pct(ko["gap"]), "above = stretched", _tone(ko["gap"])),
        ], cols=3)

        idx, filt, fc = ko["index"], ko["filter"], ko["forecast"]
        b = bench[(bench.index >= idx.index[0]) & (bench.index <= idx.index[-1])].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]),
                                 y=list(fc["upper_1s"]) + list(fc["lower_1s"][::-1]),
                                 fill="toself", fillcolor="rgba(184,150,46,0.15)", line=dict(width=0),
                                 name="±1σ", hoverinfo="skip"))
        if len(b) > 1:
            b = b / b.iloc[0] * 100
            fig.add_trace(go.Scatter(x=b.index, y=b, name=f"SPY ({ccy})", line=dict(color=SLATE, width=1.5),
                                     hovertemplate="%{x|%d %b}  %{y:.1f}<extra>SPY</extra>"))
        fig.add_trace(go.Scatter(x=idx.index, y=idx, name="Fund", line=dict(color=NAVY, width=2),
                                 hovertemplate="%{x|%d %b}  %{y:.1f}<extra>Fund</extra>"))
        fig.add_trace(go.Scatter(x=filt.index, y=filt["trend"], name="Kalman trend",
                                 line=dict(color=GOLD, width=1.5),
                                 hovertemplate="%{x|%d %b}  %{y:.1f}<extra>Trend</extra>"))
        fig.add_trace(go.Scatter(x=[filt.index[-1], *fc.index], y=[filt["trend"].iloc[-1], *fc["trend"]],
                                 name="Projection", line=dict(color=GOLD, width=2, dash="dash"),
                                 hovertemplate="%{x|%d %b}  %{y:.1f}<extra>Projection</extra>"))
        fig.add_hline(y=100, line_color=SLATE, line_width=0.5)
        return _box("Portfolio outlook", dmc.Stack([stats, _graph(fig, 210, legend=True)], gap=2),
                    f"Kalman trend on the fund NAV, rebased to 100 · {_date(idx.index[-1])}", _OUTLOOK_TEXT)
    except Exception:  # noqa: BLE001
        return _fail("Portfolio outlook")


_OUTLOOK_TEXT = """
A two-state Kalman filter (level and slope) tracks the fund's unit price
(time-weighted, so deposits and withdrawals do not move it). The dashed line
extends today's slope 21 days with a mild decay, and the band widens with the
square root of time.

Read it as the fund's **momentum**: compare *trend speed* now with 21 days ago
to see whether the trend is accelerating or rolling over. It is not a return
forecast. *NAV vs trend* shows how far the price has run ahead of, or fallen
behind, its own trend.
"""


# ── Regime detector ───────────────────────────────────────────────────────────

#: The chart's lookback. The model itself is fitted on SPY's full history.
_REGIME_DAYS = cp.LONG


@callback(Output("cp-regime", "children"), Input("cp-init", "data"))
def _regime_box(_):
    try:
        from plotly.subplots import make_subplots

        res = _regime()
        cur, params = res["current"], res["params"]
        proba = res["proba"].tail(_REGIME_DAYS)
        state = res["state"].reindex(proba.index)
        spy = loaders.get_market_prices("SPY").reindex(proba.index).ffill()

        # Row 1: SPY price over regime shading. Row 2: state probabilities.
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            vertical_spacing=0.03, specs=[[{"secondary_y": True}], [{}]])
        # Shading: one full-height bar per day on a hidden 0–1 axis, coloured by
        # that day's most likely state. Bars (not shapes) so the shading is exact
        # per day and survives the weekend rangebreaks without gaps.
        for s in rg.STATES:
            on = (state == s).astype(float).replace(0.0, np.nan)
            fig.add_trace(go.Bar(x=proba.index, y=on, name=s, marker_color=rg.STATE_COLORS[s],
                                 opacity=0.28, hoverinfo="skip", width=86_400_000),
                          row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=spy.index, y=spy, name="SPY", line=dict(color=NAVY, width=1.8), showlegend=False,
            customdata=state.to_numpy(),
            hovertemplate="%{x|%d %b %Y}  SPY %{y:.2f}<br>%{customdata}<extra></extra>"),
            row=1, col=1, secondary_y=True)

        # Probabilities as a 100% stacked area in the state colours: the share of
        # each colour is how sure the model is, so a regime building or fading
        # shows as a band swelling or thinning rather than as cells switching.
        for s in ("Bear", "Sideways", "Bull"):
            fig.add_trace(go.Scatter(
                x=proba.index, y=proba[s], name=s, showlegend=False, stackgroup="proba",
                mode="lines", line=dict(width=0.6, color=rg.STATE_COLORS[s], shape="spline", smoothing=0.6),
                fillcolor=rg.STATE_COLORS[s], opacity=0.9,
                hovertemplate=f"%{{x|%d %b %Y}}  {s}: %{{y:.0%}}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dot", line_color="white", line_width=1, row=2, col=1)

        fig.update_layout(barmode="overlay", bargap=0, legend=dict(traceorder="normal"))
        fig.update_yaxes(range=[0, 1], visible=False, row=1, col=1, secondary_y=False)
        lo, hi = float(spy.min()), float(spy.max())
        pad = (hi - lo) * 0.05
        fig.update_yaxes(range=[lo - pad, hi + pad], side="left", showgrid=True, tickprefix="$",
                         row=1, col=1, secondary_y=True)
        fig.update_yaxes(range=[0, 1], tickvals=[0, 0.5, 1], tickformat=".0%", showgrid=False,
                         tickfont=dict(size=9), title_text="prob.", row=2, col=1)
        # Skip weekends and market holidays (weekdays with no SPY close), so the
        # shading runs unbroken from one trading day to the next.
        holidays = pd.bdate_range(proba.index[0], proba.index[-1]).difference(proba.index)
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                      dict(values=[d.strftime("%Y-%m-%d") for d in holidays])],
                         showgrid=False)

        ahead = cur["proba_21d"]
        badge = dmc.Badge(f"{cur['state']} · {cur['proba'][cur['state']]:.0%}", variant="filled",
                          color={"Bull": "teal", "Sideways": "blue", "Bear": "red"}[cur["state"]], size="md")
        stats = _stats([
            ("In regime", f"{cur['days_in_regime']}d", f"typical {params.loc[cur['state'], 'exp_duration_d']:.0f}d"),
            ("Still in 21d", f"{ahead[cur['state']]:.0%}", "A²¹ from today"),
            ("Bull / Side / Bear", " / ".join(f"{cur['proba'][s]:.0%}" for s in rg.STATES), "today"),
            ("Bear in 21d", f"{ahead['Bear']:.0%}", "probability", NEG if ahead["Bear"] > 0.3 else NAVY),
            ("Regime vol", f"{params.loc[cur['state'], 'ann_vol_pct']:.0f}%",
             f"ann. ret {params.loc[cur['state'], 'ann_ret_pct']:+.0f}%"),
        ], cols={"base": 3, "sm": 5})
        return _box("Regime detector", dmc.Stack([stats, _graph(fig, 300, legend=True, left=46, right=8)], gap=2),
                    f"3-state HMM on SPY, last {_REGIME_DAYS} days · {_date(cur['date'])}",
                    _REGIME_TEXT, right=badge)
    except Exception:  # noqa: BLE001
        return _fail("Regime detector")


_REGIME_TEXT = r"""
A hidden Markov model assumes the market switches between three unobserved
states, each with its own distribution of daily SPY return and 21-day realised
volatility. It is fitted with EM (`hmmlearn`, six random restarts, the best
log-likelihood kept) on SPY's full history; the chart shows the last 252 trading days.

- **Bands** behind the price are the most likely state each day: green Bull,
  blue Sideways, red Bear.
- **Probability band** underneath: each state's probability per day, stacked to
  100% in the same colours. A colour widening means the model is growing more
  sure of that regime; the dotted line marks 50%.
- **Labels** are assigned after every fit by mean return (highest = Bull,
  lowest = Bear), so they mean the same thing night after night.
- **Probabilities are filtered**: each day uses data up to that day only (the
  forward algorithm), so the bands are what you could have known at the time.
- **Typical length** is $1/(1-p_{ii})$; **still in 21d** is today's
  probabilities times $A^{21}$.

"Bear" is the high-volatility state rather than one with a strongly negative
average: sell-offs come mixed with sharp rebounds, so its mean is low, not
necessarily below zero.
"""


# ── Alerts ────────────────────────────────────────────────────────────────────

_ALERT_COLOR = {"red": "#C0392B", "orange": "#E67E22", "green": "#2E7D6B"}


def _age(bd: int) -> str:
    return "today" if bd == 0 else f"{bd}d ago"


@callback(Output("cp-alerts", "children"), Input("cp-init", "data"))
def _alerts_box(_):
    try:
        def safe(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001
                return None
        rid = _run_id()
        fv = safe(lambda: _factor_view(rid)) if rid else None
        weights_hist = safe(loaders.get_daily_weightings_history)
        alerts = cp.build_alerts(
            today=pd.Timestamp.today().normalize(),
            regime=safe(_regime), vol=safe(lambda: _vol("SPY")), kpis=safe(_kpis), nav=safe(_nav),
            positions=safe(_positions),
            positions_as_of=weights_hist["date"].max() if weights_hist is not None and len(weights_hist) else None,
            active_exposure=fv["active"] if fv else None,
            leaderboard=fv["board"] if fv else None,
            factor_end=fv["end"] if fv else None,
            freshness=_freshness(),
        )
        counts = {lvl: sum(a["level"] == lvl for a in alerts) for lvl in _ALERT_COLOR}
        right = dmc.Group([dmc.Badge(str(n), color={"red": "red", "orange": "orange", "green": "teal"}[lvl],
                                     variant="filled", size="sm", circle=True)
                           for lvl, n in counts.items() if n], gap=3)
        if not alerts:
            body = dmc.Center(dmc.Stack([dmc.Text("All clear", fw=700, c=POS),
                                         dmc.Text("Every rule is inside its limit.", size="xs", c=TEXT_MUTED)],
                                        gap=0, align="center"), h=150)
        else:
            rows = [html.Div([
                html.Span(style={"width": 4, "alignSelf": "stretch", "borderRadius": 2, "flex": "0 0 4px",
                                 "background": _ALERT_COLOR[a["level"]]}),
                html.Div([
                    dmc.Group([dmc.Text(a["title"], size="xs", fw=700, c=NAVY, lh=1.25, truncate="end",
                                        style={"flex": 1, "minWidth": 0}),
                               dmc.Text(_age(a["age_bd"]), size="10px", c=TEXT_MUTED, style={"flex": "0 0 auto"})],
                              justify="space-between", gap=4, wrap="nowrap"),
                    dmc.Text(a["detail"], size="11px", c=TEXT_MUTED, lh=1.25),
                ], style={"flex": 1, "minWidth": 0}),
            ], style={"display": "flex", "gap": 7, "padding": "4px 0", "borderBottom": "1px solid #EEF2F7"})
                for a in alerts]
            body = dmc.ScrollArea(html.Div(rows), h=245, type="auto", offsetScrollbars=True)
        return _box("Alerts", body, "by relevance × recency", _ALERT_TEXT, right=right)
    except Exception:  # noqa: BLE001
        return _fail("Alerts")


_L = cp.LIMITS
_ALERT_TEXT = f"""
**Ranking.** Each alert scores *relevance × recency*: red 3, orange 2, green 1,
times $0.4 + 0.6 \\cdot 0.5^{{age/{cp.RECENCY_HALF_LIFE}}}$, where age is the
business days since the condition *began*. A fresh orange outranks a month-old
red, but an old red never falls below a fresh green.

- **Bear regime** (red): HMM puts more than {_L['bear_proba']:.0%} on Bear.
- **Fund drawdown limit** (red): more than {abs(_L['drawdown']):.0%} below peak.
- **Position drawdown**: a holding down more than {abs(_L['position_drop']):.0%} over 21 days (red beyond {abs(2 * _L['position_drop']):.0%}).
- **Regime change** (orange): the most likely state switched within {_L['regime_flip_days']} trading days.
- **Volatility elevated** (orange): SPY's GARCH vol above the {_L['vol_pct']:.0%} percentile of its history.
- **Volatility expanding** (orange): SPY's 21d realised vol above {_L['vol_expansion']}× its 252d.
- **Concentration** (orange): a position above {_L['max_position']:.0%} of NAV.
- **Stale data** (orange): market or NAV more than {_L['stale_bdays']} business days old, factor run more than {_L['stale_model_days']} days.
- **Tilt against the tape** (green): active exposure of at least {_L['exposure_z']}σ to a factor that cost us money over 21 days.

Thresholds live in `capital.analytics.control_panel.LIMITS`.
"""


# ── Volatility ────────────────────────────────────────────────────────────────

@callback(Output("cp-vol", "children"), Output("cp-vol-sub", "children"),
          Input("cp-vol-asset", "value"))
def _vol_box(asset):
    asset = asset or "SPY"
    try:
        v = _vol(asset)
        g = v["garch"].tail(cp.LONG)
        fig = go.Figure()
        if asset == "SPY":
            vix = loaders.get_market_prices("VIX").reindex(g.index)
            if vix.notna().any():
                fig.add_trace(go.Scatter(x=g.index, y=vix, name="VIX", line=dict(color=SLATE, width=1),
                                         hovertemplate="%{x|%d %b %Y}  %{y:.1f}<extra>VIX</extra>"))
        fig.add_trace(go.Scatter(x=g.index, y=v["rv_window"].reindex(g.index), name="Realised 21d",
                                 line=dict(color=GOLD, width=1.5),
                                 hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>RV 21d</extra>"))
        rv252 = v["rv_long"].reindex(g.index)
        if rv252.notna().any():
            fig.add_trace(go.Scatter(x=g.index, y=rv252, name="Realised 252d",
                                     line=dict(color=SAGE, width=1.5, dash="dash"),
                                     hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>RV 252d</extra>"))
        fig.add_trace(go.Scatter(x=g.index, y=g, name="GJR-GARCH", line=dict(color=NAVY, width=2),
                                 hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>GARCH</extra>"))
        fc = v["forecast"]
        fig.add_trace(go.Scatter(x=[g.index[-1], *fc.index], y=[g.iloc[-1], *fc.values],
                                 name="Forecast", line=dict(color=NAVY, width=2, dash="dot"),
                                 hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>Forecast</extra>"))
        if np.isfinite(v["long_run"]):
            fig.add_hline(y=v["long_run"], line_dash="dot", line_color=SLATE, line_width=1,
                          annotation_text="long-run", annotation_font_size=9,
                          annotation_position="top left")
        fig.update_yaxes(ticksuffix="%")

        def ratio(a, b):
            return f"{a / b:.2f}×" if np.isfinite(a) and np.isfinite(b) and b else "–"
        up = v["change_window"] > 0
        stats = _stats([
            ("GARCH now", f"{v['now']:.1f}%", f"{v['pct_all']:.0%} pctl · 1y {v['pct_1y']:.0%}",
             NEG if v["pct_all"] > cp.LIMITS["vol_pct"] else NAVY),
            ("Next 21d", f"{v['forecast_avg']:.1f}%", f"long-run {_num(v['long_run'], '{:.1f}')}%"),
            ("vs RV 21d", ratio(v["now"], v["rv_window_now"]), f"RV {v['rv_window_now']:.1f}%"),
            ("vs RV 252d", ratio(v["now"], v["rv_long_now"]), f"RV {_num(v['rv_long_now'], '{:.1f}')}%"),
            ("21d change", f"{_num(v['change_window'], '{:+.1f}')} pt", "rising" if up else "falling",
             NEG if up else POS),
            ("RV 21/252", ratio(v["rv_window_now"], v["rv_long_now"]),
             f"half-life {_num(v['half_life_d'], '{:.0f}')}d"),
        ], cols=3)
        warn = None
        if v["n_obs"] < cp.LONG:
            warn = dmc.Text(f"⚠ Only {v['n_obs']} days of history: fit and percentiles are unstable, "
                            "no 252-day window.", size="xs", c="orange")
        body = dmc.Stack([x for x in (stats, warn, _graph(fig, 180, legend=True)) if x is not None], gap=2)
        return body, f"GJR-GARCH(1,1,1) on {asset}, Student-t · {_date(v['as_of'])}"
    except Exception:  # noqa: BLE001
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True), color="red",
                         variant="light", title="Volatility failed"), ""


_VOL_TEXT = """
The GARCH line is the model's estimate of *today's* volatility, set against
the yardsticks a PM compares it with:

- **Realised 21d**: what the market actually delivered this month. GARCH above
  it means the model still remembers a recent shock.
- **Realised 252d**: the past year's level. **RV 21/252** above 1 is an
  expanding-vol term structure.
- **Long-run**: where the forecast mean-reverts to; the half-life is how fast.
- **VIX** (SPY only): the option market's 30-day implied vol.

**Next 21d** is the root of the average forecast variance over 21 days: the
number to size a one-month holding on. Percentiles rank today's GARCH vol
against its whole history and the last year.
"""


# ── Strategies: leaderboard + factor map ──────────────────────────────────────

@callback(Output("cp-strategies", "children"), Input("cp-init", "data"))
def _strategy_box(_):
    try:
        rid = _run_id()
        if not rid:
            return _box("Strategies", ui.alert("No factor-model run yet.", "yellow"))
        fv = _factor_view(rid)
        board = fv["board"]
        tilt = _strategy_tilt(board, fv["active"])
        n_per = max(1, round(W * periods_per_year(fv["frequency"]) / 252))

        lb = board.iloc[::-1]
        bars = _hbar(lb["ret_window"], lb["strategy"],
                     custom=np.c_[lb["ret_long"], lb["t_6m"], tilt.reindex(lb["strategy"]).to_numpy()],
                     hover="<b>%{y}</b><br>" + f"{W}d" + " %{x:+.2%}<br>1y %{customdata[0]:+.1%}"
                           "<br>t 6m %{customdata[1]:+.1f}<br>our tilt %{customdata[2]:+.2f}σ<extra></extra>",
                     fmt="{:+.1%}", pos=BARRA_POS, neg=BARRA_NEG)
        bars.update_xaxes(tickformat="+.0%")
        bars_g = _graph(bars, 215, left=ui.axis_margin(lb["strategy"], font_size=9, pad=12), right=36)

        # Factor map: persistence (x) against this month (y), coloured by our tilt.
        t = board["t_6m"].to_numpy(dtype=float)
        y = board["ret_window"].to_numpy(dtype=float)
        tv = tilt.reindex(board["strategy"]).to_numpy(dtype=float)
        has_tilt = bool(np.isfinite(tv).any())
        tmax = float(np.nanmax(np.abs(tv))) if has_tilt else 1.0
        xr = max(2.5, float(np.nanmax(np.abs(t))) * 1.15) if np.isfinite(t).any() else 3
        yr = float(np.nanmax(np.abs(y))) * 1.25 if np.isfinite(y).any() else 0.05
        fmap = go.Figure()
        for x0, x1 in ((-xr, -2), (2, xr)):
            fmap.add_shape(type="rect", x0=x0, x1=x1, y0=-yr, y1=yr, line_width=0,
                           fillcolor="rgba(148,163,184,0.12)", layer="below")
        fmap.add_trace(go.Scatter(
            x=t, y=y, mode="markers+text", text=board["strategy"],
            textposition=["top center" if i % 2 else "bottom center" for i in range(len(board))],
            textfont=dict(size=9, color=TEXT_MUTED),
            marker=dict(size=11, line=dict(width=1, color=SLATE),
                        color=np.nan_to_num(tv) if has_tilt else STEEL,
                        colorscale=BARRA_SCALE if has_tilt else None, cmid=0 if has_tilt else None,
                        cmin=-tmax if has_tilt else None, cmax=tmax if has_tilt else None,
                        showscale=has_tilt,
                        colorbar=dict(title=dict(text="our tilt", font=dict(size=9)), thickness=8,
                                      len=0.8, tickfont=dict(size=9)) if has_tilt else None),
            customdata=np.c_[board["hit_6m"], board["ac_1y"], tv],
            hovertemplate="<b>%{text}</b><br>t 6m %{x:+.1f} · " + f"{W}d" + " %{y:+.2%}"
                          "<br>hit %{customdata[0]:.0%} · AC %{customdata[1]:+.2f}"
                          "<br>our tilt %{customdata[2]:+.2f}σ<extra></extra>"))
        fmap.add_vline(x=0, line_color=SLATE, line_width=1)
        fmap.add_hline(y=0, line_color=SLATE, line_width=1)
        for txt, xa, ya, anchor in (("persistent & paying", 0.99, 0.99, "right"),
                                    ("persistently costing", 0.01, 0.01, "left")):
            fmap.add_annotation(text=txt, xref="paper", yref="paper", x=xa, y=ya, showarrow=False,
                                font=dict(size=9, color=TEXT_MUTED), xanchor=anchor,
                                yanchor="top" if ya > 0.5 else "bottom")
        fmap.update_xaxes(range=[-xr, xr], title_text="persistence: t-stat, last 6 months")
        fmap.update_yaxes(range=[-yr, yr], tickformat="+.0%", title_text=f"{W}d return")
        map_g = _graph(fmap, 215, left=42, bottom=30)

        body = dmc.SimpleGrid([bars_g, map_g], cols={"base": 1, "md": 2}, spacing="xs")
        return _box("Strategies", body,
                    f"style-factor returns, last {n_per} weekly periods ≈ {W}d · to {_date(fv['end'])}",
                    _STRAT_TEXT)
    except Exception:  # noqa: BLE001
        return _fail("Strategies")


_STRAT_TEXT = """
Each strategy is a **style-factor return** from the nightly cross-sectional
factor model: the payoff to one unit of exposure to that style, neutral to the
market, industries and countries. Short-side strategies carry a minus sign:
Low volatility = −Residual Vol, Low beta = −Beta, Small cap = −Size,
Balance-sheet strength = −Leverage, Illiquidity = −Liquidity. Mean reversion
is the Short-Term Reversal factor (long last month's losers).

**Left**: ranked by the 21-day return; hover for the 1-year return and t-stat.

**Right, the factor map**: persistence (t-stat of the last ~26 weekly returns)
across, this month's return up. Top right = has been paying consistently *and*
paid this month. The shaded bands mark |t| > 2, unlikely to be luck. Dot colour
is the fund's active tilt versus SPY in that strategy's direction: green dots in
the top right are bets that are working. SPY's exposures are estimated from its
returns, the same way the model prices ETFs.
"""


# ── Movers ────────────────────────────────────────────────────────────────────

@callback(Output("cp-movers", "children"), Input("cp-init", "data"))
def _movers_box(_):
    try:
        pos = _positions()
        if pos.empty:
            return _box("Winners & losers", ui.alert("No position history available."))
        p = pos.sort_values("ctr_bps")
        fig = _hbar(p["ctr_bps"], p.index, custom=np.c_[p["ret_window"], p["weight"], p["name"]],
                    hover="<b>%{y}</b> %{customdata[2]}<br>%{x:+.0f} bps<br>return %{customdata[0]:+.1%}"
                          " · weight %{customdata[1]:.1%}<extra></extra>",
                    fmt="{:+.0f}")
        fig.update_xaxes(title_text="contribution, bps of NAV")
        total = pos["ctr_bps"].sum()
        best, worst = pos.index[0], pos.index[-1]
        stats = _stats([
            ("Securities", f"{total:+.0f} bps", f"{W}d, ex cash", _tone(total)),
            ("Best", best, f"{pos.loc[best, 'ctr_bps']:+.0f} bps", POS),
            ("Worst", worst, f"{pos.loc[worst, 'ctr_bps']:+.0f} bps", NEG),
        ], cols=3)
        height = max(150, 13 * len(p) + 36)
        return _box("Winners & losers", dmc.Stack([stats, _graph(
            fig, height, left=ui.axis_margin(p.index, font_size=9, pad=12), right=30, bottom=28)], gap=2),
            f"weight × return, {W}d", _MOVERS_TEXT)
    except Exception:  # noqa: BLE001
        return _fail("Winners & losers")


_MOVERS_TEXT = """
Contribution = yesterday's weight × today's return, summed over the last 21
trading days, in basis points of NAV. Ranking by contribution rather than raw
return is what matters to the fund: +40% on a 0.5% position is worth less than
−8% on a 10% one. Hover a bar for the position's own return and current weight.
Cash and FX balances are excluded.
"""


# ── Sectors ───────────────────────────────────────────────────────────────────

@callback(Output("cp-sectors", "children"), Input("cp-init", "data"))
def _sector_box(_):
    try:
        prices = _prices(tuple(cp.SECTOR_ETFS) + ("SPY",))
        tbl = cp.asset_table(prices, cp.SECTOR_ETFS, prices["SPY"])
        if tbl.empty:
            return _box("S&P 500 sectors", ui.alert(
                "No sector ETF prices yet. Back-fill once with `capital-ingest market --days 3700`.",
                "yellow"))
        rid = _run_id()
        weights = cp.us_sector_weights(fstore.load_security_meta(rid)) if rid else pd.Series(dtype=float)
        t = tbl.iloc[::-1]
        wt = [weights.get(lbl, np.nan) for lbl in t["label"]]
        labels = [f"{r.ticker} {r.label}" + (f" ({w:.0%})" if np.isfinite(w) else "")
                  for r, w in zip(t.itertuples(), wt)]
        fig = _hbar(t["rel_window"], labels, custom=np.c_[t["ret_window"], t["ret_long"], t["rv_window"]],
                    hover="<b>%{y}</b><br>vs SPY %{x:+.1%}<br>" + f"{W}d" + " %{customdata[0]:+.1%} · "
                          "1y %{customdata[1]:+.1%}<br>vol %{customdata[2]:.0%}<extra></extra>")
        fig.update_xaxes(tickformat="+.0%", title_text=f"{W}d return relative to SPY")
        spy_w = cp.price_return(prices["SPY"], W)
        lead, lag = tbl.iloc[0], tbl.iloc[-1]
        stats = _stats([
            (f"SPY {W}d", _pct(spy_w), "the hurdle", _tone(spy_w)),
            ("Leader", lead["ticker"], _pct(lead["ret_window"]), POS),
            ("Laggard", lag["ticker"], _pct(lag["ret_window"]), NEG),
        ], cols=3)
        note = " · (x%) = est. SPY weight" if len(weights) else ""
        return _box("S&P 500 sectors", dmc.Stack([stats, _graph(
            fig, max(150, 13 * len(t) + 36), left=ui.axis_margin(labels, font_size=9, pad=12),
            right=30, bottom=28)], gap=2),
            f"SPDR sector ETFs · {_date(tbl['as_of'].max())}{note}", _SECTOR_TEXT)
    except Exception:  # noqa: BLE001
        return _fail("S&P 500 sectors")


_SECTOR_TEXT = """
The eleven SPDR sector ETFs, ranked by their 21-day return **relative to SPY**,
so a bar to the right means the sector led the index. Hover a bar for its
absolute 21-day and 1-year returns and its realised volatility.

The percentage after a sector name is its approximate weight in the S&P 500,
estimated by cap-weighting the US stocks in the factor universe (shown only when
the universe covers at least 100 US names).
"""


# ── Cross-asset ───────────────────────────────────────────────────────────────

@callback(Output("cp-cross", "children"), Input("cp-init", "data"))
def _cross_box(_):
    try:
        prices = _prices(tuple(cp.CROSS_ASSET))
        tbl = cp.asset_table(prices, cp.CROSS_ASSET, prices.get("SPY"))
        if tbl.empty:
            return _box("Cross-asset", ui.alert("No market prices.", "yellow"))
        fig = go.Figure()
        ends = []
        for tk in cp.CROSS_ASSET:
            p = prices.get(tk)
            if p is None or len(p.dropna()) <= W:
                continue
            seg = p.dropna().iloc[-W - 1:]
            path = seg / seg.iloc[0] - 1
            ends.append((tk, float(path.iloc[-1])))
            fig.add_trace(go.Scatter(
                x=path.index, y=path, name=tk, mode="lines",
                line=dict(color=ASSET_COLORS.get(tk, SLATE), width=2 if tk == "SPY" else 1.5),
                hovertemplate=f"{cp.CROSS_ASSET[tk]}<br>" + "%{x|%d %b}  %{y:+.1%}<extra></extra>"))
        # Direct end-of-line labels instead of a legend, nudged apart so they don't collide.
        ends.sort(key=lambda e: e[1])
        span = (max(e[1] for e in ends) - min(e[1] for e in ends)) or 0.01
        placed = []
        for tk, yv in ends:
            y_lab = max(yv, placed[-1] + span * 0.09) if placed else yv
            placed.append(y_lab)
            fig.add_annotation(x=1.0, xref="paper", y=y_lab, xanchor="left", showarrow=False,
                               text=f"<b>{tk}</b> {yv:+.1%}", font=dict(size=10, color=ASSET_COLORS.get(tk, SLATE)))
        fig.add_hline(y=0, line_color=SLATE, line_width=1)
        fig.update_yaxes(tickformat="+.0%")

        one_y = {r.ticker: r.ret_long for r in tbl.itertuples()}
        stats = _stats([
            ("Equities 1y", _pct(one_y.get("SPY")), "SPY", _tone(one_y.get("SPY", np.nan))),
            ("Bonds 1y", _pct(one_y.get("TLT")), "TLT", _tone(one_y.get("TLT", np.nan))),
            ("Gold 1y", _pct(one_y.get("GLD")), "GLD", _tone(one_y.get("GLD", np.nan))),
        ], cols=3)
        return _box("Cross-asset", dmc.Stack([stats, _graph(fig, max(150, 13 * len(cp.SECTOR_ETFS) + 36),
                                                             right=74, bottom=22)], gap=2),
                    f"last {W} trading days, cumulative · {_date(tbl['as_of'].max())}", _CROSS_TEXT)
    except Exception:  # noqa: BLE001
        return _fail("Cross-asset")


_CROSS_TEXT = """
Cumulative return over the last 21 trading days for the long-only sleeve:
US equities (SPY), growth (QQQ), small caps (IWM), long Treasuries (TLT),
high-yield credit (HYG) and gold (GLD). Kept apart from the strategy
leaderboard because those are dollar-neutral long/short payoffs and these are not.
"""


# ── Risk ──────────────────────────────────────────────────────────────────────

@callback(Output("cp-risk", "children"), Input("cp-init", "data"))
def _risk_box(_):
    try:
        rid = _run_id()
        if not rid:
            return _box("Risk", ui.alert("No factor-model run yet.", "yellow"))
        fv = _factor_view(rid)
        weights, exposures = fv["weights"], fv["exposures"]
        cov = fstore.load_covariance(rid)
        spec = fstore.load_specific_risk(rid)["sigma"]
        covered = weights[weights.index.isin(exposures.index)]
        if covered.empty:
            return _box("Risk", ui.alert("No current position is covered by the latest factor run.", "yellow"))
        d = riskmod.decompose_risk(covered, exposures, cov, spec, fv["manifest"].get("factor_groups", {}))
        sc = riskmod.security_contributions(covered, exposures, cov, spec)
        k = _kpis()
        sid_to_sym = dict(zip(*[loaders.get_security_master()[c] for c in ("security_id", "ticker")]))

        stats = _stats([
            ("Model vol", _pct(d["total_vol"], 1, False), "ex-ante, ann."),
            ("Factor / specific", f"{d['factor_share']:.0%} / {d['specific_share']:.0%}", "share of variance"),
            ("Realised TE", _pct(k["tracking_error"], 1, False), f"vs SPY, {k['long_label']}"),
            ("Covered", _pct(d["weight_covered"] / max(weights.sum(), 1e-9), 0, False),
             f"{d['n_covered']} of {len(weights)} names"),
        ], cols={"base": 2, "sm": 4, "xl": 2})

        active = fv["active"].dropna().sort_values()
        labels = [STYLES[f].label if f in STYLES else f for f in active.index]
        exp_fig = _hbar(active, labels,
                        custom=np.c_[fv["port_x"].reindex(active.index), fv["spy_x"].reindex(active.index)],
                        hover="<b>%{y}</b> active %{x:+.2f}σ<br>fund %{customdata[0]:+.2f} · "
                              "SPY %{customdata[1]:+.2f}<extra></extra>", fmt="{:+.2f}")
        exp_fig.update_xaxes(title_text="active exposure vs SPY (σ)")

        top = sc.head(8).iloc[::-1]
        names = [sid_to_sym.get(s, s) for s in top.index]
        rc = go.Figure()
        rc.add_trace(go.Bar(x=top["factor_ctr"], y=names, orientation="h", name="factor",
                            marker_color=NAVY, hovertemplate="%{y} factor %{x:.2%}<extra></extra>"))
        rc.add_trace(go.Bar(x=top["specific_ctr"], y=names, orientation="h", name="specific",
                            marker_color=GOLD, hovertemplate="%{y} specific %{x:.2%}<extra></extra>",
                            text=[f"{v:.0%}" for v in top["pct_of_risk"]], textposition="outside",
                            textfont=dict(size=10), cliponaxis=False))
        rc.update_layout(barmode="stack")
        rc.update_xaxes(tickformat=".0%", title_text="contribution to vol (label: % of risk)")
        rc.update_yaxes(showgrid=False)

        charts = dmc.Stack([
            _graph(exp_fig, 165, left=ui.axis_margin(labels, font_size=9, pad=12), right=34, bottom=32),
            _graph(rc, 140, legend=True, left=ui.axis_margin(names, font_size=9, pad=12), right=34, bottom=32),
        ], gap=2)
        return _box("Risk", dmc.Stack([stats, charts], gap=2),
                    f"factor model · SPY exposure R² {_num(fv['spy_r2'], '{:.2f}')}", _RISK_TEXT)
    except Exception:  # noqa: BLE001
        return _fail("Risk")


_RISK_TEXT = """
**Left**: the fund's style exposure minus SPY's, in cross-sectional standard
deviations: the fund's active factor bets. SPY's exposures are estimated by
regressing its returns on the factor returns.

**Right**: the positions that carry the most risk, split into the part that
comes through factors and the part that is stock-specific. A position's
contribution is weight × covariance with the portfolio ÷ portfolio vol, so the
bars add up to model vol. The label is the share of total risk. Sizing on risk
rather than on weight is the point: a 3% position that moves with everything
else can carry more risk than an 8% one that does not.
"""


# ── Regime-conditional strategy returns ───────────────────────────────────────

@callback(Output("cp-regime-cond", "children"), Input("cp-init", "data"))
def _regime_cond_box(_):
    try:
        rid = _run_id()
        if not rid:
            return _box("Strategies by regime", ui.alert("No factor-model run yet.", "yellow"))
        fv = _factor_view(rid)
        res = _regime()
        long = rg.regime_conditional_returns(fv["strat"], res["state"], periods_per_year(fv["frequency"]))
        if long.empty:
            return _box("Strategies by regime", ui.alert("Not enough overlap between regimes and factor history."))
        piv = long.pivot(index="strategy", columns="regime", values="ann_return").reindex(columns=list(rg.STATES))
        n = long.groupby("regime")["n_periods"].max().reindex(list(rg.STATES))
        cur = res["current"]["state"]
        piv = piv.sort_values(cur, ascending=True)
        zmax = float(np.nanmax(np.abs(piv.to_numpy()))) or 1.0
        cols = [f"<b>{s} ◀ now</b>" if s == cur else s for s in piv.columns]
        cols = [f"{c}<br>{int(n[s]) if pd.notna(n[s]) else 0} wks" for c, s in zip(cols, piv.columns)]
        fig = go.Figure(go.Heatmap(
            z=piv.to_numpy(), x=cols, y=piv.index, zmid=0, zmin=-zmax, zmax=zmax,
            colorscale=BARRA_SCALE, xgap=2, ygap=2,
            text=[[_pct(v, 0) for v in row] for row in piv.to_numpy()], texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="%{y} in %{x}: %{z:+.1%} a year<extra></extra>",
            colorbar=dict(tickformat="+.0%", thickness=8, tickfont=dict(size=9))))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        best = piv[cur].dropna().sort_values(ascending=False)
        stats = _stats([
            ("Now", cur, f"HMM {res['current']['proba'][cur]:.0%}", rg.STATE_COLORS[cur]),
            ("Best in regime", best.index[0] if len(best) else "–", _pct(best.iloc[0], 0) + " / yr" if len(best) else "", BARRA_POS),
            ("Worst in regime", best.index[-1] if len(best) else "–", _pct(best.iloc[-1], 0) + " / yr" if len(best) else "", BARRA_NEG),
        ], cols=3)
        return _box("Strategies by regime", dmc.Stack([stats, _graph(
            fig, 250, left=ui.axis_margin(piv.index, font_size=9, pad=12), right=8, bottom=34)], gap=2),
            "annualised return by the regime known at the start of each week", _COND_TEXT)
    except Exception:  # noqa: BLE001
        return _fail("Strategies by regime")


_COND_TEXT = """
Each weekly factor return is tagged with the HMM regime that was most likely at
the *previous* week's close, so the heatmap answers "given the regime we can see
now, what has each strategy tended to earn next?" (no look-ahead). Rows are
sorted by the current regime's column.

The number of weeks under each column is the sample size: the factor history
only spans the factor model's estimation window, so a regime with few weeks
gives a noisy average. Use it as a prior, not a signal.
"""
