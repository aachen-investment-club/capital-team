"""Credit & Liquidity Monitor — HY-spread credit stress, SPY microstructure
liquidity, and the combined stress score."""
import traceback

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc

from capital.analytics.credit_liquidity import CreditFilter, LiquidityFilter, StressScore
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG

dash.register_page(
    __name__, path="/credit-liquidity", name="Credit & Liquidity", order=7,
    description="Credit stress (HY OAS) and market liquidity early-warning signals.",
)


@cached_by_version
def _fetch(start: str, end: str) -> dict:
    errors = []
    hy_oas = pd.Series(dtype=float)
    try:
        hy_oas = loaders.get_fred_series("BAMLH0A0HYM2").loc[start:end]
    except Exception as e:
        errors.append(f"HY OAS: {e}")

    spy_close = spy_volume = pd.Series(dtype=float)
    try:
        spy_ohlcv = loaders.get_market_ohlcv("SPY")
        spy_close = spy_ohlcv["close"].loc[start:end].dropna()
        spy_volume = spy_ohlcv["volume"].loc[start:end].dropna()
    except Exception as e:
        errors.append(f"SPY: {e}")

    spx = pd.Series(dtype=float)
    try:
        spx = loaders.get_market_prices("SPY").loc[start:end]   # SPY as SPX proxy
    except Exception as e:
        errors.append(f"SPX: {e}")

    return {"hy_oas": hy_oas, "spy_close": spy_close,
            "spy_volume": spy_volume, "spx": spx, "errors": errors}


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _signal_chart(signal: pd.Series, title: str, color: str, height: int = 300) -> go.Figure:
    s = signal.dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s >= 0, s.values, 0),
        fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.15)", showlegend=False))
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s < 0, s.values, 0),
        fill="tozeroy", mode="none", fillcolor="rgba(16,185,129,0.15)", showlegend=False))
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(width=2, color=color), name=title,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"))
    for lvl in (1.5, -1.5):
        fig.add_hline(y=lvl, line_width=0.8,
                      line_color="rgba(255,255,255,0.2)", line_dash="dot")
    fig.update_layout(
        height=height,
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=14)),
        margin=dict(l=50, r=30, t=48, b=36),
        showlegend=False,
        yaxis=dict(title="Z-score", zeroline=False, tickfont=dict(size=11)))
    return fig


def _contiguous(mask: pd.Series):
    arr = mask.values
    in_run, start = False, 0
    for i, v in enumerate(arr):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            yield start, i
            in_run = False
    if in_run:
        yield start, len(arr)


def _stress_chart(stress_z: pd.Series, thr_hi: float, height: int = 380) -> go.Figure:
    z = stress_z.dropna()
    fig = go.Figure()
    is_extreme = z >= thr_hi
    for start_i, end_i in _contiguous(is_extreme):
        fig.add_vrect(x0=z.index[start_i], x1=z.index[min(end_i, len(z.index) - 1)],
                      fillcolor="rgba(239,68,68,0.18)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=z.index, y=np.where(z >= 0, z.values, 0),
        fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.12)", showlegend=False))
    fig.add_trace(go.Scatter(
        x=z.index, y=np.where(z < 0, z.values, 0),
        fill="tozeroy", mode="none", fillcolor="rgba(16,185,129,0.12)", showlegend=False))
    fig.add_trace(go.Scatter(
        x=z.index, y=z.values, mode="lines",
        line=dict(width=2.5, color="#F59E0B"), name="Stress score",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"))
    if not np.isnan(thr_hi):
        fig.add_hline(y=thr_hi, line_width=1.2,
                      line_color="rgba(239,68,68,0.55)", line_dash="dash")
    fig.update_layout(
        height=height,
        title=dict(text=("<b>Composite Stress Score</b>"
                         "  <span style='font-size:12px;color:#94A3B8'>"
                         "= |0.5 × (credit shock + liquidity shock)| — "
                         "<span style='color:#EF4444'>red bands = top/bottom 5% of "
                         "historical days</span></span>"),
                   x=0, xanchor="left"),
        showlegend=False,
        margin=dict(l=50, r=30, t=72, b=36),
        yaxis=dict(title="Stress Z-score", zeroline=False))
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    today = pd.Timestamp.today().date()
    return dmc.Stack([
        ui.page_title(
            "Credit & Liquidity Monitor",
            "Tracks two early warning risk signals: credit stress (via high yield "
            "spreads from FRED) and market liquidity (via SPY microstructure). "
            "Combines them into a single stress score. Spikes here tend to precede "
            "broader market drawdowns."),
        dmc.Group([
            dmc.DatePickerInput(id="cl-start", label="Start date", value="2016-01-01", w=160),
            dmc.DatePickerInput(id="cl-end", label="End date", value=today.isoformat(), w=160),
            dmc.Box(dmc.Stack([
                dmc.Text("Z-score window (trading days)", size="sm", fw=500),
                dmc.Slider(id="cl-window", min=126, max=756, value=504, step=21,
                           marks=[{"value": v, "label": str(v)} for v in (126, 252, 504, 756)],
                           w=280),
            ], gap=4)),
            dmc.Button("Run", id="cl-run", mt=22),
        ], align="end", gap="lg"),
        dcc.Loading(dmc.Box(
            dmc.Alert("Set a date range and click Run.", color="blue", variant="light"),
            id="cl-results", mt="md")),
    ])


PAIRS = [
    (("credit_level", "Credit Level", "#EF4444"),
     ("liquidity_level", "Liquidity Level", "#3B82F6")),
    (("credit_shock", "Credit Shock", "#F97316"),
     ("liquidity_shock", "Liquidity Shock", "#8B5CF6")),
    (("credit_accel", "Credit Acceleration", "#FBBF24"),
     ("liquidity_accel", "Liquidity Acceleration", "#EC4899")),
]

ROW_DESCS = [
    "Z-score of the raw level vs {w}-day rolling mean — slow-moving regime indicator",
    "Z-score of the smoothed daily change — picks up rapid deterioration",
    "Z-score of the rate-of-change of the shock — early turning-point signal",
]


def _metric(label, value):
    return dmc.Paper([dmc.Text(label, className="metric-label"),
                      dmc.Text(value, className="metric-value")], className="metric-card")


def _results_block(start, end, window) -> dmc.Stack:
    raw = _fetch(str(start), str(end))
    children = [dmc.Alert(e, color="yellow", variant="light") for e in raw["errors"]]

    if raw["hy_oas"].empty and raw["spy_close"].empty:
        return dmc.Alert("No data loaded — check market-data and FRED ingest.",
                         color="red", variant="light")

    signals = {}
    if not raw["hy_oas"].empty:
        signals.update(CreditFilter(raw["hy_oas"], window=window).run())
    if not raw["spy_close"].empty and not raw["spy_volume"].empty:
        signals.update(LiquidityFilter(raw["spy_close"], raw["spy_volume"], window=window).run())

    stress_z = pd.Series(dtype=float)
    if len(signals) >= 2:
        stress_z, _ = StressScore(signals).run()

    if not stress_z.empty:
        threshold_hi = float(stress_z.dropna().quantile(0.95))
        latest_z = float(stress_z.dropna().iloc[-1])
        pct_rank = float((stress_z.dropna() <= latest_z).mean()) * 100
        children += [
            dmc.SimpleGrid([
                _metric("Stress Z-score", f"{latest_z:+.2f}"),
                _metric("Percentile rank", f"{pct_rank:.0f}th"),
                _metric("Top-5% stress",
                        "YES ⚠" if latest_z >= threshold_hi else "NO"),
                _metric("95th pct threshold", f"{threshold_hi:.2f}"),
            ], cols={"base": 2, "sm": 4}),
            dcc.Graph(figure=_stress_chart(stress_z, threshold_hi, height=400),
                      config=GRAPH_CONFIG),
            dmc.Divider(mt="md"),
        ]

    children.append(dmc.SimpleGrid([
        dmc.Title("Credit  (HY OAS, FRED)", order=4, c="#0C1E40"),
        dmc.Title("Liquidity  (SPY Amihud + Volume)", order=4, c="#0C1E40"),
    ], cols=2))

    for i, ((ck, cl, cc), (lk, ll, lc)) in enumerate(PAIRS):
        left = (dcc.Graph(figure=_signal_chart(signals[ck], cl, cc, height=280),
                          config=GRAPH_CONFIG)
                if ck in signals else dmc.Alert(f"{cl}: no data", color="blue", variant="light"))
        right = (dcc.Graph(figure=_signal_chart(signals[lk], ll, lc, height=280),
                           config=GRAPH_CONFIG)
                 if lk in signals else dmc.Alert(f"{ll}: no data", color="blue", variant="light"))
        children.append(dmc.SimpleGrid([left, right], cols={"base": 1, "lg": 2}))
        children.append(dmc.Text(ROW_DESCS[i].replace("{w}", str(window)),
                                 size="sm", c="dimmed"))

    if not raw["hy_oas"].empty:
        hy = raw["hy_oas"]
        fig_raw = go.Figure(go.Scatter(
            x=hy.index, y=hy.values, mode="lines",
            line=dict(width=1.5, color="#EF4444"),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.0f} bps<extra></extra>"))
        fig_raw.update_layout(
            height=200, title="<b>ICE BofA US HY Option-Adjusted Spread</b>",
            yaxis_title="bps", margin=dict(l=50, r=20, t=48, b=32))
        children.append(dmc.Accordion([dmc.AccordionItem([
            dmc.AccordionControl("Raw HY OAS (basis points)"),
            dmc.AccordionPanel(dcc.Graph(figure=fig_raw, config=GRAPH_CONFIG)),
        ], value="raw")]))

    return dmc.Stack(children, gap="md")


@callback(
    Output("cl-results", "children"),
    Input("cl-run", "n_clicks"),
    State("cl-start", "value"),
    State("cl-end", "value"),
    State("cl-window", "value"),
    background=True,
    running=[(Output("cl-run", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_credit_liquidity(n_clicks, start, end, window):
    try:
        return _results_block(start, end, int(window))
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True),
                         color="red", variant="light", title="Run failed")
