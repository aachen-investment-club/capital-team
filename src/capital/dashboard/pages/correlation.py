"""Correlation Analysis — rolling and partial correlations for fixed market pairs,
plus a custom explorer."""
import traceback

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc

from capital.analytics.correlation import PartialCorr, RollingCorr
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG

dash.register_page(
    __name__, path="/correlation", name="Correlation", order=6,
    description="Rolling and partial correlations across key market pairs.",
)

FIXED_PAIRS = [
    ("SPY", "IWM", None, "Breadth", "S&P 500 vs Russell 2000 — market internals"),
    ("SPY", "TLT", None, "Macro", "Equities vs Long Treasury — risk-on/off"),
    ("SPY", "VIX", None, "Stress", "S&P 500 vs VIX — fear gauge"),
    ("HYG", "TLT", None, "Liquidity", "High-Yield vs Treasury — credit/duration spread"),
    ("IWM", "HYG", "SPY", "Partial: Small-cap ↔ Credit",
     "IWM vs HYG controlling for SPY — idiosyncratic credit sensitivity"),
]

CHART_ORDER = ["Macro", "Stress", "Breadth", "Liquidity", "Partial: Small-cap ↔ Credit"]


@cached_by_version
def _load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Daily closes for the correlation tickers (store, yfinance fallback)."""
    cols = {}
    for ticker in tickers:
        try:
            s = loaders.get_market_prices(ticker).loc[start:end]
            if not s.empty:
                cols[ticker] = s.rename(ticker)
        except Exception:
            continue
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).apply(pd.to_numeric, errors="coerce").dropna()


def _line(series: pd.Series, title: str, subtitle: str,
          yrange=(-1.05, 1.05), zero_line: bool = True, height: int = 300) -> go.Figure:
    s = series.dropna()
    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(width=2, color="#3B82F6"),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>"))
    if zero_line:
        fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s.values >= 0, s.values, 0),
        fill="tozeroy", mode="none", fillcolor="rgba(16,185,129,0.15)", showlegend=False))
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s.values < 0, s.values, 0),
        fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.15)", showlegend=False))
    title_text = (f"<b>{title}</b>"
                  f"<br><span style='font-size:11px;color:#94A3B8'>{subtitle}</span>")
    fig.update_layout(
        title=dict(text=title_text, x=0, xanchor="left"),
        height=height,
        yaxis=dict(range=list(yrange), tickformat=".2f", zeroline=False),
        xaxis=dict(showgrid=False),
        margin=dict(l=50, r=20, t=72, b=40),
        showlegend=False)
    return fig


def layout():
    today = pd.Timestamp.today().date()
    all_tickers = sorted({t for pair in FIXED_PAIRS for t in pair[:3] if t})
    return dmc.Stack([
        ui.page_title(
            "Correlation Analysis",
            "Measures how much our assets move together on a rolling basis. Useful "
            "for spotting when supposed diversifiers are no longer diversifying, or "
            "when a macro regime shift is causing previously uncorrelated assets to "
            "move in lockstep."),
        dmc.Group([
            dmc.DatePickerInput(id="corr-start", label="Start date", value="2020-01-01", w=160),
            dmc.DatePickerInput(id="corr-end", label="End date", value=today.isoformat(), w=160),
            dmc.Box(dmc.Stack([
                dmc.Text("Rolling window (days)", size="sm", fw=500),
                dmc.Slider(id="corr-window", min=10, max=120, value=30,
                           marks=[{"value": v, "label": str(v)} for v in (30, 60, 90, 120)], w=200),
            ], gap=4)),
            dmc.Box(dmc.Stack([
                dmc.Text("Smoothing window (days)", size="sm", fw=500),
                dmc.Slider(id="corr-smooth", min=5, max=40, value=10,
                           marks=[{"value": v, "label": str(v)} for v in (10, 20, 30, 40)], w=200),
            ], gap=4)),
            dmc.Button("Run", id="corr-run", mt=22),
        ], align="end", gap="lg"),
        dcc.Loading(dmc.Box(
            dmc.Alert("Set a date range and click Run.", color="blue", variant="light"),
            id="corr-results", mt="md")),

        dmc.Divider(mt="lg"),
        dmc.Accordion([dmc.AccordionItem([
            dmc.AccordionControl("Custom correlation explorer"),
            dmc.AccordionPanel(dmc.Stack([
                dmc.Text("Computes rolling (or partial, with a control) correlation of "
                         "X and Y. Extra tickers are fetched via yfinance if unknown.",
                         size="sm", c="dimmed"),
                dmc.Group([
                    dmc.Select(id="corr-x", label="X", data=all_tickers, value="SPY", w=130),
                    dmc.Select(id="corr-y", label="Y", data=all_tickers, value="TLT", w=130),
                    dmc.Select(id="corr-ctrl", label="Control",
                               data=["(none)"] + all_tickers, value="(none)", w=130),
                    dmc.NumberInput(id="corr-custom-window", label="Window (days)",
                                    min=10, max=252, value=60, w=140),
                    dmc.TextInput(id="corr-extra", label="Extra tickers (comma-separated)",
                                  placeholder="e.g. XLU, GLD", w=240),
                    dmc.Button("Compute", id="corr-custom-run", mt=22),
                ], align="end"),
                dcc.Loading(dmc.Box(id="corr-custom-result")),
            ]))], value="custom")]),
    ])


def _monitor_block(start, end, window_corr, window_smooth) -> dmc.Stack:
    all_tickers = tuple(sorted({t for pair in FIXED_PAIRS for t in pair[:3] if t}))
    prices = _load_prices(all_tickers, str(start), str(end))
    if prices.empty:
        return dmc.Alert("No price data returned — check the market-data ingest.",
                         color="red", variant="light")

    results = {}
    for x, y, ctrl, label, desc in FIXED_PAIRS:
        if x not in prices.columns or y not in prices.columns:
            continue
        if ctrl is None:
            rc = RollingCorr(prices[x], prices[y], window_corr, window_smooth)
            results[label] = {"series": rc.run_rho(), "desc": desc,
                              "type": "rolling", "x": x, "y": y, "ctrl": None}
        else:
            if ctrl not in prices.columns:
                continue
            pc = PartialCorr(prices[x], prices[y], prices[ctrl])
            results[label] = {"series": pc.rolling(window=window_corr), "desc": desc,
                              "type": "partial", "x": x, "y": y, "ctrl": ctrl}

    def _chart(label, height=320):
        r = results[label]
        tag = "(partial)" if r["type"] == "partial" else ""
        ctrl_note = f" | control: {r['ctrl']}" if r["ctrl"] else ""
        return dcc.Graph(figure=_line(r["series"], f"{r['x']} ↔ {r['y']} {tag}",
                                      r["desc"] + ctrl_note, height=height),
                         config=GRAPH_CONFIG)

    children = [
        ui.section("Market Correlation Monitor"),
        dmc.Text(f"{start} → {end}  ·  rolling {window_corr}d, smooth {window_smooth}d",
                 size="sm", c="dimmed"),
        dmc.SimpleGrid([_chart(k) for k in CHART_ORDER[:2] if k in results],
                       cols={"base": 1, "lg": 2}),
        dmc.SimpleGrid([_chart(k) for k in CHART_ORDER[2:4] if k in results],
                       cols={"base": 1, "lg": 2}),
    ]
    if "Partial: Small-cap ↔ Credit" in results:
        r = results["Partial: Small-cap ↔ Credit"]
        children.append(dcc.Graph(
            figure=_line(r["series"],
                         f"{r['x']} ↔ {r['y']}  (partial, controlling for {r['ctrl']})",
                         r["desc"], height=280),
            config=GRAPH_CONFIG))
    return dmc.Stack(children, gap="md")


@callback(
    Output("corr-results", "children"),
    Input("corr-run", "n_clicks"),
    State("corr-start", "value"),
    State("corr-end", "value"),
    State("corr-window", "value"),
    State("corr-smooth", "value"),
    background=True,
    running=[(Output("corr-run", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_correlation(n_clicks, start, end, window_corr, window_smooth):
    try:
        return _monitor_block(start, end, window_corr, window_smooth)
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True),
                         color="red", variant="light", title="Correlation run failed")


@callback(
    Output("corr-custom-result", "children"),
    Output("corr-x", "data"),
    Output("corr-y", "data"),
    Output("corr-ctrl", "data"),
    Input("corr-custom-run", "n_clicks"),
    State("corr-x", "value"),
    State("corr-y", "value"),
    State("corr-ctrl", "value"),
    State("corr-custom-window", "value"),
    State("corr-extra", "value"),
    State("corr-start", "value"),
    State("corr-end", "value"),
    State("corr-smooth", "value"),
    prevent_initial_call=True,
)
def run_custom(n_clicks, x, y, ctrl, window, extra, start, end, smooth):
    base = tuple(sorted({t for pair in FIXED_PAIRS for t in pair[:3] if t}))
    extras = tuple(t.strip().upper() for t in (extra or "").split(",") if t.strip())
    prices = _load_prices(base + extras, str(start), str(end))
    available = sorted(prices.columns.tolist())
    opts = available
    ctrl_opts = ["(none)"] + available

    if x not in prices.columns or y not in prices.columns:
        return (dmc.Alert("One or both tickers not available in loaded data.",
                          color="yellow", variant="light"), opts, opts, ctrl_opts)
    if x == y:
        return (dmc.Alert("X and Y must be different.", color="yellow", variant="light"),
                opts, opts, ctrl_opts)

    if ctrl == "(none)" or not ctrl:
        s = RollingCorr(prices[x], prices[y], int(window), int(smooth)).run_rho()
        title = f"{x} ↔ {y}  (rolling)"
        sub = f"Rolling {window}d correlation"
    else:
        if ctrl not in prices.columns:
            return (dmc.Alert(f"Control ticker '{ctrl}' not in data.",
                              color="yellow", variant="light"), opts, opts, ctrl_opts)
        s = PartialCorr(prices[x], prices[y], prices[ctrl]).rolling(window=int(window))
        title = f"{x} ↔ {y}  (partial, control: {ctrl})"
        sub = f"Partial correlation removing {ctrl}'s influence · window={window}d"

    scalar = float(s.dropna().iloc[-1]) if not s.dropna().empty else float("nan")
    block = dmc.Stack([
        dmc.Paper([dmc.Text("Latest correlation", className="metric-label"),
                   dmc.Text(f"{scalar:+.3f}", className="metric-value")],
                  className="metric-card", w=220),
        dcc.Graph(figure=_line(s, title, sub, height=340), config=GRAPH_CONFIG),
    ])
    return block, opts, opts, ctrl_opts
