"""Trend Detection — SMA / Kalman 2D / UKF trend filters with linear forecasts,
across benchmarks, portfolio positions, the aggregate NAV, and custom tickers.
Each tab runs as a background callback."""
import traceback

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots

from capital.analytics.credit_liquidity import CreditFilter, LiquidityFilter, StressScore
from capital.analytics.trend import (
    GJRGarch,
    KalmanFilter2D,
    SMAFilter,
    UKFFilter,
    forecast_linear,
)
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG

dash.register_page(
    __name__, path="/trend", name="Trend", order=8,
    description="Kalman/UKF/SMA trend filters with regime detection and forecasts.",
)

BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "EWG": "MSCI Germany",
    "EWQ": "MSCI France",
    "TLT": "20Y Treasury",
    "GLD": "Gold",
}

_COLORS = {"sma": "#F59E0B", "kalman": "#3B82F6", "ukf": "#A855F7"}


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ── Data loaders ──────────────────────────────────────────────────────────────

@cached_by_version
def _load_benchmark_prices(start: str, end: str) -> pd.DataFrame:
    cols = {}
    for ticker in BENCHMARKS:
        s = loaders.get_market_prices(ticker).loc[start:end]
        if not s.empty:
            cols[ticker] = s
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


@cached_by_version
def _load_portfolio_positions(start: str, end: str) -> dict:
    """Prices for active holdings from the EOD store."""
    sm = loaders.get_security_master()
    wh = loaders.get_daily_weightings_history()
    if not wh.empty:
        current_symbols = set(wh[wh["date"] == wh["date"].max()]["symbol"].tolist())
        sm = sm[sm["ticker"].isin(current_symbols)]
    sm = sm[sm["asset_type"] != "INDEX"]
    out = {}
    for _, row in sm.iterrows():
        eod = loaders.get_eod_prices(row["security_id"])
        if eod.empty:
            continue
        eod = eod.copy()
        eod["date"] = pd.to_datetime(eod["date"])
        col = "adj_close" if "adj_close" in eod.columns else "close"
        s = eod.set_index("date")[col].loc[start:end].dropna().sort_index()
        if len(s) >= 20:
            out[row["ticker"]] = s.rename(row["ticker"])
    return out


@cached_by_version
def _load_portfolio_aggregate(start: str, end: str) -> pd.Series:
    pb = loaders.get_portfolio_and_benchmarks()
    port = pb[pb["ticker"] == "PORTFOLIO"].set_index("date")["index_value"]
    port.index = pd.to_datetime(port.index)
    return port.loc[start:end].sort_index().rename("Portfolio NAV")


@cached_by_version
def _stress_series(start: str, end: str) -> pd.Series:
    """Composite stress score (same construction as the Credit & Liquidity page)."""
    try:
        hy = loaders.get_fred_series("BAMLH0A0HYM2").loc[start:end]
        spy = loaders.get_market_ohlcv("SPY")
        signals = {}
        if not hy.empty:
            signals.update(CreditFilter(hy, window=504).run())
        close = spy["close"].loc[start:end].dropna()
        volume = spy["volume"].loc[start:end].dropna()
        if not close.empty and not volume.empty:
            signals.update(LiquidityFilter(close, volume, window=504).run())
        if len(signals) >= 2:
            z, _ = StressScore(signals).run()
            return z
    except Exception:
        pass
    return pd.Series(dtype=float)


# ── Filters ───────────────────────────────────────────────────────────────────

def _run_filters(name: str, prices: pd.Series, stress: pd.Series, cfg: dict) -> dict:
    result: dict = {"name": name, "prices": prices}
    if cfg["use_sma"]:
        try:
            result["sma"] = SMAFilter(window=cfg["sma_window"]).run(prices)
        except Exception:
            pass
    if cfg["use_kalman"]:
        try:
            result["kalman"] = KalmanFilter2D().run(prices)
        except Exception:
            pass
    if cfg["use_ukf"]:
        vol = pd.Series(dtype=float)
        if cfg["use_gjr"]:
            try:
                vol = GJRGarch().fit(prices)
            except Exception:
                pass
        try:
            result["ukf"] = UKFFilter().run(prices, vol=vol, stress=stress)
        except Exception:
            pass
    result["forecasts"] = {}
    for key in ("sma", "kalman", "ukf"):
        if key in result:
            try:
                result["forecasts"][key] = forecast_linear(result[key], cfg["horizon"])
            except Exception:
                pass
    return result


# ── Chart ─────────────────────────────────────────────────────────────────────

def _trend_chart(result: dict, labels: dict, height: int = 480) -> go.Figure:
    prices = result["prices"].dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.70, 0.30], vertical_spacing=0.06)
    fig.layout.annotations = []
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values, mode="lines",
        line=dict(width=1.2, color="rgba(148,163,184,0.45)"),
        showlegend=False, hovertemplate="Price: %{y:.2f}<extra></extra>"), row=1, col=1)

    for key, color in _COLORS.items():
        if key not in result:
            continue
        df = result[key].dropna()
        if df.empty:
            continue
        lbl = labels[key]
        for is_bull, seg_color in ((True, "#10B981"), (False, "#EF4444")):
            seg = df["trend"].where(df["slope"] > 0 if is_bull else df["slope"] <= 0)
            fig.add_trace(go.Scatter(
                x=seg.index, y=seg.values, mode="lines",
                line=dict(width=2.5, color=seg_color),
                showlegend=False, hoverinfo="skip", connectgaps=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[df["trend"].iloc[-1]], mode="markers",
            marker=dict(size=8, color=color, symbol="circle"),
            showlegend=False,
            hovertemplate=f"{lbl}: %{{y:.2f}}<extra></extra>"), row=1, col=1)

        if key in result.get("forecasts", {}):
            fc = result["forecasts"][key]
            fig.add_trace(go.Scatter(
                x=list(fc.index) + list(fc.index[::-1]),
                y=list(fc["upper_1s"]) + list(fc["lower_1s"][::-1]),
                fill="toself", fillcolor=_hex_rgba(color, 0.10),
                line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=fc.index, y=fc["trend"].values, mode="lines",
                line=dict(width=1.8, color=color, dash="dash"),
                showlegend=False,
                hovertemplate=f"{lbl} fcast: %{{y:.2f}}<extra></extra>"), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df.index, y=df["slope"].values, mode="lines",
            line=dict(width=1.5, color=color), showlegend=False,
            hovertemplate=f"{lbl} slope: %{{y:.5f}}<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=np.where(df["slope"].values > 0, df["slope"].values, 0),
            fill="tozeroy", mode="none", fillcolor="rgba(16,185,129,0.10)",
            showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=np.where(df["slope"].values <= 0, df["slope"].values, 0),
            fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.10)",
            showlegend=False, hoverinfo="skip"), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1, line_width=0.8,
                  line_color="rgba(255,255,255,0.18)", line_dash="dot")
    fig.update_layout(
        height=height,
        title=dict(text=f"<b>{result['name']}</b>", x=0, xanchor="left", font=dict(size=15)),
        showlegend=False,
        margin=dict(l=55, r=30, t=44, b=40),
        hovermode="x unified")
    fig.update_yaxes(tickfont=dict(size=10), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=10), zeroline=False, row=2, col=1)
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10), row=2, col=1)
    return fig


def _render_results(all_results: dict, cfg: dict):
    if not all_results:
        return dmc.Alert("No results to display.", color="yellow", variant="light")

    labels = {"sma": f"SMA-{cfg['sma_window']}", "kalman": "Kalman 2D", "ukf": "UKF"}
    key_parts = []
    if cfg["use_sma"]:
        key_parts.append(f"<span style='color:#F59E0B'>●</span> SMA-{cfg['sma_window']}")
    if cfg["use_kalman"]:
        key_parts.append("<span style='color:#3B82F6'>●</span> Kalman 2D")
    if cfg["use_ukf"]:
        key_parts.append("<span style='color:#A855F7'>●</span> UKF")
    key_parts += ["<span style='color:#10B981'>━</span> Bullish",
                  "<span style='color:#EF4444'>━</span> Bearish",
                  "<span style='color:#94A3B8'>╌</span> Forecast"]
    legend = dcc.Markdown("  ·  ".join(key_parts), dangerously_allow_html=True)

    rows = []
    for asset_name, r in all_results.items():
        for key, label in labels.items():
            if key not in r:
                continue
            df = r[key].dropna()
            if df.empty:
                continue
            slope_now = float(df["slope"].iloc[-1])
            trend_now = float(df["trend"].iloc[-1])
            price_now = float(r["prices"].dropna().iloc[-1])
            row = {"Asset": asset_name, "Filter": label,
                   "Regime": "↑ Bullish" if slope_now > 0 else "↓ Bearish",
                   "Price": f"{price_now:.2f}", "Trend": f"{trend_now:.2f}",
                   "Slope": f"{slope_now:+.5f}"}
            if key in r.get("forecasts", {}):
                fc_end = float(r["forecasts"][key]["trend"].iloc[-1])
                row["Fcast end"] = f"{fc_end:.2f}"
                row["Exp. Δ"] = f"{fc_end - trend_now:+.2f}"
            rows.append(row)

    children = [legend]
    if rows:
        children.append(ui.df_table(pd.DataFrame(rows).fillna("")))

    names = list(all_results.keys())
    if len(names) == 1:
        children.append(dcc.Graph(figure=_trend_chart(all_results[names[0]], labels, 520),
                                  config=GRAPH_CONFIG))
    else:
        charts = [dcc.Graph(figure=_trend_chart(all_results[nm], labels, 420),
                            config=GRAPH_CONFIG) for nm in names]
        children.append(dmc.SimpleGrid(charts, cols={"base": 1, "lg": 2}))
    return dmc.Stack(children, gap="md")


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    today = pd.Timestamp.today().date()
    return dmc.Stack([
        ui.page_title(
            "Trend Detection",
            "Runs Kalman filter and moving average models on each asset to extract "
            "the underlying price trend and estimate whether momentum is building or "
            "fading. The UKF variant adapts its noise model using GJR GARCH "
            "volatility so it reacts faster in stressed markets."),
        dmc.Group([
            dmc.DatePickerInput(id="trend-start", label="Start date", value="2022-01-01", w=150),
            dmc.DatePickerInput(id="trend-end", label="End date", value=today.isoformat(), w=150),
            dmc.CheckboxGroup(
                id="trend-filters", label="Filters",
                value=["sma", "kalman", "ukf"],
                children=dmc.Group([
                    dmc.Checkbox(label="SMA", value="sma"),
                    dmc.Checkbox(label="Kalman 2D", value="kalman"),
                    dmc.Checkbox(label="UKF", value="ukf"),
                ], mt=6)),
            dmc.Box(dmc.Stack([
                dmc.Text("SMA window", size="sm", fw=500),
                dmc.Slider(id="trend-sma-window", min=10, max=200, value=50,
                           marks=[{"value": v, "label": str(v)} for v in (50, 100, 200)], w=180),
            ], gap=4)),
            dmc.Box(dmc.Stack([
                dmc.Text("Forecast horizon (days)", size="sm", fw=500),
                dmc.Slider(id="trend-horizon", min=5, max=60, value=20,
                           marks=[{"value": v, "label": str(v)} for v in (20, 40, 60)], w=180),
            ], gap=4)),
            dmc.CheckboxGroup(
                id="trend-extras", label="Extras",
                value=["gjr", "stress"],
                children=dmc.Group([
                    dmc.Checkbox(label="GJR-GARCH vol", value="gjr"),
                    dmc.Checkbox(label="Credit/Liquidity stress", value="stress"),
                ], mt=6)),
        ], align="end", gap="lg"),
        dmc.Divider(mt="sm"),
        dmc.Tabs([
            dmc.TabsList([
                dmc.TabsTab("Benchmarks", value="bench"),
                dmc.TabsTab("Portfolio positions", value="pos"),
                dmc.TabsTab("Portfolio aggregate", value="agg"),
                dmc.TabsTab("Custom search", value="custom"),
            ]),
            dmc.TabsPanel(dmc.Stack([
                dmc.Button("Run benchmarks", id="trend-run-bench", mt="sm", w=200),
                dcc.Loading(dmc.Box(id="trend-bench-results")),
            ]), value="bench"),
            dmc.TabsPanel(dmc.Stack([
                dmc.Button("Load positions", id="trend-run-pos", mt="sm", w=200),
                dcc.Loading(dmc.Box(id="trend-pos-results")),
            ]), value="pos"),
            dmc.TabsPanel(dmc.Stack([
                dmc.Button("Run aggregate", id="trend-run-agg", mt="sm", w=200),
                dcc.Loading(dmc.Box(id="trend-agg-results")),
            ]), value="agg"),
            dmc.TabsPanel(dmc.Stack([
                dmc.Group([
                    dmc.TextInput(id="trend-custom-ticker", label="Ticker / RIC",
                                  placeholder="e.g. AAPL, VOD.L", w=220),
                    dmc.TextInput(id="trend-custom-label", label="Display name (optional)",
                                  placeholder="e.g. Apple", w=200),
                    dmc.Button("Run", id="trend-run-custom", mt=22),
                ], align="end", mt="sm"),
                dcc.Loading(dmc.Box(id="trend-custom-results")),
            ]), value="custom"),
        ], value="bench"),
    ])


def _cfg(filters, sma_window, horizon, extras) -> dict:
    return {
        "use_sma": "sma" in (filters or []),
        "use_kalman": "kalman" in (filters or []),
        "use_ukf": "ukf" in (filters or []),
        "use_gjr": "gjr" in (extras or []),
        "use_stress": "stress" in (extras or []),
        "sma_window": int(sma_window),
        "horizon": int(horizon),
    }


_STATES = [
    State("trend-start", "value"), State("trend-end", "value"),
    State("trend-filters", "value"), State("trend-sma-window", "value"),
    State("trend-horizon", "value"), State("trend-extras", "value"),
]


def _stress_for(cfg, start, end):
    return _stress_series(str(start), str(end)) if cfg["use_stress"] else pd.Series(dtype=float)


@callback(
    Output("trend-bench-results", "children"),
    Input("trend-run-bench", "n_clicks"), *_STATES,
    background=True,
    running=[(Output("trend-run-bench", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_bench(n, start, end, filters, sma_window, horizon, extras):
    try:
        cfg = _cfg(filters, sma_window, horizon, extras)
        stress = _stress_for(cfg, start, end)
        prices_df = _load_benchmark_prices(str(start), str(end))
        if prices_df.empty:
            return dmc.Alert("Could not load benchmark prices.", color="red", variant="light")
        results = {}
        for ticker, label in BENCHMARKS.items():
            if ticker not in prices_df.columns:
                continue
            s = prices_df[ticker].dropna()
            if len(s) >= 30:
                results[label] = _run_filters(label, s, stress, cfg)
        return _render_results(results, cfg)
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True), color="red",
                         variant="light", title="Trend run failed")


@callback(
    Output("trend-pos-results", "children"),
    Input("trend-run-pos", "n_clicks"), *_STATES,
    background=True,
    running=[(Output("trend-run-pos", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_positions(n, start, end, filters, sma_window, horizon, extras):
    try:
        cfg = _cfg(filters, sma_window, horizon, extras)
        stress = _stress_for(cfg, start, end)
        pos_map = _load_portfolio_positions(str(start), str(end))
        if not pos_map:
            return dmc.Alert("No positions loaded — check the EOD store.",
                             color="red", variant="light")
        results = {t: _run_filters(t, s, stress, cfg) for t, s in pos_map.items()}
        return _render_results(results, cfg)
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True), color="red",
                         variant="light", title="Trend run failed")


@callback(
    Output("trend-agg-results", "children"),
    Input("trend-run-agg", "n_clicks"), *_STATES,
    background=True,
    running=[(Output("trend-run-agg", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_aggregate(n, start, end, filters, sma_window, horizon, extras):
    try:
        cfg = _cfg(filters, sma_window, horizon, extras)
        stress = _stress_for(cfg, start, end)
        agg = _load_portfolio_aggregate(str(start), str(end))
        if agg.empty:
            return dmc.Alert("No portfolio NAV data found.", color="red", variant="light")
        results = {"AIC Portfolio": _run_filters("AIC Portfolio", agg, stress, cfg)}
        return _render_results(results, cfg)
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True), color="red",
                         variant="light", title="Trend run failed")


@callback(
    Output("trend-custom-results", "children"),
    Input("trend-run-custom", "n_clicks"),
    State("trend-custom-ticker", "value"),
    State("trend-custom-label", "value"), *_STATES,
    background=True,
    running=[(Output("trend-run-custom", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_custom(n, ticker, label, start, end, filters, sma_window, horizon, extras):
    if not (ticker or "").strip():
        return dmc.Alert("Enter a ticker first.", color="blue", variant="light")
    try:
        cfg = _cfg(filters, sma_window, horizon, extras)
        stress = _stress_for(cfg, start, end)
        ticker = ticker.strip()
        label = (label or "").strip() or ticker

        # Security master first (LSEG-ingested history), market store / yfinance fallback
        s = pd.Series(dtype=float)
        sm = loaders.get_security_master()
        match = sm[(sm["ric"] == ticker) | (sm["ticker"] == ticker)]
        if not match.empty:
            eod = loaders.get_eod_prices(match.iloc[0]["security_id"])
            if not eod.empty:
                eod = eod.copy()
                eod["date"] = pd.to_datetime(eod["date"])
                col = "adj_close" if "adj_close" in eod.columns else "close"
                s = eod.set_index("date")[col].loc[str(start):str(end)].dropna().sort_index()
        if s.empty:
            s = loaders.get_market_prices(ticker).loc[str(start):str(end)]

        if s.empty or len(s) < 30:
            return dmc.Alert(f"Not enough data for {ticker} (need ≥ 30 trading days).",
                             color="yellow", variant="light")
        results = {label: _run_filters(label, s.rename(label), stress, cfg)}
        return _render_results(results, cfg)
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True), color="red",
                         variant="light", title="Trend run failed")
