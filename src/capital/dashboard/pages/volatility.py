"""Volatility — GARCH-family backtest, model rankings, and forward forecast.
Model fitting runs as a background callback."""
import traceback
from datetime import date

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc

from capital.analytics.volatility import (
    ALL_MODELS,
    ann_vol_from_var,
    evaluate_models,
    forecast_ahead,
    log_returns_pct,
    run_backtest,
)
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG

dash.register_page(
    __name__, path="/volatility", name="Volatility", order=4,
    description="GARCH-family volatility models: backtest, rankings, forecast.",
)

_DATE_MIN = date(2015, 1, 1)
_BENCH_TICKERS = {"SPX", "MSCI_WORLD", "MSCI_EUROPE", "60_40"}
_MODEL_COLOR = {"GARCH": "#2563EB", "GJR-GARCH": "#10B981"}


def _model_color(name: str) -> str:
    return _MODEL_COLOR.get(name, "#94A3B8")


def _yesterday() -> date:
    return date.today() - pd.Timedelta(days=1)


@cached_by_version
def _get_returns(asset: str) -> pd.Series | None:
    """Log returns (%) for any asset: EOD store first, benchmark JSON fallback."""
    if asset == "Portfolio":
        df = loaders.get_portfolio_and_benchmarks()
        port = df[df["ticker"] == "PORTFOLIO"][["date", "daily_return"]].dropna()
        if port.empty:
            return None
        s = port.set_index("date")["daily_return"] * 100
        s.index = pd.to_datetime(s.index)
        return s.sort_index().dropna()

    master = loaders.get_security_master()
    row = master[master["ticker"] == asset]
    if not row.empty:
        eod = loaders.get_eod_prices(row.iloc[0]["security_id"])
        if not eod.empty:
            price_col = "adj_close" if "adj_close" in eod.columns else "close"
            eod = eod.copy()
            eod["date"] = pd.to_datetime(eod["date"])
            prices = eod.set_index("date")[price_col].sort_index()
            rets = log_returns_pct(prices)
            if len(rets) >= 10:
                return rets

    if asset in _BENCH_TICKERS:
        df = loaders.get_portfolio_and_benchmarks()
        sub = df[df["ticker"] == asset][["date", "daily_return"]].dropna()
        if not sub.empty:
            s = sub.set_index("date")["daily_return"] * 100
            s.index = pd.to_datetime(s.index)
            return s.sort_index().dropna()
    return None


def layout():
    today = _yesterday()
    sm = loaders.get_security_master()
    equities = sm[sm["asset_type"] != "INDEX"]["ticker"].tolist() if not sm.empty else []
    idx_tickers = sm[sm["asset_type"] == "INDEX"]["ticker"].tolist() if not sm.empty else []
    asset_options = ["Portfolio"] + equities + idx_tickers
    return dmc.Stack([
        ui.page_title(
            "Volatility",
            "Fits GARCH family models to estimate how volatile each asset currently "
            "is, compares model forecasts against realised vol, and projects "
            "volatility forward over your chosen horizon."),
        dmc.Group([
            dmc.Select(id="vol-asset", label="Asset", data=asset_options,
                       value="Portfolio", searchable=True, w=200),
            dmc.MultiSelect(id="vol-models", label="Models", data=list(ALL_MODELS),
                            value=list(ALL_MODELS), w=260),
            dmc.DatePickerInput(id="vol-backtest-from", label="Backtest from",
                                value=date(2025, 1, 1).isoformat(),
                                minDate=_DATE_MIN.isoformat(), maxDate=today.isoformat(), w=170),
            dmc.Box(dmc.Stack([
                dmc.Text("Forecast horizon (days)", size="sm", fw=500),
                dmc.Slider(id="vol-horizon", min=5, max=90, value=30, step=5,
                           marks=[{"value": v, "label": str(v)} for v in (30, 60, 90)], w=220),
            ], gap=4)),
            dmc.Button("Run", id="vol-run", mt=22),
        ], align="end", gap="lg"),
        dcc.Loading(dmc.Box(
            dmc.Alert("Configure settings and click Run to fit the models.",
                      color="blue", variant="light"),
            id="vol-results", mt="md")),
    ])


def _run_models(asset, models, backtest_from, horizon) -> dmc.Stack:
    rets_raw = _get_returns(asset)
    if rets_raw is None or len(rets_raw) < 20:
        return dmc.Alert(
            f"Not enough price history for {asset} (need ≥ 20 trading days). "
            "Try a different asset or ingest more data.", color="yellow", variant="light")

    bt = run_backtest(rets_raw, pd.Timestamp(backtest_from).date(), models)
    eval_df = evaluate_models(bt["variances"], bt["proxy_rv"])
    fc = forecast_ahead(rets_raw, models, steps=horizon)

    # ── 1. Backtest chart ──
    test_idx = bt["test_returns"].index
    proxy_ann = ann_vol_from_var(bt["proxy_rv"])
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=test_idx, y=proxy_ann, fill="tozeroy",
        fillcolor="rgba(148,163,184,0.18)",
        line=dict(color="rgba(148,163,184,0.5)", width=1),
        name="|Daily Return| × √252 (proxy)",
        hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>Proxy RV</extra>"))
    for name, h_series in bt["variances"].items():
        ann_v = ann_vol_from_var(h_series.reindex(test_idx).ffill())
        fig_bt.add_trace(go.Scatter(
            x=test_idx, y=ann_v, mode="lines", name=name,
            line=dict(color=_model_color(name), width=2),
            hovertemplate=f"%{{x|%d %b %Y}}  %{{y:.1f}}<extra>{name}</extra>"))
    fig_bt.update_layout(
        yaxis_title="Annualised Volatility (%)", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=40, b=60))
    fig_bt.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    # ── 2. Rankings ──
    disp_eval = eval_df.copy()
    disp_eval["MSE"] = disp_eval["MSE"].map("{:.6f}".format)
    disp_eval["QLIKE"] = disp_eval["QLIKE"].map("{:.4f}".format)
    verdict = None
    if "DM stat" in eval_df.columns:
        disp_eval["DM stat"] = disp_eval["DM stat"].map(
            lambda v: f"{float(v):.3f}" if pd.notna(v) and not isinstance(v, str) else "—")
        disp_eval["DM p-val"] = disp_eval["DM p-val"].map(
            lambda v: f"{float(v):.3f}" if pd.notna(v) and not isinstance(v, str) else "—")
        dm_stat, dm_p = eval_df["DM stat"].iloc[0], eval_df["DM p-val"].iloc[0]
        best, second = eval_df["Model"].iloc[0], eval_df["Model"].iloc[1]
        if pd.notna(dm_stat):
            sig = dm_p < 0.05
            verdict = (f"The Diebold-Mariano test (QLIKE loss, H₀: equal accuracy) gives "
                       f"stat = {dm_stat:.3f}, p = {dm_p:.3f}. "
                       + (f"{best} is significantly better than {second} at the 5% level."
                          if sig else
                          f"No statistically significant difference between {best} and {second}."))

    bar_fig = go.Figure()
    for _, row in eval_df.iterrows():
        bar_fig.add_trace(go.Bar(
            x=[row["Model"]], y=[row["QLIKE"]], name=row["Model"],
            marker_color=_model_color(row["Model"]),
            text=[f"{row['QLIKE']:.4f}"], textposition="outside"))
    bar_fig.update_layout(
        yaxis_title="Mean QLIKE (lower = better)", showlegend=False,
        height=260, margin=dict(l=60, r=20, t=20, b=40))

    # ── 3. Forecast ──
    fig_fc = go.Figure()
    for name in models:
        hist, fore = fc["history"].get(name), fc["forecast"].get(name)
        if hist is None or fore is None:
            continue
        hist_ann, fore_ann = ann_vol_from_var(hist), ann_vol_from_var(fore)
        fig_fc.add_trace(go.Scatter(
            x=hist_ann.index, y=hist_ann, mode="lines", name=name,
            line=dict(color=_model_color(name), width=2), legendgroup=name,
            hovertemplate=f"%{{x|%d %b %Y}}  %{{y:.1f}}<extra>{name}</extra>"))
        join_x = [hist_ann.index[-1]] + list(fore_ann.index)
        join_y = [float(hist_ann.iloc[-1])] + list(fore_ann.values)
        fig_fc.add_trace(go.Scatter(
            x=join_x, y=join_y, mode="lines", name=f"{name} forecast",
            line=dict(color=_model_color(name), width=2, dash="dash"),
            legendgroup=name, showlegend=True,
            hovertemplate=f"%{{x|%d %b %Y}}  %{{y:.1f}}<extra>{name} forecast</extra>"))
    last_hist = max(fc["history"][m].index[-1] for m in models if m in fc["history"])
    fig_fc.add_vline(x=last_hist.timestamp() * 1000, line_dash="dot", line_color="#BFDBFE",
                     annotation_text="Today", annotation_position="top right")
    fig_fc.update_layout(
        yaxis_title="Annualised Volatility (%)", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=40, b=60))
    fig_fc.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    n_fc_days = len(next(iter(fc["forecast"].values())))
    children = [
        ui.section(f"Backtest — {asset}  ·  from {backtest_from}"),
        dcc.Graph(figure=fig_bt, config=GRAPH_CONFIG),
        dmc.Text("Proxy RV is the absolute daily log-return scaled to annualised vol. "
                 "Model lines show conditional vol propagated through the test period "
                 "using parameters estimated on training data only.", size="sm", c="dimmed"),
        dmc.Divider(mt="lg"),
        ui.section("Model Rankings"),
        ui.df_table(disp_eval),
    ]
    if verdict:
        children.append(dmc.Text(verdict, size="sm", c="dimmed"))
    children += [
        dcc.Graph(figure=bar_fig, config=GRAPH_CONFIG),
        dmc.Divider(mt="lg"),
        ui.section(f"Volatility Forecast — next {n_fc_days} days"),
        dcc.Graph(figure=fig_fc, config=GRAPH_CONFIG),
        dmc.Text("Solid lines: in-sample conditional vol (last 90 trading days, fit on "
                 "full history). Dashed lines: multi-step-ahead forecast. GARCH-family "
                 "forecasts converge toward long-run volatility as the horizon increases.",
                 size="sm", c="dimmed"),
    ]
    return dmc.Stack(children, gap="md")


@callback(
    Output("vol-results", "children"),
    Input("vol-run", "n_clicks"),
    State("vol-asset", "value"),
    State("vol-models", "value"),
    State("vol-backtest-from", "value"),
    State("vol-horizon", "value"),
    background=True,
    running=[(Output("vol-run", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_volatility(n_clicks, asset, models, backtest_from, horizon):
    if not models:
        return dmc.Alert("Select at least one model.", color="blue", variant="light")
    try:
        return _run_models(asset, models, backtest_from, horizon)
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True),
                         color="red", variant="light", title="Volatility run failed")
