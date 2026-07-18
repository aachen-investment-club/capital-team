"""Portfolio Optimiser — efficient frontier, cvxpy optimiser (background callback),
and weighting-scheme backtest comparison."""
from datetime import date

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc

from capital.analytics.weighting import (
    build_basket_stats,
    build_price_matrix,
    efficient_frontier,
    get_weights,
    portfolio_stats,
    scheme_nav,
    scheme_nav_fixed,
    scheme_stats,
)
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG

dash.register_page(
    __name__, path="/weighting", name="Weighting", order=3,
    description="Efficient frontier, quadratic optimiser and weighting-scheme backtests.",
)

_DATE_MIN = date(2024, 1, 1)
_REBAL_OPTIONS = {"Daily": "B", "Weekly": "W-FRI", "Monthly": "MS", "Never": None}

SCHEME_COLORS = {
    "Equal weight":    "#3B82F6",
    "Inverse vol":     "#10B981",
    "Momentum":        "#F59E0B",
    "Min beta":        "#06B6D4",
    "Current weights": "#A855F7",
    "Optimised":       "#EF4444",
}


def _yesterday() -> date:
    return date.today() - pd.Timedelta(days=1)


# ── Cached data helpers ───────────────────────────────────────────────────────

@cached_by_version
def _master_non_index() -> pd.DataFrame:
    sm = loaders.get_security_master()
    return sm[sm["asset_type"] != "INDEX"].reset_index(drop=True)


@cached_by_version
def _load_all_eod() -> dict:
    sm = _master_non_index()
    return {row["security_id"]: loaders.get_eod_prices(row["security_id"])
            for _, row in sm.iterrows()}


@cached_by_version
def _spx_returns() -> pd.Series:
    df = loaders.get_portfolio_and_benchmarks()
    spx = df[df["ticker"] == "SPX"][["date", "daily_return"]].copy()
    return spx.dropna(subset=["daily_return"]).set_index("date")["daily_return"]


@cached_by_version
def _portfolio_returns() -> pd.Series:
    df = loaders.get_portfolio_and_benchmarks()
    port = df[df["ticker"] == "PORTFOLIO"][["date", "daily_return"]].copy()
    return port.dropna(subset=["daily_return"]).set_index("date")["daily_return"]


@cached_by_version
def _price_matrix() -> pd.DataFrame:
    sm = _master_non_index()
    return build_price_matrix(sm, lambda sid, _v="": loaders.get_eod_prices(sid), "")


@cached_by_version
def _basket_stats(backtest_from: str) -> list[dict]:
    return build_basket_stats(_master_non_index(), _load_all_eod(),
                              _spx_returns(), pd.Timestamp(backtest_from).date())


@cached_by_version
def _frontier(backtest_from: str) -> tuple:
    return efficient_frontier(_basket_stats(backtest_from))


@cached_by_version
def _scheme_nav(scheme: str, freq: str | None) -> pd.Series:
    prices = _price_matrix()
    spx = _spx_returns() if scheme == "beta" else None
    return scheme_nav(prices, scheme, freq, spx)


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    today = _yesterday()
    return dmc.Stack([
        ui.page_title(
            "Portfolio Optimiser",
            "Shows current and historical allocation by position and basket, and runs "
            "a quadratic optimiser to suggest better weights given your risk aversion "
            "and beta constraints."),
        dcc.Store(id="opt-store"),
        dmc.Group([
            dmc.DatePickerInput(id="opt-backtest-from", label="Backtest data from",
                                value=date(2025, 1, 1).isoformat(),
                                minDate=_DATE_MIN.isoformat(), maxDate=today.isoformat(), w=180),
            dmc.Box(dmc.Stack([
                dmc.Text("Risk aversion (λ)", size="sm", fw=500),
                dmc.Slider(id="opt-risk-aversion", min=0.1, max=5.0, value=1.0, step=0.1,
                           marks=[{"value": v, "label": str(v)} for v in (1, 3, 5)], w=220),
            ], gap=4)),
            dmc.Box(dmc.Stack([
                dmc.Text("Max portfolio beta", size="sm", fw=500),
                dmc.Slider(id="opt-max-beta", min=0.3, max=2.0, value=1.0, step=0.05,
                           marks=[{"value": v, "label": str(v)} for v in (0.5, 1.0, 1.5, 2.0)], w=220),
            ], gap=4)),
            dmc.Button("Run Optimiser", id="opt-run", mt=22),
        ], align="end", gap="xl"),

        ui.section("Efficient Frontier"),
        ui.graph("opt-frontier"),
        dcc.Loading(dmc.Box(
            dmc.Alert("Click Run Optimiser to generate optimal weights.",
                      color="blue", variant="light"),
            id="opt-results")),

        dmc.Divider(mt="lg"),
        ui.section("Weighting Scheme Comparison"),
        dmc.Select(id="opt-rebal-freq", label="Rebalance frequency",
                   data=list(_REBAL_OPTIONS.keys()), value="Monthly", w=200),
        dmc.Text("1-year backtest of weighting schemes applied to current holdings.",
                 size="sm", c="dimmed"),
        ui.graph("opt-scheme-chart"),
        dcc.Loading(dmc.Box(id="opt-scheme-stats")),
    ])


# ── Frontier ──────────────────────────────────────────────────────────────────

def _frontier_figure(backtest_from: str, opt_result: dict | None) -> go.Figure:
    basket_stats = _basket_stats(backtest_from)
    fig = go.Figure()
    if not basket_stats:
        fig.update_layout(height=420, annotations=[dict(
            text=f"No securities have sufficient price history from {backtest_from}.",
            showarrow=False, font=dict(size=14, color="#64748B"))])
        return fig

    try:
        f_vols, f_rets = _frontier(backtest_from)
        if len(f_vols) > 1:
            fig.add_trace(go.Scatter(
                x=list(f_vols), y=list(f_rets), mode="lines", name="Efficient Frontier",
                line=dict(color="#2563EB", width=2, dash="dash")))
    except Exception:
        pass

    port_rets = _portfolio_returns()
    port_rets = port_rets[port_rets.index >= pd.Timestamp(backtest_from)]
    if len(port_rets) >= 5:
        fig.add_trace(go.Scatter(
            x=[float(port_rets.std() * np.sqrt(252))],
            y=[float(port_rets.mean() * 252)],
            mode="markers", name="Current Portfolio",
            marker=dict(symbol="diamond", size=14, color="#F59E0B"),
            hovertemplate="<b>Current Portfolio</b><br>Ann Return: %{y:.1%}"
                          "<br>Ann Vol: %{x:.1%}<extra></extra>"))

    for b in basket_stats:
        fig.add_trace(go.Scatter(
            x=[b["Ann Vol"]], y=[b["Ann Return"]],
            mode="markers+text", name=b["Symbol"],
            text=[b["Symbol"]], textposition="top center",
            marker=dict(size=9, opacity=0.75),
            hovertemplate=(f"<b>{b['Symbol']}</b><br>Ann Return: %{{y:.1%}}<br>"
                           f"Ann Vol: %{{x:.1%}}<br>Beta: {b['Beta']:.2f}<extra></extra>")))

    if opt_result is not None:
        fig.add_trace(go.Scatter(
            x=[opt_result["stats"]["Ann Vol"]], y=[opt_result["stats"]["Ann Return"]],
            mode="markers", name="Optimised Portfolio",
            marker=dict(symbol="star", size=18, color="#EF4444"),
            hovertemplate=("<b>Optimised Portfolio</b><br>Ann Return: %{y:.1%}"
                           f"<br>Ann Vol: %{{x:.1%}}<br>Beta: {opt_result['stats']['Beta']:.2f}"
                           "<extra></extra>")))

    fig.update_layout(
        xaxis_title="Annualised Volatility", yaxis_title="Annualised Return",
        xaxis_tickformat=".0%", yaxis_tickformat=".0%",
        legend=dict(orientation="v"), height=520,
        margin=dict(l=60, r=20, t=40, b=60))
    fig.update_xaxes(rangemode="tozero")
    return fig


@callback(
    Output("opt-frontier", "figure"),
    Input("opt-backtest-from", "value"),
    Input("opt-store", "data"),
)
def update_frontier(backtest_from, opt_result):
    return _frontier_figure(backtest_from, opt_result)


# ── Optimiser (background) ────────────────────────────────────────────────────

@callback(
    Output("opt-store", "data"),
    Input("opt-run", "n_clicks"),
    State("opt-backtest-from", "value"),
    State("opt-risk-aversion", "value"),
    State("opt-max-beta", "value"),
    background=True,
    running=[(Output("opt-run", "loading"), True, False)],
    prevent_initial_call=True,
)
def run_optimiser(n_clicks, backtest_from, risk_aversion, max_beta):
    basket_stats = _basket_stats(backtest_from)
    if not basket_stats:
        return {"error": f"No securities have sufficient price history from {backtest_from}."}
    try:
        weights = get_weights(basket_stats, max_beta=max_beta, risk_aversion=risk_aversion)
        stats = portfolio_stats(basket_stats, weights)
        return {
            "weights": [float(w) for w in weights],
            "stats": {k: float(v) for k, v in stats.items()},
            "symbols": [b["Symbol"] for b in basket_stats],
            "names": [b["Name"] for b in basket_stats],
            "ind_vols": [b["Ann Vol"] for b in basket_stats],
            "ind_rets": [b["Ann Return"] for b in basket_stats],
            "betas": [b["Beta"] for b in basket_stats],
        }
    except Exception as e:
        return {"error": str(e)}


@callback(Output("opt-results", "children"), Input("opt-store", "data"))
def show_optimiser_results(result):
    if result is None:
        return dmc.Alert("Click Run Optimiser to generate optimal weights.",
                         color="blue", variant="light")
    if "error" in result:
        return dmc.Alert(result["error"], color="red", variant="light")

    df_w = pd.DataFrame({
        "Symbol": result["symbols"], "Name": result["names"],
        "Weight": result["weights"], "Ann Return": result["ind_rets"],
        "Ann Vol": result["ind_vols"], "Beta": result["betas"],
    }).sort_values("Weight", ascending=False).reset_index(drop=True)

    bar_fig = go.Figure(go.Bar(
        x=df_w["Symbol"], y=df_w["Weight"], marker_color="#2563EB",
        text=df_w["Weight"].map("{:.1%}".format), textposition="outside"))
    bar_fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Weight",
                          height=320, margin=dict(l=40, r=20, t=20, b=40))

    disp = df_w.copy()
    disp["Weight"] = disp["Weight"].map("{:.1%}".format)
    disp["Ann Return"] = disp["Ann Return"].map("{:+.1%}".format)
    disp["Ann Vol"] = disp["Ann Vol"].map("{:.1%}".format)
    disp["Beta"] = disp["Beta"].map("{:.2f}".format)

    def _metric(label, value):
        return dmc.Paper([dmc.Text(label, className="metric-label"),
                          dmc.Text(value, className="metric-value")],
                         className="metric-card")

    return dmc.Stack([
        dmc.Divider(mt="md"),
        dmc.SimpleGrid([
            _metric("Ann Return", f"{result['stats']['Ann Return']:+.2%}"),
            _metric("Ann Vol", f"{result['stats']['Ann Vol']:.2%}"),
            _metric("Portfolio β", f"{result['stats']['Beta']:.2f}"),
        ], cols={"base": 1, "sm": 3}),
        ui.section("Optimal Weights"),
        dcc.Graph(figure=bar_fig, config=GRAPH_CONFIG),
        ui.df_table(disp),
    ])


# ── Weighting Scheme Comparison ───────────────────────────────────────────────

@callback(
    Output("opt-scheme-chart", "figure"),
    Output("opt-scheme-stats", "children"),
    Input("opt-rebal-freq", "value"),
    Input("opt-store", "data"),
)
def update_scheme_comparison(rebal_label, opt_result):
    rebal_freq = _REBAL_OPTIONS[rebal_label]
    prices = _price_matrix()
    if prices.empty:
        fig = go.Figure()
        fig.update_layout(height=380, annotations=[dict(
            text="Not enough price history to build comparison.",
            showarrow=False, font=dict(size=14, color="#64748B"))])
        return fig, None

    navs = [
        ("Equal weight", _scheme_nav("equal", rebal_freq)),
        ("Inverse vol", _scheme_nav("inv_vol", rebal_freq)),
        ("Momentum", _scheme_nav("momentum", rebal_freq)),
        ("Min beta", _scheme_nav("beta", rebal_freq)),
    ]

    # Current portfolio weights as static fixed-weight simulation
    wgt_hist = loaders.get_daily_weightings_history()
    if not wgt_hist.empty:
        latest_w = wgt_hist[wgt_hist["date"] == wgt_hist["date"].max()]
        latest_w = latest_w[~latest_w["symbol"].str.startswith("CASH_")]
        sym_to_idx = {col: i for i, col in enumerate(prices.columns)}
        w_cur = np.zeros(len(prices.columns))
        for _, row in latest_w.iterrows():
            if row["symbol"] in sym_to_idx:
                w_cur[sym_to_idx[row["symbol"]]] = float(row["pct_nav"])
        if w_cur.sum() > 0:
            w_cur /= w_cur.sum()
            navs.append(("Current weights", scheme_nav_fixed(prices, w_cur)))

    # Optimised weights simulation (after Run Optimiser)
    if opt_result and "error" not in opt_result:
        sym_to_idx = {col: i for i, col in enumerate(prices.columns)}
        w_opt = np.zeros(len(prices.columns))
        for sym, wt in zip(opt_result["symbols"], opt_result["weights"]):
            if sym in sym_to_idx:
                w_opt[sym_to_idx[sym]] = wt
        if w_opt.sum() > 0:
            w_opt /= w_opt.sum()
            navs.append(("Optimised", scheme_nav_fixed(prices, w_opt)))

    fig = go.Figure()
    for label, nav in navs:
        s = nav.dropna()
        dash_style = "dash" if label == "Optimised" else None
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label,
            line=dict(width=2.5 if label == "Optimised" else 2,
                      color=SCHEME_COLORS[label], dash=dash_style),
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=1.0, line_width=1, line_color="rgba(255,255,255,0.15)", line_dash="dot")
    fig.update_layout(
        title=dict(text="<b>NAV comparison</b> — start at 1.0", x=0, xanchor="left"),
        height=420,
        yaxis=dict(tickformat=".2f", zeroline=False),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=50, r=20, t=48, b=60))

    rows = [scheme_stats(nav, label) for label, nav in navs]
    df_stats = pd.DataFrame(rows)
    for col in ("Total return", "Ann return", "Ann vol", "Max drawdown"):
        df_stats[col] = df_stats[col].map("{:+.2%}".format)
    df_stats["Sharpe"] = df_stats["Sharpe"].map("{:.2f}".format)
    return fig, ui.df_table(df_stats)
