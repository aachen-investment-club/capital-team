"""Equities — per-security candlestick with period filter and headline metrics."""
import math
from datetime import date, timedelta

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback
from plotly.subplots import make_subplots

from capital.dashboard import components as ui
from capital.data import loaders

dash.register_page(
    __name__, path="/equities", name="Equities", order=2,
    description="Per-security price history: candlestick, volume, return and volatility.",
)

PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "All"]


def _select_options() -> list[dict]:
    master = loaders.get_security_master()
    return [{"value": sid, "label": f"{t}  —  {n}"}
            for sid, t, n in zip(master["security_id"], master["ticker"], master["name"])]


def layout():
    options = _select_options()
    if not options:
        return dmc.Alert("Security master is empty. Add securities to config/security_master.csv.",
                         color="yellow", variant="light")
    return dmc.Stack([
        ui.page_title(
            "Equities",
            "Ranks each holding by standard investment factors (momentum, valuation, "
            "and quality) so you can see at a glance which stocks are looking strong "
            "or weak relative to the rest."),
        dmc.Group([
            dmc.Select(id="eq-security", data=options, value=options[0]["value"],
                       searchable=True, w=340),
            dmc.SegmentedControl(id="eq-period", data=PERIODS, value="1Y"),
            dmc.Checkbox(id="eq-volume", label="Volume", checked=True),
        ], align="center"),
        dmc.Text(id="eq-history-note", size="sm", c="dimmed"),
        dmc.SimpleGrid(id="eq-metrics", cols={"base": 1, "sm": 3}),
        ui.graph("eq-chart"),
    ])


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


def _metric_card(label: str, value: str) -> dmc.Paper:
    return dmc.Paper([
        dmc.Text(label, className="metric-label"),
        dmc.Text(value, className="metric-value"),
    ], className="metric-card")


@callback(
    Output("eq-chart", "figure"),
    Output("eq-metrics", "children"),
    Output("eq-history-note", "children"),
    Input("eq-security", "value"),
    Input("eq-period", "value"),
    Input("eq-volume", "checked"),
)
def update_equity(security_id, period, show_volume):
    master = loaders.get_security_master()
    row = master[master["security_id"] == security_id]
    ticker_label = row["ticker"].iloc[0] if len(row) else security_id

    df = loaders.get_eod_prices(security_id)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=380, annotations=[dict(
            text="No EOD data for this security yet — it will appear after the next nightly ingest.",
            showarrow=False, font=dict(size=14, color="#64748B"))])
        return fig, [], ""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    view = _filter_period(df, period)
    note = f"Available history: {df['date'].min()} → {df['date'].max()}"
    if view.empty:
        fig = go.Figure()
        fig.update_layout(height=380, annotations=[dict(
            text=f"No data in the selected period ({period}). Try a longer window.",
            showarrow=False, font=dict(size=14, color="#64748B"))])
        return fig, [], note

    # Metrics on the full series, not the period view
    latest = view.iloc[-1]
    all_returns = df["close"].pct_change().dropna()
    n_calendar_days = max((df["date"].iloc[-1] - df["date"].iloc[0]).days, 1)
    total_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    ann_return = (1 + total_ret) ** (365.25 / n_calendar_days) - 1
    ann_vol = all_returns.std() * math.sqrt(252) if len(all_returns) > 1 else None
    metrics = [
        _metric_card("Latest Close", f"{latest['close']:,.2f}"),
        _metric_card("Ann. Return", f"{ann_return * 100:+.1f}%"),
        _metric_card("Ann. Volatility", f"{ann_vol * 100:.1f}%" if ann_vol is not None else "—"),
    ]

    if show_volume:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.78, 0.22], vertical_spacing=0.03)
        price_row, vol_row = 1, 2
    else:
        fig = make_subplots(rows=1, cols=1)
        price_row, vol_row = 1, None

    fig.add_trace(
        go.Candlestick(
            x=view["date"],
            open=view["open"], high=view["high"],
            low=view["low"], close=view["close"],
            name=ticker_label,
            increasing_line_color="#2E7D6B", decreasing_line_color="#8B2E4A",
            increasing_fillcolor="#2E7D6B", decreasing_fillcolor="#8B2E4A",
        ),
        row=price_row, col=1,
    )

    if show_volume and vol_row is not None:
        bar_colors = ["#2E7D6B" if c >= o else "#8B2E4A"
                      for c, o in zip(view["close"], view["open"])]
        fig.add_trace(
            go.Bar(x=view["date"], y=view["volume"], name="Volume",
                   marker_color=bar_colors, marker_opacity=0.55, showlegend=False),
            row=vol_row, col=1,
        )

    fig.update_layout(
        title=ticker_label,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=False,
        height=520 if show_volume else 400,
    )
    fig.update_yaxes(title_text="Price", row=price_row, col=1)
    if show_volume and vol_row is not None:
        fig.update_yaxes(title_text="", showticklabels=False, row=vol_row, col=1)

    return fig, metrics, note
