"""Barra Factor Model — reference page for the heavy-compute pattern:
the model run is a background callback (Run button + loading state), and all
inputs are small values; data comes from the version-keyed loaders.
"""
import traceback
from datetime import date

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc

from capital.analytics.barra import (
    STYLE_FACTORS,
    build_average_exposure_matrix,
    build_exposure_matrix,
    estimate_factor_returns,
    portfolio_attribution,
    portfolio_weighted_exposure,
    sector_exposures,
)
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG

dash.register_page(
    __name__, path="/barra", name="Barra", order=5,
    description="Cross-sectional factor model: exposures, factor returns, attribution.",
)

_DATE_MIN = date(2024, 1, 1)

_GREEN, _RED = "#10B981", "#EF4444"


def _yesterday() -> date:
    return date.today() - pd.Timedelta(days=1)


# ── Cached inputs ─────────────────────────────────────────────────────────────

@cached_by_version
def _load_eod() -> dict:
    """{ric: eod_df, ticker: eod_df} for every active security with data."""
    sm = loaders.get_security_master()
    result: dict = {}
    for rec in sm.to_dict("records"):
        eod = loaders.get_eod_prices(str(rec["security_id"]))
        if isinstance(eod, pd.DataFrame) and not eod.empty:
            result[str(rec["ric"])] = eod
            result[str(rec["ticker"])] = eod
    return result


@cached_by_version
def _latest_weights() -> pd.Series:
    df = loaders.get_daily_weightings_history()
    if df.empty:
        return pd.Series(dtype=float)
    latest = df[df["date"] == df["date"].max()]
    latest = latest[~latest["symbol"].str.startswith("CASH_")]
    total = latest["pct_nav"].sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return latest.set_index("symbol")["pct_nav"] / total


# ── Chart helpers (unchanged from the Streamlit page) ─────────────────────────

def _hbar(series: pd.Series, title: str, subtitle: str = "", height: int = 420):
    s = series.dropna().sort_values()
    colors = [_GREEN if v >= 0 else _RED for v in s.values]
    fig = go.Figure(go.Bar(
        y=s.index.tolist(), x=s.values, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.2f}" for v in s.values], textposition="outside",
        textfont=dict(size=13),
        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.25)")
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px;color:#94A3B8'>{subtitle}</span>"
    fig.update_layout(
        template="capital",
        title=dict(text=title_text, x=0, xanchor="left", pad=dict(l=0, b=8)),
        height=height,
        xaxis=dict(title="Z-score", zeroline=False,
                   tickfont=dict(size=12), title_font=dict(size=13)),
        yaxis=dict(tickfont=dict(size=13)),
        margin=dict(l=20, r=100, t=72, b=48),
        bargap=0.35,
    )
    return fig


def _heatmap(df: pd.DataFrame, title: str, height: int = 500):
    zmax = max(abs(df.values[np.isfinite(df.values)].max()),
               abs(df.values[np.isfinite(df.values)].min()), 2)
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale="RdYlGn", zmid=0, zmin=-zmax, zmax=zmax,
        text=[[f"{v:+.2f}" if np.isfinite(v) else "—" for v in row] for row in df.values],
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y} · %{x}: %{z:+.3f}<extra></extra>",
        colorbar=dict(title="Z-score", thickness=14, len=0.8),
    ))
    fig.update_layout(
        template="capital",
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left"),
        height=height,
        xaxis=dict(side="top", tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=20, r=80, t=80, b=20),
    )
    return fig


def _sector_heatmap(df: pd.DataFrame, subtitle: str = "", height: int = 420):
    zmax = max(abs(df.values[np.isfinite(df.values)].max()),
               abs(df.values[np.isfinite(df.values)].min()), 1.5)
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
        colorscale="RdYlGn", zmid=0, zmin=-zmax, zmax=zmax,
        text=[[f"{v:+.2f}" if np.isfinite(v) else "—" for v in row] for row in df.values],
        texttemplate="%{text}", textfont=dict(size=12),
        hovertemplate="Sector: %{y}<br>Factor: %{x}<br>Avg Z-score: %{z:+.3f}<extra></extra>",
        colorbar=dict(title="Avg Z-score", thickness=14, len=0.8),
    ))
    title_text = "<b>Factor Exposures by Sector</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px;color:#94A3B8'>{subtitle}</span>"
    fig.update_layout(
        template="capital",
        title=dict(text=title_text, x=0, xanchor="left"),
        height=height,
        xaxis=dict(side="top", tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=20, r=80, t=96, b=20),
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def _month_options():
    today = _yesterday()
    months = pd.date_range(start=pd.Timestamp(_DATE_MIN).to_period("M").to_timestamp(),
                           end=pd.Timestamp(today).to_period("M").to_timestamp(), freq="MS")
    return [m.strftime("%b %Y") for m in months]


def layout():
    today = _yesterday()
    months = _month_options()
    return dmc.Stack([
        ui.page_title(
            "Barra Factor Model",
            "Takes a cross section of the market and fits a regression to explain "
            "what drove our returns. Factors include sector exposure, momentum, and "
            "volatility rather than individual stock moves. Daily shows Z-scores as "
            "of the selected date; monthly averages daily Z-scores across every "
            "business day in the selected range."),
        dmc.SegmentedControl(id="barra-view", value="Daily", data=["Daily", "Monthly"]),
        dmc.Group([
            dmc.Box(dmc.DatePickerInput(id="barra-date", label="Date",
                                        value=today.isoformat(),
                                        minDate=_DATE_MIN.isoformat(),
                                        maxDate=today.isoformat(), w=180),
                    id="barra-daily-controls"),
            dmc.Box(dmc.Group([
                dmc.Select(id="barra-from-month", label="From month", data=months,
                           value=months[max(0, len(months) - 3)], w=160),
                dmc.Select(id="barra-to-month", label="To month", data=months,
                           value=months[-1], w=160),
            ]), id="barra-monthly-controls", style={"display": "none"}),
            dmc.Button("Run Barra", id="barra-run", mt=24),
        ], align="start"),
        dcc.Loading(dmc.Box(
            dmc.Alert("Select a date or month range and click Run Barra.",
                      color="blue", variant="light"),
            id="barra-results", mt="md",
        )),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("barra-daily-controls", "style"),
    Output("barra-monthly-controls", "style"),
    Input("barra-view", "value"),
)
def toggle_controls(view):
    daily = {} if view == "Daily" else {"display": "none"}
    monthly = {"display": "none"} if view == "Daily" else {}
    return daily, monthly


def _resolve_window(view, day_value, from_month, to_month):
    today = _yesterday()
    if view == "Daily":
        d = pd.Timestamp(day_value).date()
        return d, d
    start = pd.Timestamp(from_month).date()
    end = min((pd.Timestamp(to_month) + pd.offsets.MonthEnd(0)).date(), today)
    return start, end


def _run_model(view, hist_start, hist_end) -> dmc.Stack:
    fund_df = loaders.get_fundamentals()
    if fund_df.empty:
        return dmc.Alert("No fundamentals data found — run `capital-ingest fund` first.",
                         color="yellow", variant="light")
    eod_cache = _load_eod()
    sm = loaders.get_security_master()

    fund_df = fund_df.copy()
    fund_df["date"] = pd.to_datetime(fund_df["date"])
    as_of_date = hist_end
    fund_asof = fund_df[fund_df["date"] <= pd.Timestamp(as_of_date)]
    if fund_asof.empty:
        return dmc.Alert(f"No fundamentals data on or before {as_of_date}.",
                         color="yellow", variant="light")

    snap_raw = fund_asof.sort_values("date").groupby("ric", as_index=False).last()
    snap_records = [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        for row in snap_raw.to_dict("records")
    ]
    snap_date = fund_asof["date"].max()

    known_non_equity = set(sm[sm["asset_type"].isin(["ETF", "INDEX"])]["ticker"].tolist())
    equity_records = [r for r in snap_records
                      if str(r.get("ticker", "")) not in known_non_equity]
    if len(equity_records) < 3:
        equity_records = snap_records

    X_snap = build_exposure_matrix(equity_records, eod_cache, as_of_date, fund_df)
    if X_snap.empty:
        return dmc.Alert("Exposure matrix is empty. Check fundamentals data.",
                         color="yellow", variant="light")

    if view == "Daily":
        display_X = X_snap
    else:
        dates = pd.bdate_range(hist_start, hist_end)
        display_X = build_average_exposure_matrix(fund_df, eod_cache, dates, known_non_equity)
        if display_X.empty:
            display_X = X_snap

    X_for_sec = X_snap.copy()
    for _col in STYLE_FACTORS:
        if _col in display_X.columns and _col in X_for_sec.columns:
            X_for_sec[_col] = display_X[_col].reindex(X_for_sec.index)
    sec_exp = sector_exposures(X_for_sec)

    port_weights = _latest_weights()
    port_exp = pd.Series(dtype=float)
    matched_note = ""
    if not port_weights.empty:
        equity_weights = port_weights[port_weights.index.isin(display_X.index)]
        if not equity_weights.empty:
            equity_weights = equity_weights / equity_weights.sum()
            port_exp = portfolio_weighted_exposure(display_X, equity_weights)
            matched_note = (f"Portfolio: {len(equity_weights)} equity positions matched "
                            f"({', '.join(equity_weights.index.tolist())})")

    # WLS attribution on single-date snapshot
    tr_series, mcap_series = {}, {}
    for ticker in X_snap.index:
        ric_matches = [r.get("ric") for r in equity_records if str(r.get("ticker")) == ticker]
        ric = str(ric_matches[0]) if ric_matches else ticker
        for key in (ric, ticker):
            if key in eod_cache:
                eod = eod_cache[key]
                if isinstance(eod, pd.DataFrame) and not eod.empty:
                    eod = eod.copy()
                    eod["date"] = pd.to_datetime(eod["date"])
                    p_col = "adj_close" if "adj_close" in eod.columns else "close"
                    sub = eod[eod["date"] <= pd.Timestamp(as_of_date)].sort_values("date")
                    if len(sub) >= 5:
                        p_now, p_start = float(sub[p_col].iloc[-1]), float(sub[p_col].iloc[0])
                        if p_start > 0:
                            tr_series[ticker] = p_now / p_start - 1.0
                break
        mcap_m = [r.get("market_cap") for r in equity_records if str(r.get("ticker")) == ticker]
        if mcap_m:
            try:
                mcap_series[ticker] = float(mcap_m[0])
            except (TypeError, ValueError):
                pass

    factor_returns, fit = None, None
    try:
        factor_returns, fit = estimate_factor_returns(
            X_snap, pd.Series(tr_series), pd.Series(mcap_series))
    except ValueError:
        pass

    mode_label = (f"as of {as_of_date}" if view == "Daily"
                  else f"avg {hist_start.strftime('%b %Y')} → {hist_end.strftime('%b %Y')}")

    # ── Assemble output sections ──
    children = [dmc.Text(f"Snapshot: {snap_date.date()}  ·  {len(snap_records)} securities",
                         size="sm", c="dimmed")]
    if matched_note:
        children.append(dmc.Text(matched_note, size="sm", c="dimmed"))

    children.append(ui.section("Market Factor Landscape"))
    if not sec_exp.empty:
        style_cols = [c for c in STYLE_FACTORS if c in sec_exp.columns]
        children.append(dcc.Graph(
            figure=_sector_heatmap(sec_exp[style_cols], subtitle=mode_label,
                                   height=max(380, 52 * len(sec_exp) + 120)),
            config=GRAPH_CONFIG))

    if factor_returns is not None:
        fr = factor_returns
        r2 = fit.rsquared if fit is not None else None
        if r2 is not None:
            children.append(dmc.Text(f"WLS cross-sectional regression  ·  R² = {r2:.3f}",
                                     size="sm", c="dimmed"))
        style_fr = fr[[f for f in STYLE_FACTORS if f in fr.index]]
        industry_fr = fr[[f for f in fr.index if f.startswith("Industry_")]]
        industry_fr.index = industry_fr.index.str.replace("Industry_", "", regex=False)
        children.append(dmc.SimpleGrid([
            dcc.Graph(figure=_hbar(style_fr, "Style Factor Returns",
                                   f"As of {as_of_date}", 400), config=GRAPH_CONFIG),
            dcc.Graph(figure=_hbar(industry_fr, "Industry Factor Returns",
                                   f"As of {as_of_date}", 400), config=GRAPH_CONFIG),
        ], cols={"base": 1, "lg": 2}))

    children.append(dmc.Divider(mt="lg"))
    children.append(ui.section("Style Factor Exposures"))
    if not port_exp.empty:
        children.append(dcc.Graph(
            figure=_hbar(port_exp.dropna(), "Portfolio Factor Tilts", mode_label, 400),
            config=GRAPH_CONFIG))

    style_cols = [c for c in STYLE_FACTORS if c in display_X.columns]
    port_equities = [t for t in port_weights.index if t in display_X.index]
    style_pivot = (display_X.loc[port_equities, style_cols].dropna(how="all")
                   if port_equities else pd.DataFrame())
    if not style_pivot.empty:
        children.append(dcc.Graph(
            figure=_heatmap(style_pivot,
                            f"Style Factor Exposures — Individual Positions  ·  {mode_label}",
                            height=max(380, 44 * len(style_pivot) + 120)),
            config=GRAPH_CONFIG))
    else:
        children.append(dmc.Alert("No equity positions found in the exposure matrix.",
                                  color="blue", variant="light"))

    return dmc.Stack(children, gap="md")


@callback(
    Output("barra-results", "children"),
    Input("barra-run", "n_clicks"),
    State("barra-view", "value"),
    State("barra-date", "value"),
    State("barra-from-month", "value"),
    State("barra-to-month", "value"),
    prevent_initial_call=True,
)
def run_barra(n_clicks, view, day_value, from_month, to_month):
    try:
        hist_start, hist_end = _resolve_window(view, day_value, from_month, to_month)
        return _run_model(view, hist_start, hist_end)
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True),
                         color="red", variant="light", title="Barra run failed")
