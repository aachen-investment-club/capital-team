import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Force-reload lib.barra so Streamlit doesn't serve a stale cached module
for _mod in list(sys.modules.keys()):
    if _mod == "lib.barra" or _mod.startswith("lib.barra."):
        del sys.modules[_mod]

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date

from lib.theme import inject_css, FAVICON
from lib.data import (
    get_security_master,
    get_eod_prices,
    get_daily_weightings_history,
    _eod_data_version,
)
from lib.fundamentals import get_fundamentals
from lib.barra import (
    STYLE_FACTORS,
    build_exposure_matrix,
    build_average_exposure_matrix,
    portfolio_weighted_exposure,
    universe_factor_stats,
    sector_exposures,
    estimate_factor_returns,
    portfolio_attribution,
)

st.set_page_config(page_title="Barra · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Barra Factor Model")
with st.popover("ℹ"):
    st.markdown("""Takes a cross section of the market and fits a regression to explain what drove our returns. Factors include sector exposure, momentum, and volatility rather than individual stock moves. Daily shows Z-scores as of the selected date; monthly averages daily Z-scores across every business day in the selected range.""")

_TODAY    = date.today() - pd.Timedelta(days=1)
_DATE_MIN = date(2024, 1, 1)
_DEFAULT_START = (pd.Timestamp(_TODAY) - pd.DateOffset(months=3)).date()


# ── Controls ──────────────────────────────────────────────────────────────────

view_mode = st.radio(
    "View", ["Daily", "Monthly"], horizontal=True, label_visibility="collapsed"
)
st.write("")

if view_mode == "Daily":
    _c1, _c2 = st.columns([3, 1])
    hist_start = _c1.date_input("Date", value=_TODAY, min_value=_DATE_MIN, max_value=_TODAY)
    hist_end   = hist_start
    run_btn    = _c2.button("Run Barra", type="primary", use_container_width=True)
else:
    _all_months   = pd.date_range(
        start=pd.Timestamp(_DATE_MIN).to_period("M").to_timestamp(),
        end=pd.Timestamp(_TODAY).to_period("M").to_timestamp(),
        freq="MS",
    )
    _month_labels = [m.strftime("%b %Y") for m in _all_months]
    _default_from = max(0, len(_month_labels) - 3)

    _c1, _c2, _c3 = st.columns([2, 2, 1])
    _sel_from  = _c1.selectbox("From month", _month_labels, index=_default_from)
    _sel_to    = _c2.selectbox("To month",   _month_labels, index=len(_month_labels) - 1)
    run_btn    = _c3.button("Run Barra", type="primary", use_container_width=True)

    hist_start = pd.Timestamp(_sel_from).date()
    hist_end   = min((pd.Timestamp(_sel_to) + pd.offsets.MonthEnd(0)).date(), _TODAY)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_eod(cache_version: str) -> dict:
    sm = get_security_master()
    result: dict = {}
    for rec in sm.to_dict("records"):
        eod = get_eod_prices(str(rec["security_id"]), cache_version)
        if isinstance(eod, pd.DataFrame) and not eod.empty:
            result[str(rec["ric"])]    = eod
            result[str(rec["ticker"])] = eod
    return result


@st.cache_data(ttl=3600)
def _latest_weights() -> pd.Series:
    df = get_daily_weightings_history()
    if df.empty:
        return pd.Series(dtype=float)
    latest = df[df["date"] == df["date"].max()]
    latest = latest[~latest["symbol"].str.startswith("CASH_")]
    total  = latest["pct_nav"].sum()
    if total <= 0:
        return pd.Series(dtype=float)
    return latest.set_index("symbol")["pct_nav"] / total


@st.cache_data(ttl=3600, show_spinner=False)
def _build_avg_exposure(start, end, cache_ver, _fund_df, _eod_cache):
    sm     = get_security_master()
    non_eq = set(sm[sm["asset_type"].isin(["ETF", "INDEX"])]["ticker"].tolist())
    dates  = pd.bdate_range(start, end)
    return build_average_exposure_matrix(_fund_df, _eod_cache, dates, non_eq)


version   = _eod_data_version()
fund_df   = get_fundamentals()
eod_cache = _load_eod(version)

if fund_df.empty:
    st.warning(
        "No fundamentals data found. "
        "Run `python scripts/ingest_fundamentals.py` to populate it."
    )
    st.stop()


# ── Session state ─────────────────────────────────────────────────────────────

if "barra_result" not in st.session_state:
    st.session_state["barra_result"] = None

_input_key = (view_mode, hist_start, hist_end)
if st.session_state.get("_barra_input_key") != _input_key:
    st.session_state["_barra_input_key"] = _input_key
    st.session_state["barra_result"] = None

if run_btn:
    n_days = pd.bdate_range(hist_start, hist_end).size
    label  = "Building factor model…" if n_days == 1 else f"Averaging {n_days} daily cross-sections…"
    with st.spinner(label):
        try:
            fund_df["date"] = pd.to_datetime(fund_df["date"])

            # Cross-section as of end date (always needed for WLS / attribution)
            as_of_date = hist_end
            fund_asof  = fund_df[fund_df["date"] <= pd.Timestamp(as_of_date)]
            if fund_asof.empty:
                st.error(f"No fundamentals data on or before {as_of_date}.")
                st.stop()

            snap_raw = (
                fund_asof.sort_values("date")
                .groupby("ric", as_index=False)
                .last()
            )
            snap_records = [
                {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
                for row in snap_raw.to_dict("records")
            ]

            snap_date = fund_asof["date"].max()
            st.caption(f"Snapshot: {snap_date.date()}  ·  {len(snap_records)} securities")

            sm = get_security_master()
            known_non_equity = set(
                sm[sm["asset_type"].isin(["ETF", "INDEX"])]["ticker"].tolist()
            )
            equity_records = [
                r for r in snap_records
                if str(r.get("ticker", "")) not in known_non_equity
            ]
            if len(equity_records) < 3:
                equity_records = snap_records

            # Single-date exposure matrix (used for WLS / attribution)
            X_snap = build_exposure_matrix(equity_records, eod_cache, as_of_date, fund_df)
            if X_snap.empty:
                st.error("Exposure matrix is empty. Check fundamentals data.")
                st.stop()

            # Display matrix: single date for daily, averaged for monthly
            if view_mode == "Daily":
                display_X = X_snap
            else:
                display_X = _build_avg_exposure(hist_start, hist_end, version, fund_df, eod_cache)
                if display_X.empty:
                    display_X = X_snap

            # Merge averaged Z-scores into X_snap so industry dummies are present
            # for sector_exposures grouping (sector assignments come from X_snap,
            # style factor values come from display_X)
            X_for_sec = X_snap.copy()
            for _col in STYLE_FACTORS:
                if _col in display_X.columns and _col in X_for_sec.columns:
                    X_for_sec[_col] = display_X[_col].reindex(X_for_sec.index)
            sec_exp = sector_exposures(X_for_sec)

            port_weights = _latest_weights()
            port_exp     = pd.Series(dtype=float)
            if not port_weights.empty:
                equity_weights = port_weights[port_weights.index.isin(display_X.index)]
                if not equity_weights.empty:
                    equity_weights = equity_weights / equity_weights.sum()
                    port_exp = portfolio_weighted_exposure(display_X, equity_weights)
                    st.caption(
                        f"Portfolio: {len(equity_weights)} equity positions matched  "
                        f"({', '.join(equity_weights.index.tolist())})"
                    )

            # WLS attribution on single-date snapshot
            tr_series   = {}
            mcap_series = {}
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
                                p_now   = float(sub[p_col].iloc[-1])
                                p_start = float(sub[p_col].iloc[0])
                                if p_start > 0:
                                    tr_series[ticker] = p_now / p_start - 1.0
                            break
                mcap_m = [r.get("market_cap") for r in equity_records if str(r.get("ticker")) == ticker]
                if mcap_m:
                    try:
                        mcap_series[ticker] = float(mcap_m[0])
                    except (TypeError, ValueError):
                        pass

            factor_returns = None
            fit            = None
            port_attr      = {}
            try:
                factor_returns, fit = estimate_factor_returns(
                    X_snap, pd.Series(tr_series), pd.Series(mcap_series)
                )
                if not port_weights.empty and factor_returns is not None:
                    port_attr = portfolio_attribution(X_snap, factor_returns, port_weights)
            except ValueError:
                pass

            st.session_state["barra_result"] = {
                "display_X":     display_X,
                "sec_exp":       sec_exp,
                "port_exp":      port_exp,
                "port_attr":     port_attr,
                "factor_returns": factor_returns,
                "fit":           fit,
                "as_of":         as_of_date,
                "hist_start":    hist_start,
                "hist_end":      hist_end,
                "view_mode":     view_mode,
                "n":             len(display_X),
            }
        except Exception:
            import traceback
            st.error(traceback.format_exc())
            st.session_state["barra_result"] = None

res = st.session_state["barra_result"]
if res is None:
    st.info("Select a date or month range and click **Run Barra**.")
    st.stop()


# ── Chart helpers ─────────────────────────────────────────────────────────────

_GREEN = "#10B981"
_RED   = "#EF4444"
_BLUE  = "#3B82F6"
_GREY  = "#64748B"


def _hbar(series: pd.Series, title: str, subtitle: str = "", height: int = 420):
    """Horizontal diverging bar — sorted by value, colour-coded pos/neg."""
    s = series.dropna().sort_values()
    colors = [_GREEN if v >= 0 else _RED for v in s.values]
    fig = go.Figure(go.Bar(
        y=s.index.tolist(),
        x=s.values,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:+.2f}" for v in s.values],
        textposition="outside",
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
        xaxis=dict(
            title="Z-score", zeroline=False,
            tickfont=dict(size=12), title_font=dict(size=13),
        ),
        yaxis=dict(tickfont=dict(size=13)),
        margin=dict(l=20, r=100, t=72, b=48),
        bargap=0.35,
    )
    return fig


def _heatmap(df: pd.DataFrame, title: str, height: int = 500):
    """Factor exposure heatmap — securities on Y axis, factors on X."""
    zmax = max(abs(df.values[np.isfinite(df.values)].max()),
               abs(df.values[np.isfinite(df.values)].min()), 2)
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale="RdYlGn",
        zmid=0, zmin=-zmax, zmax=zmax,
        text=[[f"{v:+.2f}" if np.isfinite(v) else "—" for v in row] for row in df.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
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
    """Sector × factor heatmap."""
    zmax = max(abs(df.values[np.isfinite(df.values)].max()),
               abs(df.values[np.isfinite(df.values)].min()), 1.5)
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale="RdYlGn",
        zmid=0, zmin=-zmax, zmax=zmax,
        text=[[f"{v:+.2f}" if np.isfinite(v) else "—" for v in row] for row in df.values],
        texttemplate="%{text}",
        textfont=dict(size=12),
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


# ── Shared display data ───────────────────────────────────────────────────────

display_X = res["display_X"]
sec_exp   = res["sec_exp"]
port_exp  = res["port_exp"]
port_attr = res["port_attr"]

_mode_label = (
    f"as of {res['as_of']}"
    if res["view_mode"] == "Daily"
    else f"avg {res['hist_start'].strftime('%b %Y')} → {res['hist_end'].strftime('%b %Y')}"
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MARKET FACTOR LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Market Factor Landscape")
st.write("")

if not sec_exp.empty:
    style_cols = [c for c in STYLE_FACTORS if c in sec_exp.columns]
    st.plotly_chart(
        _sector_heatmap(
            sec_exp[style_cols],
            subtitle=_mode_label,
            height=max(380, 52 * len(sec_exp) + 120),
        ),
        use_container_width=True,
    )

if res["factor_returns"] is not None:
    fr = res["factor_returns"]
    r2 = res["fit"].rsquared if res["fit"] is not None else None
    st.write("")
    st.caption(f"WLS cross-sectional regression  ·  R² = {r2:.3f}" if r2 is not None else "")
    style_fr    = fr[[f for f in STYLE_FACTORS if f in fr.index]]
    industry_fr = fr[[f for f in fr.index if f.startswith("Industry_")]]
    industry_fr.index = industry_fr.index.str.replace("Industry_", "", regex=False)
    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.plotly_chart(
            _hbar(style_fr, "Style Factor Returns", f"As of {res['as_of']}", 400),
            use_container_width=True,
        )
    with c4:
        st.plotly_chart(
            _hbar(industry_fr, "Industry Factor Returns", f"As of {res['as_of']}", 400),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. STYLE FACTOR EXPOSURES
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Style Factor Exposures")
st.write("")

if not port_exp.empty:
    st.plotly_chart(
        _hbar(
            port_exp.dropna(),
            title="Portfolio Factor Tilts",
            subtitle=_mode_label,
            height=400,
        ),
        use_container_width=True,
    )
    st.write("")

style_cols       = [c for c in STYLE_FACTORS if c in display_X.columns]
port_weights_all = _latest_weights()
port_equities    = [t for t in port_weights_all.index if t in display_X.index]
style_pivot      = (
    display_X.loc[port_equities, style_cols].dropna(how="all")
    if port_equities else pd.DataFrame()
)

if not style_pivot.empty:
    st.plotly_chart(
        _heatmap(
            style_pivot,
            title=f"Style Factor Exposures — Individual Positions  ·  {_mode_label}",
            height=max(380, 44 * len(style_pivot) + 120),
        ),
        use_container_width=True,
    )
else:
    st.info("No equity positions found in the exposure matrix.")
