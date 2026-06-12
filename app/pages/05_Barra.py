import sys
import pathlib
import importlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Force-reload lib.barra so Streamlit doesn't serve a stale cached module
for _mod in list(sys.modules.keys()):
    if _mod == "lib.barra" or _mod.startswith("lib.barra."):
        del sys.modules[_mod]

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
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
    st.markdown("""Takes a cross section of the market and fits a regression to explain what drove our returns. Factors include sector exposure, momentum, and volatility rather than individual stock moves. Tells you whether losses came from bad factor bets or from idiosyncratic stock risk.""")

_TODAY    = date.today() - pd.Timedelta(days=1)
_DATE_MIN = date(2024, 1, 1)


# ── Controls ──────────────────────────────────────────────────────────────────

_c1, _c2 = st.columns([2, 1])
as_of_date = _c1.date_input("As-of date", value=_TODAY, min_value=_DATE_MIN, max_value=_TODAY)
run_btn    = _c2.button("Run Barra", type="primary", use_container_width=True)


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

if run_btn:
    with st.spinner("Building factor model…"):
        try:
            fund_df["date"] = pd.to_datetime(fund_df["date"])
            fund_asof = fund_df[fund_df["date"] <= pd.Timestamp(as_of_date)]
            if fund_asof.empty:
                st.error(f"No fundamentals data on or before {as_of_date}.")
                st.stop()
            # Take the most recent row per RIC — avoids missing stocks that
            # didn't trade on the single latest date (different market calendars)
            snap_raw = (
                fund_asof.sort_values("date")
                .groupby("ric", as_index=False)
                .last()
            )

            # Convert to plain Python list-of-dicts — avoids pandas accessor issues
            snap_records = [
                {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
                for row in snap_raw.to_dict("records")
            ]

            snap_date = fund_asof["date"].max()
            st.caption(f"Snapshot: {snap_date.date()}  ·  {len(snap_records)} securities")

            # Exclude only the tickers we explicitly know are ETFs or indices
            # (from security_master). Everything else in the fundamentals universe
            # is assumed to be an equity — barra_universe.csv only contains stocks.
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

            # Build exposure matrix — pass full fund_df so market_cap history
            # can fill in momentum/growth for securities without EOD prices
            X = build_exposure_matrix(equity_records, eod_cache, as_of_date, fund_df)

            if X.empty:
                st.error("Exposure matrix is empty. Check fundamentals data.")
                st.stop()

            # Universe-level summaries (always available)
            uni_stats   = universe_factor_stats(X)
            sec_exp     = sector_exposures(X)

            # Portfolio weighted exposures (always available)
            port_weights = _latest_weights()
            port_exp     = pd.Series(dtype=float)
            if not port_weights.empty:
                # Only keep equity positions that are in the exposure matrix
                equity_weights = port_weights[port_weights.index.isin(X.index)]
                if not equity_weights.empty:
                    equity_weights = equity_weights / equity_weights.sum()
                    port_exp = portfolio_weighted_exposure(X, equity_weights)
                    st.caption(
                        f"Portfolio: {len(equity_weights)} equity positions matched  "
                        f"({', '.join(equity_weights.index.tolist())})"
                    )

            # Factor return estimation via WLS (optional — needs price history)
            tr_series   = {}
            mcap_series = {}
            for ticker in X.index:
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
                                p_now = float(sub[p_col].iloc[-1])
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
                    X, pd.Series(tr_series), pd.Series(mcap_series)
                )
                if not port_weights.empty and factor_returns is not None:
                    port_attr = portfolio_attribution(X, factor_returns, port_weights)
            except ValueError:
                pass  # Factor returns optional — exposures still shown

            st.session_state["barra_result"] = {
                "X":              X,
                "uni_stats":      uni_stats,
                "sec_exp":        sec_exp,
                "port_exp":       port_exp,
                "port_attr":      port_attr,
                "factor_returns": factor_returns,
                "fit":            fit,
                "as_of":          as_of_date,
                "n":              len(X),
            }
        except Exception:
            import traceback
            st.error(traceback.format_exc())
            st.session_state["barra_result"] = None

res = st.session_state["barra_result"]
if res is None:
    st.info("Select a date and click **Run Barra**.")
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


def _grouped_bar(df: pd.DataFrame, title: str, subtitle: str = "", height: int = 420):
    """Grouped vertical bar for portfolio vs universe comparison."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Portfolio",
        x=df.index.tolist(), y=df["Portfolio"].tolist(),
        marker_color=_BLUE, marker_line_width=0,
        text=[f"{v:+.2f}" for v in df["Portfolio"]],
        textposition="outside", textfont=dict(size=12),
        hovertemplate="%{x} — Portfolio: %{y:+.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Universe avg",
        x=df.index.tolist(), y=df["Universe"].tolist(),
        marker_color=_GREY, marker_line_width=0,
        text=[f"{v:+.2f}" for v in df["Universe"]],
        textposition="outside", textfont=dict(size=12),
        hovertemplate="%{x} — Universe: %{y:+.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.25)")
    title_text = f"<b>{title}</b>"
    if subtitle:
        title_text += f"<br><span style='font-size:12px;color:#94A3B8'>{subtitle}</span>"
    fig.update_layout(
        template="capital",
        title=dict(text=title_text, x=0, xanchor="left", pad=dict(l=0, b=8)),
        barmode="group",
        height=height,
        xaxis=dict(tickfont=dict(size=14), tickangle=0),
        yaxis=dict(title="Weighted Z-score", tickfont=dict(size=12), zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=13)),
        margin=dict(l=60, r=40, t=80, b=60),
        bargap=0.25, bargroupgap=0.08,
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


def _sector_heatmap(df: pd.DataFrame, height: int = 420):
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
    fig.update_layout(
        template="capital",
        title=dict(
            text="<b>Factor Exposures by Sector</b>"
                 "<br><span style='font-size:12px;color:#94A3B8'>Average Z-score per GICS sector</span>",
            x=0, xanchor="left",
        ),
        height=height,
        xaxis=dict(side="top", tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=12), autorange="reversed"),
        margin=dict(l=20, r=80, t=96, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. MARKET FACTOR LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Market Factor Landscape")
st.write("")

X_all = res["X"]
sec   = res["sec_exp"]


# Sector heatmap — full width
if not sec.empty:
    style_cols = [c for c in STYLE_FACTORS if c in sec.columns]
    st.plotly_chart(_sector_heatmap(sec[style_cols], height=max(380, 52 * len(sec) + 120)),
                    use_container_width=True)

# WLS factor returns (only if available)
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
        st.plotly_chart(_hbar(style_fr, "Style Factor Returns",
                              "Return attributable to each style factor this period", 400),
                        use_container_width=True)
    with c4:
        st.plotly_chart(_hbar(industry_fr, "Industry Factor Returns",
                              "Return attributable to each industry this period", 400),
                        use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PORTFOLIO FACTOR POSITIONING
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Portfolio Factor Positioning")
st.write("")

X        = res["X"]
port_exp = res["port_exp"]
port_attr= res["port_attr"]

# Portfolio factor tilts — single series, no universe comparison
if not port_exp.empty:
    st.plotly_chart(
        _hbar(
            port_exp.dropna(),
            title="Portfolio Factor Tilts",
            subtitle="Weighted-average style factor exposure across equity positions",
            height=400,
        ),
        use_container_width=True,
    )
    st.write("")

# Per-position heatmap — only portfolio equity positions
style_cols    = [c for c in STYLE_FACTORS if c in X.columns]
port_weights_all = _latest_weights()
port_equities = [t for t in port_weights_all.index if t in X.index]
style_pivot   = X.loc[port_equities, style_cols].dropna(how="all") if port_equities else pd.DataFrame()

if not style_pivot.empty:
    st.plotly_chart(
        _heatmap(
            style_pivot,
            title="Style Factor Exposures — Individual Positions",
            height=max(380, 44 * len(style_pivot) + 120),
        ),
        use_container_width=True,
    )
else:
    st.info("No equity positions found in the exposure matrix.")

# Attribution (only if WLS succeeded)
if port_attr:
    st.write("")
    fc = port_attr["factor_contributions"]
    style_fc    = fc[[f for f in STYLE_FACTORS if f in fc.index]]
    industry_fc = fc[[f for f in fc.index if f.startswith("Industry_")]]
    industry_fc.index = industry_fc.index.str.replace("Industry_", "", regex=False)

    m1, m2 = st.columns(2)
    m1.metric("Total factor return", f"{port_attr['total_factor_return']:+.2%}")
    m2.metric("Style contribution",  f"{style_fc.sum():+.2%}")
    st.write("")

    ca, cb = st.columns(2, gap="large")
    with ca:
        st.plotly_chart(_hbar(style_fc, "Style Factor Contributions",
                              "How much each style factor added/detracted from portfolio return", 400),
                        use_container_width=True)
    with cb:
        if not industry_fc.empty:
            st.plotly_chart(_hbar(industry_fc, "Industry Contributions",
                                  "How much each industry factor contributed", 400),
                            use_container_width=True)
