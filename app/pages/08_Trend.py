import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
for _mod in list(sys.modules.keys()):
    if _mod in ("lib.trend",) or _mod.startswith("lib.trend."):
        del sys.modules[_mod]

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from lib.theme import inject_css, FAVICON
from lib.trend import SMAFilter, KalmanFilter2D, GJRGarch, UKFFilter, forecast_linear

st.set_page_config(page_title="Trend Detection · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Trend Detection")
with st.popover("ℹ"):
    st.markdown("""Runs Kalman filter and moving average models on each asset to extract the underlying price trend and estimate whether momentum is building or fading. The UKF variant adapts its noise model using GJR GARCH volatility so it reacts faster in stressed markets.""")

# ── Controls ──────────────────────────────────────────────────────────────────

_r1c1, _r1c2, _r1c3, _r1c4, _r1c5, _r1c6 = st.columns([1.2, 1.2, 0.8, 0.8, 0.8, 1.5])
date_start       = _r1c1.date_input("Start date", value=pd.Timestamp("2022-01-01").date())
date_end         = _r1c2.date_input("End date",   value=pd.Timestamp.today().date())
use_sma          = _r1c3.checkbox("SMA",          value=True)
use_kalman       = _r1c4.checkbox("Kalman 2D",    value=True)
use_ukf          = _r1c5.checkbox("UKF",          value=True)
forecast_horizon = _r1c6.slider("Forecast horizon (days)", 5, 60, 20)

_r2c1, _r2c2, _r2c3, _r2c4 = st.columns([1, 1, 1, 3])
sma_window = _r2c1.slider("SMA window", 10, 200, 50) if use_sma else 50
use_gjr    = _r2c2.checkbox("GJR-GARCH vol",           value=True,
                             help="Fits GJR-GARCH(1,1,1) per asset.")
use_stress = _r2c3.checkbox("Credit/Liquidity stress", value=True,
                             help="Pulls stress score from the Credit & Liquidity page if run.")
st.divider()

# ── Constants ─────────────────────────────────────────────────────────────────

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
_LABELS = {"sma": f"SMA-{sma_window}", "kalman": "Kalman 2D", "ukf": "UKF"}


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_benchmark_prices(start: str, end: str) -> pd.DataFrame:
    from lib.data import get_market_prices
    cols = {}
    for ticker in BENCHMARKS.keys():
        s = get_market_prices(ticker).loc[start:end]
        if not s.empty:
            cols[ticker] = s
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_portfolio_positions_lseg(start: str, end: str) -> dict[str, pd.Series]:
    """Load prices for active holdings from S3 EOD data via lib.data."""
    from lib.data import get_security_master, get_daily_weightings_history, get_eod_prices, _eod_data_version
    sm = get_security_master()
    wh = get_daily_weightings_history()
    if not wh.empty:
        latest_date     = wh["date"].max()
        current_symbols = set(wh[wh["date"] == latest_date]["symbol"].tolist())
        sm = sm[sm["ticker"].isin(current_symbols)]
    sm = sm[sm["asset_type"] != "INDEX"]
    if sm.empty:
        return {}
    version = _eod_data_version()
    out = {}
    for _, row in sm.iterrows():
        eod = get_eod_prices(row["security_id"], version)
        if eod.empty:
            continue
        eod = eod.copy()
        eod["date"] = pd.to_datetime(eod["date"])
        col = "adj_close" if "adj_close" in eod.columns else "close"
        s = eod.set_index("date")[col].loc[start:end].dropna().sort_index()
        if len(s) >= 20:
            out[row["ticker"]] = s.rename(row["ticker"])
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _load_portfolio_aggregate(start: str, end: str) -> pd.Series:
    from lib.data import get_portfolio_and_benchmarks
    pb   = get_portfolio_and_benchmarks()
    port = pb[pb["ticker"] == "PORTFOLIO"].set_index("date")["index_value"]
    port.index = pd.to_datetime(port.index)
    return port.loc[start:end].sort_index().rename("Portfolio NAV")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_custom_lseg(ric: str, start: str, end: str) -> pd.Series:
    """Fetch a custom LSEG RIC — used only for ad-hoc search, not regular data."""
    from lib.data import get_security_master, get_eod_prices, _eod_data_version
    sm = get_security_master()
    match = sm[sm["ric"] == ric]
    if not match.empty:
        version = _eod_data_version()
        eod = get_eod_prices(match.iloc[0]["security_id"], version)
        if not eod.empty:
            eod["date"] = pd.to_datetime(eod["date"])
            col = "adj_close" if "adj_close" in eod.columns else "close"
            return eod.set_index("date")[col].loc[start:end].dropna().sort_index().rename(ric)
    # Fallback: try market data store
    from lib.data import get_market_prices
    ticker = ric.split(".")[0]
    s = get_market_prices(ticker).loc[start:end]
    if not s.empty:
        return s.rename(ric)
    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_custom_yf(ticker: str, start: str, end: str) -> pd.Series:
    from lib.data import get_market_prices
    return get_market_prices(ticker).loc[start:end].rename(ticker)


# ── Stress ────────────────────────────────────────────────────────────────────

def _get_stress() -> pd.Series:
    cl = st.session_state.get("cl_result")
    if cl is None:
        return pd.Series(dtype=float)
    return cl.get("stress_z", pd.Series(dtype=float))


# ── Filter computation (cached) ───────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_filters(
    name: str, start: str, end: str,
    prices_values: tuple, prices_index: tuple,
    stress_values: tuple, stress_index: tuple,
    run_sma: bool, run_kalman: bool, run_ukf: bool,
    sma_win: int, gjr_enabled: bool, horizon: int,
) -> dict:
    prices = pd.Series(list(prices_values), index=pd.DatetimeIndex(list(prices_index)))
    stress = (pd.Series(list(stress_values), index=pd.DatetimeIndex(list(stress_index)))
              if stress_values else pd.Series(dtype=float))

    result: dict = {}

    if run_sma:
        try:
            result["sma"] = SMAFilter(window=sma_win).run(prices)
        except Exception:
            pass

    if run_kalman:
        try:
            result["kalman"] = KalmanFilter2D().run(prices)
        except Exception:
            pass

    if run_ukf:
        vol = pd.Series(dtype=float)
        if gjr_enabled:
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
                result["forecasts"][key] = forecast_linear(result[key], horizon)
            except Exception:
                pass

    return result


def _run_filters(name: str, prices: pd.Series, stress: pd.Series) -> dict:
    sv = tuple(stress.values.tolist()) if not stress.empty else ()
    si = tuple(stress.index.astype(str).tolist()) if not stress.empty else ()

    cached = _cached_filters(
        name=name,
        start=str(prices.index[0].date()),
        end=str(prices.index[-1].date()),
        prices_values=tuple(prices.values.tolist()),
        prices_index=tuple(prices.index.astype(str).tolist()),
        stress_values=sv,
        stress_index=si,
        run_sma=use_sma,
        run_kalman=use_kalman,
        run_ukf=use_ukf,
        sma_win=sma_window,
        gjr_enabled=use_gjr,
        horizon=forecast_horizon,
    )
    cached["name"]   = name
    cached["prices"] = prices
    return cached


# ── Chart ─────────────────────────────────────────────────────────────────────

def _trend_chart(result: dict, height: int = 480) -> go.Figure:
    prices = result["prices"].dropna()
    name   = result["name"]

    # Count active filters to size top margin correctly
    active = [k for k in ("sma", "kalman", "ukf") if k in result]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.70, 0.30],
        vertical_spacing=0.06,
        subplot_titles=("", "Slope oscillator"),
    )
    # Remove auto-generated subplot title annotations (we'll use our own)
    fig.layout.annotations = []

    # Price line — no legend entry, just hover
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines",
        line=dict(width=1.2, color="rgba(148,163,184,0.45)"),
        showlegend=False,
        hovertemplate="Price: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    for key, color in _COLORS.items():
        if key not in result:
            continue
        df  = result[key].dropna()
        if df.empty:
            continue
        lbl = _LABELS[key]

        # ── Green/red trend segments (no legend — colour speaks for itself)
        for is_bull, seg_color in ((True, "#10B981"), (False, "#EF4444")):
            seg = df["trend"].where(df["slope"] > 0 if is_bull else df["slope"] <= 0)
            fig.add_trace(go.Scatter(
                x=seg.index, y=seg.values,
                mode="lines",
                line=dict(width=2.5, color=seg_color),
                showlegend=False,
                hoverinfo="skip",
                connectgaps=False,
            ), row=1, col=1)

        # hover-only label (no legend entry — legend shown once at page level)
        fig.add_trace(go.Scatter(
            x=[df.index[-1]], y=[df["trend"].iloc[-1]],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle"),
            showlegend=False,
            hovertemplate=f"{lbl}: %{{y:.2f}}<extra></extra>",
        ), row=1, col=1)

        # ── Forecast band + dashed line (no extra legend entry)
        if key in result.get("forecasts", {}):
            fc = result["forecasts"][key]
            fig.add_trace(go.Scatter(
                x=list(fc.index) + list(fc.index[::-1]),
                y=list(fc["upper_1s"]) + list(fc["lower_1s"][::-1]),
                fill="toself",
                fillcolor=_hex_rgba(color, 0.10),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=fc.index, y=fc["trend"].values,
                mode="lines",
                line=dict(width=1.8, color=color, dash="dash"),
                showlegend=False,
                hovertemplate=f"{lbl} fcast: %{{y:.2f}}<extra></extra>",
            ), row=1, col=1)

        # ── Slope oscillator (row 2)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["slope"].values,
            mode="lines",
            line=dict(width=1.5, color=color),
            showlegend=False,
            hovertemplate=f"{lbl} slope: %{{y:.5f}}<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=np.where(df["slope"].values > 0, df["slope"].values, 0),
            fill="tozeroy", mode="none",
            fillcolor="rgba(16,185,129,0.10)", showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=np.where(df["slope"].values <= 0, df["slope"].values, 0),
            fill="tozeroy", mode="none",
            fillcolor="rgba(239,68,68,0.10)", showlegend=False,
            hoverinfo="skip",
        ), row=2, col=1)

    fig.add_hline(y=0, row=2, col=1,
                  line_width=0.8, line_color="rgba(255,255,255,0.18)", line_dash="dot")

    fig.update_layout(
        template="capital",
        height=height,
        title=dict(
            text=f"<b>{name}</b>",
            x=0, xanchor="left",
            font=dict(size=15),
        ),
        showlegend=False,
        margin=dict(l=55, r=30, t=44, b=40),
        hovermode="x unified",
    )
    fig.update_yaxes(tickfont=dict(size=10), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=10), zeroline=False, row=2, col=1)
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10), row=2, col=1)
    return fig


def _render_results(all_results: dict):
    """Render summary table + charts for a results dict."""
    if not all_results:
        st.warning("No results to display.")
        return

    # ── One shared legend + key for the whole section ─────────────────────────
    key_parts = []
    if use_sma:
        key_parts.append(f"<span style='color:#F59E0B'>●</span> SMA-{sma_window}")
    if use_kalman:
        key_parts.append("<span style='color:#3B82F6'>●</span> Kalman 2D")
    if use_ukf:
        key_parts.append("<span style='color:#A855F7'>●</span> UKF")
    key_parts += [
        "<span style='color:#10B981'>━</span> Bullish",
        "<span style='color:#EF4444'>━</span> Bearish",
        "<span style='color:#94A3B8'>╌</span> Forecast",
    ]
    st.markdown("  &nbsp;·&nbsp;  ".join(key_parts), unsafe_allow_html=True)
    st.write("")

    # Summary table
    rows = []
    for asset_name, r in all_results.items():
        for key, label in _LABELS.items():
            if key not in r:
                continue
            df = r[key].dropna()
            if df.empty:
                continue
            slope_now = float(df["slope"].iloc[-1])
            trend_now = float(df["trend"].iloc[-1])
            price_now = float(r["prices"].dropna().iloc[-1])
            regime    = "↑ Bullish" if slope_now > 0 else "↓ Bearish"
            row = {
                "Asset": asset_name, "Filter": label, "Regime": regime,
                "Price": f"{price_now:.2f}", "Trend": f"{trend_now:.2f}",
                "Slope": f"{slope_now:+.5f}",
            }
            if key in r.get("forecasts", {}):
                fc     = r["forecasts"][key]
                fc_end = float(fc["trend"].iloc[-1])
                row["Fcast end"] = f"{fc_end:.2f}"
                row["Exp. Δ"]    = f"{fc_end - trend_now:+.2f}"
            rows.append(row)

    if rows:
        df_sum = pd.DataFrame(rows)
        def _cr(v):
            if "Bullish" in str(v): return "color:#10B981;font-weight:600"
            if "Bearish" in str(v): return "color:#EF4444;font-weight:600"
            return ""
        st.dataframe(df_sum.style.applymap(_cr, subset=["Regime"]),
                     use_container_width=True, hide_index=True)

    st.write("")

    # Charts — 2-column grid
    names = list(all_results.keys())
    n     = len(names)
    if n == 1:
        st.plotly_chart(_trend_chart(all_results[names[0]], height=520),
                        use_container_width=True)
    else:
        for i in range(0, n, 2):
            c1, c2 = st.columns(2, gap="large")
            for col, nm in zip([c1, c2], names[i:i+2]):
                with col:
                    st.plotly_chart(_trend_chart(all_results[nm], height=420),
                                    use_container_width=True)


# ── Main tabs ─────────────────────────────────────────────────────────────────

tab_bench, tab_pos, tab_agg, tab_custom = st.tabs([
    "Benchmarks", "Portfolio positions", "Portfolio aggregate", "Custom search"
])

stress = _get_stress() if use_stress else pd.Series(dtype=float)
if use_stress and stress.empty:
    st.info("No stress data — run Credit & Liquidity page first.")

# Param key used to detect when settings change and results need refreshing
_trend_params = (
    str(date_start), str(date_end),
    use_sma, use_kalman, use_ukf,
    sma_window, use_gjr, use_stress, forecast_horizon,
)

def _stale(key: str) -> bool:
    """True if the cached result was computed under different params."""
    return (
        key not in st.session_state
        or st.session_state.get(f"{key}_params") != _trend_params
    )

def _store(key: str, value):
    st.session_state[key] = value
    st.session_state[f"{key}_params"] = _trend_params

# ── Tab 1: Benchmarks ─────────────────────────────────────────────────────────

with tab_bench:
    if _stale("trend_bench"):
        with st.spinner("Fetching benchmark prices…"):
            prices_df = _load_benchmark_prices(str(date_start), str(date_end))
        if prices_df.empty:
            st.error("Could not load benchmark prices.")
        else:
            bench_results = {}
            with st.spinner("Fitting filters…"):
                for ticker, label in BENCHMARKS.items():
                    if ticker not in prices_df.columns:
                        continue
                    s = prices_df[ticker].dropna()
                    if len(s) < 30:
                        continue
                    bench_results[label] = _run_filters(label, s, stress)
            _store("trend_bench", bench_results)

    if "trend_bench" in st.session_state:
        _render_results(st.session_state["trend_bench"])

# ── Tab 2: Portfolio positions ────────────────────────────────────────────────

with tab_pos:
    run_pos = st.button("Load positions", type="primary", key="run_pos")
    if run_pos or (not _stale("trend_pos")):
        if _stale("trend_pos") or run_pos:
            with st.spinner("Fetching portfolio positions from LSEG…"):
                pos_map = _load_portfolio_positions_lseg(str(date_start), str(date_end))
            if not pos_map:
                st.error("No positions loaded. Check LSEG connectivity.")
            else:
                pos_results = {}
                with st.spinner("Fitting filters…"):
                    for ticker, s in pos_map.items():
                        pos_results[ticker] = _run_filters(ticker, s, stress)
                _store("trend_pos", pos_results)

    if "trend_pos" in st.session_state:
        _render_results(st.session_state["trend_pos"])
    else:
        st.info("Click **Load positions** to fetch LSEG data.")

# ── Tab 3: Portfolio aggregate ────────────────────────────────────────────────

with tab_agg:
    if _stale("trend_agg"):
        with st.spinner("Loading portfolio NAV…"):
            agg = _load_portfolio_aggregate(str(date_start), str(date_end))
        if agg.empty:
            st.error("No portfolio NAV data found.")
        else:
            agg_results = {"AIC Portfolio": _run_filters("AIC Portfolio", agg, stress)}
            _store("trend_agg", agg_results)

    if "trend_agg" in st.session_state:
        _render_results(st.session_state["trend_agg"])

# ── Tab 4: Custom search ──────────────────────────────────────────────────────

with tab_custom:
    st.markdown("Enter any ticker or LSEG RIC to run it through the filters.")
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    with cc1:
        custom_input = st.text_input("Ticker / LSEG RIC",
                                     placeholder="e.g. AAPL, MSFT.O, VOD.L, ^GSPC")
    with cc2:
        custom_source = st.radio("Source", ["yfinance", "LSEG"], horizontal=True)
    with cc3:
        custom_label = st.text_input("Display name (optional)", placeholder="e.g. Apple")

    run_custom = st.button("Run", type="primary", key="run_custom")

    if run_custom and custom_input.strip():
        ticker = custom_input.strip()
        label  = custom_label.strip() or ticker
        with st.spinner(f"Fetching {ticker}…"):
            try:
                if custom_source == "yfinance":
                    s = _load_custom_yf(ticker, str(date_start), str(date_end))
                else:
                    s = _load_custom_lseg(ticker, str(date_start), str(date_end))

                if s.empty or len(s) < 30:
                    st.error(f"Not enough data for {ticker} (need ≥ 30 trading days).")
                else:
                    with st.spinner("Fitting filters…"):
                        custom_result = _run_filters(label, s, stress)
                    st.session_state["trend_custom"] = {label: custom_result}
            except Exception as e:
                st.error(f"Failed to load {ticker}: {e}")

    if "trend_custom" in st.session_state:
        _render_results(st.session_state["trend_custom"])
    elif not (run_custom and custom_input.strip()):
        st.info("Enter a ticker above and click **Run**.")
