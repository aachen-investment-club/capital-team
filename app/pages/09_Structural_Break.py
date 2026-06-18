import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
for _mod in list(sys.modules.keys()):
    if _mod in ("lib.structural_break", "lib.credit_liquidity", "lib.trend", "lib.correlation") \
       or any(_mod.startswith(p) for p in ("lib.structural_break.", "lib.credit_liquidity.",
                                            "lib.trend.", "lib.correlation.")):
        del sys.modules[_mod]

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from lib.theme import inject_css, FAVICON
from lib.structural_break import CUSUMDetector, StudentTBOCPD, build_level_signal
from lib.credit_liquidity import CreditFilter, LiquidityFilter, StressScore
from lib.trend import GJRGarch

st.set_page_config(
    page_title="Structural Break · AIC",
    page_icon=FAVICON,
    layout="wide",
)
inject_css()
st.title("Structural Break Detection")
with st.popover("ℹ"):
    st.markdown("""Detects when the market has shifted into a new regime (e.g. from calm to stressed) using two methods: CUSUM, which flags when a stress signal persistently drifts above or below its baseline, and BOCPD, which estimates the probability that the current observation belongs to a new distribution.""")

# ── Sidebar ───────────────────────────────────────────────────────────────────

# ── Controls ──────────────────────────────────────────────────────────────────

_r1c1, _r1c2, _r1c3, _r1c4, _r1c5, _r1c6 = st.columns([1.2, 1.2, 0.8, 0.8, 0.8, 0.8])
date_start         = _r1c1.date_input("Start date", value=pd.Timestamp("2019-01-01").date())
date_end           = _r1c2.date_input("End date",   value=pd.Timestamp.today().date())
source_benchmarks  = _r1c3.checkbox("Benchmarks",          value=True)
source_positions   = _r1c4.checkbox("Positions",           value=False)
source_aggregate   = _r1c5.checkbox("Portfolio NAV",       value=False)
source_custom      = _r1c6.checkbox("Custom",              value=False)

_r2c1, _r2c2, _r2c3, _r2c4, _r2c5 = st.columns([1.5, 1.5, 1.5, 1.5, 0.8])
cusum_k = _r2c1.slider("CUSUM k (allowance)",  0.1, 2.0, 0.3, step=0.05,
                       help="Per-step allowance before CUSUM accumulates. Lower = more sensitive.")
cusum_h = _r2c2.slider("CUSUM h (threshold)",  1.0, 15.0, 6.0, step=0.5,
                       help="Decision boundary. Higher = fewer, higher-confidence signals.")
hazard_options = {"1/50  — short (~10 wk)": 1/50,
                  "1/150 — medium (~30 wk)": 1/150,
                  "1/300 — long  (~60 wk)": 1/300}
hazard_label = _r2c3.selectbox("BOCPD hazard rate", list(hazard_options.keys()), index=1)
bocpd_hazard = hazard_options[hazard_label]
bocpd_nu     = _r2c4.slider("BOCPD ν (d.o.f.)", 3, 20, 5,
                             help="Student-t tail weight. Lower = heavier tails.")
run_btn      = _r2c5.button("Refresh", type="primary", use_container_width=True)

custom_ric = ""
if source_custom:
    custom_ric = st.text_input(
        "RIC or Yahoo ticker",
        placeholder="e.g. NVDA or NVDA.O",
        help="Try Yahoo Finance ticker first. For LSEG-only names use the full RIC.",
    )

st.divider()


# ── Constants ─────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "TLT": "20Y Treasury",
    "GLD": "Gold",
    "HYG": "High-Yield Bond",
}

_C_UP   = "#10B981"   # green  — upward break / anomaly
_C_DOWN = "#EF4444"   # red    — downward break / stress
_C_LINE = "#3B82F6"   # blue   — signal series
_C_GREY = "#64748B"   # neutral fill


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_macro(start: str, end: str, _bust: int = 3) -> dict:
    """Load HY OAS, SPY, TLT, VIX from S3 via lib.data."""
    from lib.data import get_fred_series, get_market_prices, get_market_ohlcv
    result = {}
    try:
        result["hy_oas"] = get_fred_series("BAMLH0A0HYM2").loc[start:end]
    except Exception as e:
        st.warning(f"HY OAS failed: {e}")
        result["hy_oas"] = pd.Series(dtype=float)
    for key, ticker in [("spy_close", "SPY"), ("tlt_close", "TLT"), ("vix_close", "VIX")]:
        try:
            result[key] = get_market_prices(ticker).loc[start:end]
        except Exception:
            result[key] = pd.Series(dtype=float)
    try:
        spy_ohlcv = get_market_ohlcv("SPY")
        result["spy_vol"] = spy_ohlcv["volume"].loc[start:end].dropna()
    except Exception:
        result["spy_vol"] = pd.Series(dtype=float)
    return result


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
def _load_portfolio_positions(start: str, end: str) -> dict[str, pd.Series]:
    """Load prices for active holdings from S3 EOD data via lib.data."""
    from lib.data import get_security_master, get_daily_weightings_history, get_eod_prices, _eod_data_version
    sm = get_security_master()
    wh = get_daily_weightings_history()
    if not wh.empty:
        latest = wh["date"].max()
        syms   = set(wh[wh["date"] == latest]["symbol"].tolist())
        sm     = sm[sm["ticker"].isin(syms)]
    sm = sm[sm["asset_type"] != "INDEX"]
    if sm.empty:
        return {}
    version = _eod_data_version()
    out = {}
    for _, row in sm.iterrows():
        eod = get_eod_prices(row["security_id"], version)
        if eod.empty:
            continue
        eod["date"] = pd.to_datetime(eod["date"])
        col = "adj_close" if "adj_close" in eod.columns else "close"
        s = eod.set_index("date")[col].loc[start:end].dropna().sort_index()
        if len(s) >= 60:
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
def _load_custom_yf(ticker: str, start: str, end: str) -> pd.Series:
    from lib.data import get_market_prices
    return get_market_prices(ticker).loc[start:end].rename(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_custom_lseg(ric: str, start: str, end: str) -> pd.Series:
    from lib.data import get_security_master, get_eod_prices, _eod_data_version, get_market_prices
    sm = get_security_master()
    match = sm[sm["ric"] == ric]
    if not match.empty:
        version = _eod_data_version()
        eod = get_eod_prices(match.iloc[0]["security_id"], version)
        if not eod.empty:
            eod["date"] = pd.to_datetime(eod["date"])
            col = "adj_close" if "adj_close" in eod.columns else "close"
            return eod.set_index("date")[col].loc[start:end].dropna().sort_index().rename(ric)
    ticker = ric.split(".")[0]
    s = get_market_prices(ticker).loc[start:end]
    if not s.empty:
        return s.rename(ric)
    return pd.Series(dtype=float)



# ── Signal bus ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _build_signal_bus(
    hy_oas_values: tuple, hy_oas_index: tuple,
    spy_close_values: tuple, spy_close_index: tuple,
    spy_vol_values: tuple, spy_vol_index: tuple,
    _bust: int = 3,
) -> dict:
    """
    Builds the core signal bus from macro data:
      gjr_vol, c_level, liq_level, stress_idx, level_signal
    All series are serialised as tuples for cache-key stability.
    """
    hy_oas    = pd.Series(list(hy_oas_values),    index=pd.DatetimeIndex(list(hy_oas_index)))
    spy_close = pd.Series(list(spy_close_values), index=pd.DatetimeIndex(list(spy_close_index)))
    spy_vol   = pd.Series(list(spy_vol_values),   index=pd.DatetimeIndex(list(spy_vol_index)))

    out = {}

    # GJR-GARCH vol from SPY
    try:
        gjr = GJRGarch()
        gjr.fit(spy_close)
        out["gjr_vol"]     = gjr.zscore_vol        # unclipped z-score for signal bus
        out["gjr_vol_ann"] = gjr.annualised_vol    # annualised % for display
    except Exception:
        out["gjr_vol"]     = pd.Series(0.0, index=spy_close.index)
        out["gjr_vol_ann"] = pd.Series(dtype=float)

    # Credit filter from HY OAS
    try:
        cf = CreditFilter(hy_oas)
        out["c_level"] = cf.level()
        out["c_shock"] = cf.shock()
    except Exception:
        out["c_level"] = pd.Series(dtype=float)
        out["c_shock"] = pd.Series(dtype=float)

    # Liquidity filter from SPY price + volume
    try:
        lf = LiquidityFilter(spy_close, spy_vol)
        out["liq_level"] = lf.level()
        out["liq_shock"] = lf.shock(out["liq_level"])
    except Exception:
        out["liq_level"] = pd.Series(dtype=float)
        out["liq_shock"] = pd.Series(dtype=float)

    # Stress index
    try:
        ss = StressScore({"credit_shock": out["c_shock"], "liquidity_shock": out["liq_shock"]})
        out["stress_idx"], _ = ss.run()
    except Exception:
        out["stress_idx"] = pd.Series(dtype=float)

    # Level signal — primary input to CUSUM and BOCPD
    try:
        out["level_signal"] = build_level_signal(
            out["gjr_vol"], out["c_level"], out["liq_level"]
        )
    except Exception:
        out["level_signal"] = pd.Series(dtype=float)

    return out


# ── Chart builders ─────────────────────────────────────────────────────────────

def _cusum_chart(price: pd.Series, level: pd.Series,
                 cusum_sig: pd.Series, cusum_stats: pd.DataFrame,
                 title: str, height: int = 600) -> go.Figure:
    """
    3-panel CUSUM chart:
      Panel 1: Price + coloured backdrop by CUSUM regime state
      Panel 2: Level signal (z-scored composite)
      Panel 3: CUSUM S+ and S− statistics with threshold lines
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.25, 0.30],
        vertical_spacing=0.04,
    )

    # ── Panel 1: Price + regime backdrop ──────────────────────────────────────
    price_clean = price.dropna()
    if not price_clean.empty:
        # Regime colouring: +1 = green, -1 = red, 0 = neutral
        aligned_sig = cusum_sig.reindex(price_clean.index, method="ffill").fillna(0)
        prev_state = 0
        seg_start  = price_clean.index[0]

        def _add_backdrop(x0, x1, state):
            if state == 0:
                return
            color = _rgba(_C_UP if state == 1 else _C_DOWN, 0.12)
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color,
                          line_width=0, row=1, col=1)

        for dt, sig in aligned_sig.items():
            if sig != prev_state:
                _add_backdrop(seg_start, dt, prev_state)
                seg_start = dt
                prev_state = int(sig)
        _add_backdrop(seg_start, price_clean.index[-1], prev_state)

        fig.add_trace(go.Scatter(
            x=price_clean.index, y=price_clean.values,
            mode="lines", name="Price",
            line=dict(color="#94A3B8", width=1.5),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # Break event markers
        breaks_up   = cusum_sig[cusum_sig ==  1]
        breaks_down = cusum_sig[cusum_sig == -1]
        for bx, by_label, color, sym in [
            (breaks_up,   "Upward break",   _C_UP,   "triangle-up"),
            (breaks_down, "Downward break", _C_DOWN, "triangle-down"),
        ]:
            if not bx.empty:
                prices_at = price_clean.reindex(bx.index, method="nearest")
                fig.add_trace(go.Scatter(
                    x=bx.index, y=prices_at.values,
                    mode="markers", name=by_label,
                    marker=dict(symbol=sym, size=10, color=color,
                                line=dict(width=1, color="white")),
                    showlegend=False,
                    hovertemplate=f"%{{x|%Y-%m-%d}} — {by_label}<extra></extra>",
                ), row=1, col=1)

    # ── Panel 2: Level signal ─────────────────────────────────────────────────
    lev = level.dropna()
    if not lev.empty:
        fig.add_trace(go.Scatter(
            x=lev.index, y=np.where(lev.values >= 0, lev.values, 0),
            fill="tozeroy", mode="none",
            fillcolor=_rgba(_C_DOWN, 0.25), showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=lev.index, y=np.where(lev.values < 0, lev.values, 0),
            fill="tozeroy", mode="none",
            fillcolor=_rgba(_C_UP, 0.20), showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=lev.index, y=lev.values, mode="lines",
            line=dict(color=_C_LINE, width=1.2),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)", row=2, col=1)

    # ── Panel 3: CUSUM statistics ─────────────────────────────────────────────
    if not cusum_stats.empty:
        fig.add_trace(go.Scatter(
            x=cusum_stats.index, y=cusum_stats["S_pos"].values,
            mode="lines", name="S+",
            line=dict(color=_C_UP, width=1.5),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d} S+: %{y:.3f}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=cusum_stats.index, y=cusum_stats["S_neg"].values,
            mode="lines", name="S−",
            line=dict(color=_C_DOWN, width=1.5),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d} S−: %{y:.3f}<extra></extra>",
        ), row=3, col=1)
        # Threshold lines
        fig.add_hline(y=0,  line_width=1, line_color="rgba(255,255,255,0.2)", row=3, col=1)
        fig.add_hline(y=cusum_stats["S_pos"].max() if not cusum_stats.empty else 6,
                      line_width=0)  # dummy — use annotation instead
        fig.add_hline(y=0,  line_width=0, row=3, col=1)

    title_text = (
        f"<b>{title} — CUSUM</b>"
        f"<br><span style='font-size:11px;color:#94A3B8'>"
        f"k={cusum_k:.2f} · h={cusum_h:.1f} · "
        f"green backdrop = upward regime · red backdrop = downward regime"
        f"</span>"
    )
    fig.update_layout(
        template="capital",
        title=dict(text=title_text, x=0, xanchor="left"),
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=72, b=40),
        xaxis3=dict(showgrid=False),
    )
    fig.update_yaxes(zeroline=False)
    fig.update_xaxes(showgrid=False)

    # Row labels
    fig.update_yaxes(title_text="Price",        row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Level signal", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="S+ / S−",      row=3, col=1, title_font=dict(size=10))

    return fig


def _bocpd_chart(price: pd.Series, bocpd_norm: pd.Series, bocpd_raw: pd.Series,
                 title: str, height: int = 580) -> go.Figure:
    """
    3-panel BOCPD chart:
      Panel 1: Price + high-anomaly backdrop (top 20% of normalised surprise)
      Panel 2: Normalised surprise (0–1) filled
      Panel 3: Raw changepoint probability
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25],
        vertical_spacing=0.04,
    )

    price_clean = price.dropna()
    norm_clean  = bocpd_norm.dropna()
    raw_clean   = bocpd_raw.dropna()

    # ── Panel 1: Price + anomaly backdrop ────────────────────────────────────
    if not price_clean.empty and not norm_clean.empty:
        threshold = norm_clean.quantile(0.80)
        aligned   = norm_clean.reindex(price_clean.index, method="ffill").fillna(0)
        in_anom   = False
        seg_start = price_clean.index[0]

        def _add_anom(x0, x1, active):
            if active:
                fig.add_vrect(x0=x0, x1=x1,
                              fillcolor=_rgba(_C_DOWN, 0.14),
                              line_width=0, row=1, col=1)

        for dt, val in aligned.items():
            now_anom = val >= threshold
            if now_anom != in_anom:
                _add_anom(seg_start, dt, in_anom)
                seg_start = dt
                in_anom = now_anom
        _add_anom(seg_start, price_clean.index[-1], in_anom)

        fig.add_trace(go.Scatter(
            x=price_clean.index, y=price_clean.values,
            mode="lines",
            line=dict(color="#94A3B8", width=1.5),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Panel 2: Normalised surprise (0–1) ────────────────────────────────────
    if not norm_clean.empty:
        fig.add_trace(go.Scatter(
            x=norm_clean.index, y=norm_clean.values,
            fill="tozeroy", mode="lines",
            fillcolor=_rgba(_C_DOWN, 0.22),
            line=dict(color=_C_DOWN, width=1.2),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
        ), row=2, col=1)
        threshold = norm_clean.quantile(0.80)
        fig.add_hline(y=threshold, line_width=1, line_dash="dot",
                      line_color="rgba(239,68,68,0.5)", row=2, col=1)

    # ── Panel 3: Raw changepoint probability ──────────────────────────────────
    if not raw_clean.empty:
        fig.add_trace(go.Scatter(
            x=raw_clean.index, y=raw_clean.values,
            fill="tozeroy", mode="lines",
            fillcolor=_rgba(_C_GREY, 0.25),
            line=dict(color=_C_GREY, width=1.0),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>",
        ), row=3, col=1)

    hazard_disp = f"hazard=1/{round(1/bocpd_hazard)}" if bocpd_hazard > 0 else "hazard=0"
    title_text = (
        f"<b>{title} — BOCPD</b>"
        f"<br><span style='font-size:11px;color:#94A3B8'>"
        f"{hazard_disp} · ν={bocpd_nu} · "
        f"red backdrop = top-20% anomaly · dotted line = 80th-percentile threshold"
        f"</span>"
    )
    fig.update_layout(
        template="capital",
        title=dict(text=title_text, x=0, xanchor="left"),
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=72, b=40),
    )
    fig.update_yaxes(zeroline=False)
    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(title_text="Price",         row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Surprise (0–1)", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="P(changepoint)", row=3, col=1, title_font=dict(size=10))

    return fig


# ── Run pipeline ──────────────────────────────────────────────────────────────

def _run_detectors(price: pd.Series, level: pd.Series,
                   k: float, h: float, hazard: float, nu: float) -> dict:
    """Run CUSUM + BOCPD on the level signal; price is used for plotting only."""
    cusum   = CUSUMDetector(k=k, h=h)
    cusum_sig   = cusum.detect(level)
    cusum_stats = cusum.stat_series(level)

    bocpd      = StudentTBOCPD(hazard=hazard, nu=nu)
    bocpd_norm = bocpd.run(level)
    bocpd_raw  = bocpd.raw_probs

    return {
        "price":       price,
        "level":       level,
        "cusum_sig":   cusum_sig,
        "cusum_stats": cusum_stats,
        "bocpd_norm":  bocpd_norm,
        "bocpd_raw":   bocpd_raw,
    }


def _summary_table(results: dict):
    """
    Render a clean summary table — one row per asset.
    Columns: Asset | Regime | Last break | CUSUM ↑ / ↓ | BOCPD surprise | Anomaly level
    """
    STATE_ICON  = {1: "🟢", -1: "🔴", 0: "⚪"}
    STATE_LABEL = {1: "Upward break", -1: "Downward break", 0: "Stable"}

    rows = []
    for name, r in results.items():
        sig         = r["cusum_sig"]
        last_breaks = sig[sig != 0]
        last_date   = last_breaks.index[-1].strftime("%d %b %Y") if not last_breaks.empty else "—"
        n_up        = int((sig ==  1).sum())
        n_down      = int((sig == -1).sum())
        state       = int(last_breaks.iloc[-1]) if not last_breaks.empty else 0

        norm    = r["bocpd_norm"].dropna()
        surprise     = float(norm.iloc[-1])    if not norm.empty else 0.0
        surprise_pct = float((norm <= norm.iloc[-1]).mean() * 100) if not norm.empty else 0.0

        # Anomaly bar: filled blocks proportional to surprise percentile
        bar_filled = round(surprise_pct / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)

        rows.append({
            "Asset":          name,
            "state":          state,
            "Regime":         f"{STATE_ICON[state]} {STATE_LABEL[state]}",
            "Last break":     last_date,
            "CUSUM ↑ / ↓":   f"{n_up} / {n_down}",
            "Surprise score": f"{surprise:.2f}",
            "Anomaly level":  f"{bar}  {surprise_pct:.0f}th pct",
        })

    # Header row
    cols = st.columns([2, 2, 1.6, 1.2, 1.4, 2.5])
    headers = ["Asset", "Regime", "Last break", "CUSUM ↑/↓", "Surprise", "Anomaly level"]
    for col, h in zip(cols, headers):
        col.markdown(f"<span style='color:#94A3B8;font-size:11px;text-transform:uppercase;"
                     f"letter-spacing:0.05em'>{h}</span>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:4px 0 8px;border-color:#1E293B'>", unsafe_allow_html=True)

    for row in rows:
        state = row["state"]
        name_color = {"Upward break": "#10B981",
                      "Downward break": "#EF4444",
                      "Stable": "#CBD5E1"}[STATE_LABEL[state]]

        cols = st.columns([2, 2, 1.6, 1.2, 1.4, 2.5])
        cols[0].markdown(f"**{row['Asset']}**")
        cols[1].markdown(
            f"<span style='color:{name_color};font-weight:600'>{row['Regime']}</span>",
            unsafe_allow_html=True,
        )
        cols[2].markdown(row["Last break"])
        cols[3].markdown(row["CUSUM ↑ / ↓"])
        cols[4].markdown(row["Surprise score"])
        cols[5].markdown(
            f"<span style='font-family:monospace;font-size:12px'>{row['Anomaly level']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:2px 0;border-color:#1E293B'>", unsafe_allow_html=True)


# ── Auto-run / session state ──────────────────────────────────────────────────

_sources_key = (
    source_benchmarks, source_positions, source_aggregate,
    source_custom, custom_ric.strip(),
)
_params_key = (
    str(date_start), str(date_end),
    cusum_k, cusum_h, bocpd_hazard, bocpd_nu,
    *_sources_key,
)

_needs_run = (
    run_btn
    or st.session_state.get("sb_result") is None
    or st.session_state.get("sb_params") != _params_key
)

if _needs_run:
    s_str = str(date_start)
    e_str = str(date_end)

    with st.spinner("Loading macro data…"):
        macro = _load_macro(s_str, e_str)

    hy_oas    = macro.get("hy_oas",    pd.Series(dtype=float))
    spy_close = macro.get("spy_close", pd.Series(dtype=float))
    spy_vol   = macro.get("spy_vol",   pd.Series(dtype=float))

    if spy_close.empty:
        st.error("SPY data unavailable — cannot build signal bus.")
        st.stop()

    with st.spinner("Building signal bus (GJR-GARCH + Credit + Liquidity)…"):
        bus = _build_signal_bus(
            tuple(hy_oas.values),    tuple(hy_oas.index),
            tuple(spy_close.values), tuple(spy_close.index),
            tuple(spy_vol.values),   tuple(spy_vol.index),
        )

    level_signal = bus.get("level_signal", pd.Series(dtype=float))
    if level_signal.empty:
        st.error("Level signal is empty — check FRED and yfinance connectivity.")
        st.stop()

    # ── Assemble target series ──────────────────────────────────────────────
    targets: dict[str, pd.Series] = {}

    if source_benchmarks:
        with st.spinner("Loading benchmark prices…"):
            bench_df = _load_benchmark_prices(s_str, e_str)
        for col in bench_df.columns:
            targets[f"{col} ({BENCHMARKS.get(col, col)})"] = bench_df[col].dropna()

    if source_positions:
        with st.spinner("Loading portfolio positions via LSEG…"):
            positions = _load_portfolio_positions(s_str, e_str)
        targets.update(positions)

    if source_aggregate:
        with st.spinner("Loading portfolio aggregate…"):
            port_nav = _load_portfolio_aggregate(s_str, e_str)
        if not port_nav.empty:
            targets["Portfolio NAV"] = port_nav

    if source_custom and custom_ric.strip():
        ric = custom_ric.strip()
        with st.spinner(f"Loading {ric}…"):
            try:
                custom_series = _load_custom_yf(ric, s_str, e_str)
                if custom_series.empty:
                    raise ValueError("empty from yfinance")
                targets[ric] = custom_series
            except Exception:
                try:
                    custom_series = _load_custom_lseg(ric, s_str, e_str)
                    if not custom_series.empty:
                        targets[ric] = custom_series
                    else:
                        st.warning(f"No data found for '{ric}'.")
                except Exception as e2:
                    st.warning(f"Custom fetch failed: {e2}")

    # ── Run detectors on each target ────────────────────────────────────────
    all_results: dict[str, dict] = {}
    for name, price_series in targets.items():
        aligned_level = level_signal.reindex(
            level_signal.index.union(price_series.index)
        ).interpolate("time").reindex(level_signal.index)

        res = _run_detectors(
            price_series, aligned_level,
            k=cusum_k, h=cusum_h,
            hazard=bocpd_hazard, nu=bocpd_nu,
        )
        all_results[name] = res

    st.session_state["sb_result"]  = {
        "results":      all_results,
        "bus":          bus,
        "macro":        macro,
        "level_signal": level_signal,
        "start":        s_str,
        "end":          e_str,
        "k":            cusum_k,
        "h":            cusum_h,
        "hazard":       bocpd_hazard,
        "nu":           bocpd_nu,
    }
    st.session_state["sb_params"] = _params_key


# ── Render ────────────────────────────────────────────────────────────────────

sb = st.session_state["sb_result"]

results      = sb["results"]
level_signal = sb["level_signal"]
bus          = sb["bus"]

if not results:
    st.warning("No price series loaded. Enable at least one source in the sidebar.")
    st.stop()

st.caption(
    f"{sb['start']} → {sb['end']}  ·  "
    f"CUSUM k={sb['k']:.2f} h={sb['h']:.1f}  ·  "
    f"BOCPD hazard=1/{round(1/sb['hazard'])} ν={sb['nu']}"
)

# ── Macro signal bus overview ─────────────────────────────────────────────────

with st.expander("Macro signal bus", expanded=True):
    stress = bus.get("stress_idx", pd.Series(dtype=float)).dropna()
    gjr    = bus.get("gjr_vol",    pd.Series(dtype=float)).dropna()
    lev_s  = level_signal.dropna()

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.metric(
            "Stress index (latest)",
            f"{stress.iloc[-1]:.3f}" if not stress.empty else "—",
            delta=f"vs 1M avg: {stress.iloc[-1] - stress.iloc[-21:].mean():+.3f}" if len(stress) > 21 else None,
        )
    gjr_ann = bus.get("gjr_vol_ann", pd.Series(dtype=float)).dropna()
    with ec2:
        st.metric(
            "GJR-GARCH vol (ann.)",
            f"{gjr_ann.iloc[-1]:.1%}" if not gjr_ann.empty else "—",
        )
    with ec3:
        st.metric(
            "Level signal (latest)",
            f"{lev_s.iloc[-1]:.3f}" if not lev_s.empty else "—",
            delta=f"{lev_s.iloc[-1]:+.3f}σ",
            delta_color="inverse",
        )

    if not lev_s.empty:
        fig_bus = go.Figure()
        fig_bus.add_trace(go.Scatter(
            x=lev_s.index, y=np.where(lev_s.values >= 0, lev_s.values, 0),
            fill="tozeroy", mode="none",
            fillcolor=_rgba(_C_DOWN, 0.20), showlegend=False,
        ))
        fig_bus.add_trace(go.Scatter(
            x=lev_s.index, y=np.where(lev_s.values < 0, lev_s.values, 0),
            fill="tozeroy", mode="none",
            fillcolor=_rgba(_C_UP, 0.15), showlegend=False,
        ))
        fig_bus.add_trace(go.Scatter(
            x=lev_s.index, y=lev_s.values, mode="lines",
            line=dict(color=_C_LINE, width=1.5),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}σ<extra></extra>",
        ))
        fig_bus.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
        fig_bus.update_layout(
            template="capital",
            title=dict(
                text="<b>Composite level signal</b>"
                     "<br><span style='font-size:11px;color:#94A3B8'>"
                     "Expanding z-score of 10d-smoothed (GJR-vol + credit level + liquidity level)"
                     "</span>",
                x=0, xanchor="left",
            ),
            height=220,
            showlegend=False,
            margin=dict(l=50, r=20, t=64, b=30),
            yaxis=dict(zeroline=False),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_bus, use_container_width=True)

# ── Per-asset results ─────────────────────────────────────────────────────────

st.subheader("Regime summary")
_summary_table(results)

st.divider()

for name, r in results.items():
    st.subheader(name)
    tab_cusum, tab_bocpd = st.tabs(["CUSUM", "BOCPD"])

    with tab_cusum:
        st.plotly_chart(
            _cusum_chart(
                r["price"], r["level"],
                r["cusum_sig"], r["cusum_stats"],
                title=name, height=580,
            ),
            use_container_width=True,
        )
        st.markdown(
            "<small>"
            "**Green backdrop** = CUSUM detected upward stress shift (S+ > h, then reset).  "
            "**Red backdrop** = downward stress shift detected.  "
            "**Triangles** mark break events.  "
            "Panel 3 shows the running S+ (green) and S− (red) statistics."
            "</small>",
            unsafe_allow_html=True,
        )

    with tab_bocpd:
        st.plotly_chart(
            _bocpd_chart(
                r["price"], r["bocpd_norm"], r["bocpd_raw"],
                title=name, height=560,
            ),
            use_container_width=True,
        )
        st.markdown(
            "<small>"
            "**Red backdrop** = normalised changepoint surprise above 80th percentile.  "
            "**Dotted line** = 80th-percentile threshold.  "
            "Panel 2: normalised surprise (0–1) smoothed over 20 days.  "
            "Panel 3: raw P(changepoint) from Student-t run-length distribution."
            "</small>",
            unsafe_allow_html=True,
        )
    st.write("")
