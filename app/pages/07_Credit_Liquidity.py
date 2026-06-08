import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
for _mod in list(sys.modules.keys()):
    if _mod in ("lib.credit_liquidity",) or _mod.startswith("lib.credit_liquidity."):
        del sys.modules[_mod]

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from lib.theme import inject_css, FAVICON
from lib.credit_liquidity import CreditFilter, LiquidityFilter, StressScore

st.set_page_config(page_title="Credit & Liquidity · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Credit & Liquidity Monitor")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    date_start = st.date_input("Start date", value=pd.Timestamp("2016-01-01").date())
    date_end   = st.date_input("End date",   value=pd.Timestamp.today().date())
    window     = st.slider("Z-score window (trading days)", 126, 756, 504,
                           help="504 ≈ 2 years. Longer = more stable baselines.")
    run_btn = st.button("Run", type="primary", use_container_width=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch(start: str, end: str, _bust: int = 2) -> dict:
    from lib.credit_liquidity import load_fred, load_spy_lseg  # fresh import each call
    import yfinance as yf
    errors = []

    hy_oas = pd.Series(dtype=float)
    try:
        hy_oas = load_fred("BAMLH0A0HYM2", start, end)
    except Exception as e:
        errors.append(f"FRED HY OAS: {e}")

    spy_close = spy_volume = pd.Series(dtype=float)
    try:
        spy_close, spy_volume = load_spy_lseg(start, end)
    except Exception as e:
        errors.append(f"LSEG SPY: {e}")

    spx = pd.Series(dtype=float)
    try:
        raw = yf.download("^GSPC", start=start, end=end, progress=False)["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        spx = raw.squeeze().sort_index().astype(float)
        spx.index = pd.to_datetime(spx.index)
    except Exception as e:
        errors.append(f"yfinance SPX: {e}")

    return {"hy_oas": hy_oas, "spy_close": spy_close,
            "spy_volume": spy_volume, "spx": spx, "errors": errors}


# ── Session state ─────────────────────────────────────────────────────────────

if "cl_result" not in st.session_state:
    st.session_state["cl_result"] = None

if run_btn:
    with st.spinner("Fetching data and running filters…"):
        raw = _fetch(str(date_start), str(date_end), _bust=2)
        for e in raw["errors"]:
            st.sidebar.warning(e)

        if raw["hy_oas"].empty and raw["spy_close"].empty:
            st.error("No data loaded — check LSEG and FRED connectivity.")
            st.stop()

        signals = {}
        if not raw["hy_oas"].empty:
            signals.update(CreditFilter(raw["hy_oas"], window=window).run())
        if not raw["spy_close"].empty and not raw["spy_volume"].empty:
            signals.update(LiquidityFilter(raw["spy_close"], raw["spy_volume"], window=window).run())

        stress_z = stress_p = pd.Series(dtype=float)
        if len(signals) >= 2:
            ss = StressScore(signals)
            stress_z, stress_p = ss.run()

        # Empirical significance: top/bottom 5% of stress score history
        threshold_hi = threshold_lo = np.nan
        if not stress_z.empty:
            threshold_hi = float(stress_z.dropna().quantile(0.95))
            threshold_lo = np.nan

        st.session_state["cl_result"] = {
            "signals":      signals,
            "stress_z":     stress_z,
            "spx":          raw["spx"],
            "hy_oas":       raw["hy_oas"],
            "threshold_hi": threshold_hi,
            "threshold_lo": threshold_lo,
            "window":       window,
        }

res = st.session_state["cl_result"]
if res is None:
    st.info("Set a date range and click **Run**.")
    st.stop()

signals      = res["signals"]
stress_z     = res["stress_z"]
spx          = res["spx"]
hy_oas       = res["hy_oas"]
threshold_hi = res["threshold_hi"]
threshold_lo = res["threshold_lo"]


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _spx_trace(spx: pd.Series, common_idx) -> go.Scatter:
    return go.Scatter(
        x=spx.loc[common_idx].index,
        y=spx.loc[common_idx].values,
        mode="lines",
        line=dict(width=1, color="rgba(148,163,184,0.35)"),
        name="S&P 500",
        hovertemplate="SPX: %{y:,.0f}<extra></extra>",
        yaxis="y2",
    )


def _signal_chart(signal: pd.Series, title: str, color: str, height: int = 300) -> go.Figure:
    s = signal.dropna()
    fig = go.Figure()

    # Green/red fill
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s >= 0, s.values, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(239,68,68,0.15)", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s < 0, s.values, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(16,185,129,0.15)", showlegend=False,
    ))

    # Signal line
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(width=2, color=color),
        name=title,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
    ))

    # Reference lines at ±1.5
    for lvl in [1.5, -1.5]:
        fig.add_hline(y=lvl, line_width=0.8,
                      line_color="rgba(255,255,255,0.2)", line_dash="dot")

    fig.update_layout(
        template="capital", height=height,
        title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", font=dict(size=14)),
        margin=dict(l=50, r=30, t=48, b=36),
        showlegend=False,
        yaxis=dict(title="Z-score", zeroline=False, tickfont=dict(size=11)),
    )
    return fig


def _stress_chart(stress_z: pd.Series, spx: pd.Series,
                  thr_hi: float, thr_lo: float, height: int = 380) -> go.Figure:
    z = stress_z.dropna()
    fig = go.Figure()

    # Highlight top-5% stress periods in red
    is_extreme = z >= thr_hi
    for start_i, end_i in _contiguous(is_extreme):
        fig.add_vrect(
            x0=z.index[start_i], x1=z.index[min(end_i, len(z.index)-1)],
            fillcolor="rgba(239,68,68,0.18)", line_width=0, layer="below",
        )

    # Fill
    fig.add_trace(go.Scatter(
        x=z.index, y=np.where(z >= 0, z.values, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(239,68,68,0.12)", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=z.index, y=np.where(z < 0, z.values, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(16,185,129,0.12)", showlegend=False,
    ))

    # Stress line
    fig.add_trace(go.Scatter(
        x=z.index, y=z.values, mode="lines",
        line=dict(width=2.5, color="#F59E0B"),
        name="Stress score",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
    ))

    # Threshold lines
    if not np.isnan(thr_hi):
        fig.add_hline(y=thr_hi, line_width=1.2,
                      line_color="rgba(239,68,68,0.55)", line_dash="dash")

    fig.update_layout(
        template="capital", height=height,
        title=dict(
            text=(
                "<b>Composite Stress Score</b>"
                "  <span style='font-size:12px;color:#94A3B8'>"
                "= |0.5 × (credit shock + liquidity shock)| — "
                "<span style='color:#EF4444'>red bands = top/bottom 5% of historical days</span>"
                "</span>"
            ),
            x=0, xanchor="left",
        ),
        showlegend=False,
        margin=dict(l=50, r=30, t=72, b=36),
        yaxis=dict(title="Stress Z-score", zeroline=False),
    )
    return fig


def _contiguous(mask: pd.Series):
    """Yield (start, end) index pairs for contiguous True runs."""
    arr = mask.values
    in_run = False
    start = 0
    for i, v in enumerate(arr):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            yield start, i
            in_run = False
    if in_run:
        yield start, len(arr)


# ─────────────────────────────────────────────────────────────────────────────
# STRESS SCORE
# ─────────────────────────────────────────────────────────────────────────────

if not stress_z.empty:
    latest_z   = float(stress_z.dropna().iloc[-1])
    is_extreme = latest_z >= threshold_hi
    pct_rank   = float((stress_z.dropna() <= latest_z).mean()) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stress Z-score",   f"{latest_z:+.2f}")
    m2.metric("Percentile rank",  f"{pct_rank:.0f}th")
    m3.metric("Top-5% stress",    "YES" if latest_z >= threshold_hi else "NO",
              delta="⚠ elevated" if latest_z >= threshold_hi else None,
              delta_color="inverse")
    m4.metric("95th pct threshold", f"{threshold_hi:.2f}")
    st.write("")
    st.plotly_chart(_stress_chart(stress_z, spx, threshold_hi, threshold_lo, height=400),
                    use_container_width=True)
    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CREDIT  |  LIQUIDITY  — side by side per row
# ─────────────────────────────────────────────────────────────────────────────

PAIRS = [
    (("credit_level",    "Credit Level",        "#EF4444"),
     ("liquidity_level", "Liquidity Level",      "#3B82F6")),
    (("credit_shock",    "Credit Shock",         "#F97316"),
     ("liquidity_shock", "Liquidity Shock",      "#8B5CF6")),
    (("credit_accel",    "Credit Acceleration",  "#FBBF24"),
     ("liquidity_accel", "Liquidity Acceleration","#EC4899")),
]

ROW_LABELS = ["Level", "Shock", "Acceleration"]
ROW_DESCS  = [
    "Z-score of the raw level vs {w}-day rolling mean — slow-moving regime indicator",
    "Z-score of the smoothed daily change — picks up rapid deterioration",
    "Z-score of the rate-of-change of the shock — early turning-point signal",
]

# Column headers
hc1, hc2 = st.columns(2, gap="large")
hc1.markdown("### Credit  *(HY OAS, FRED)*")
hc2.markdown("### Liquidity  *(SPY Amihud + Volume)*")
st.write("")

for i, ((ck, cl, cc), (lk, ll, lc)) in enumerate(PAIRS):
    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        if ck in signals:
            st.plotly_chart(
                _signal_chart(signals[ck], cl, cc, height=280),
                use_container_width=True,
            )
        else:
            st.info(f"{cl}: no data")

    with c_right:
        if lk in signals:
            st.plotly_chart(
                _signal_chart(signals[lk], ll, lc, height=280),
                use_container_width=True,
            )
        else:
            st.info(f"{ll}: no data")

    st.caption(ROW_DESCS[i].replace("{w}", str(res["window"])))
    st.write("")

# Raw HY OAS
if not hy_oas.empty:
    st.divider()
    with st.expander("Raw HY OAS (basis points)", expanded=False):
        fig_raw = go.Figure(go.Scatter(
            x=hy_oas.index, y=hy_oas.values, mode="lines",
            line=dict(width=1.5, color="#EF4444"),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.0f} bps<extra></extra>",
        ))
        fig_raw.update_layout(
            template="capital", height=200,
            title="<b>ICE BofA US HY Option-Adjusted Spread</b>",
            yaxis_title="bps",
            margin=dict(l=50, r=20, t=48, b=32),
        )
        st.plotly_chart(fig_raw, use_container_width=True)
