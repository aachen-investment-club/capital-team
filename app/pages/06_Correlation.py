import sys
import pathlib
import importlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

for _mod in list(sys.modules.keys()):
    if _mod in ("lib.correlation",) or _mod.startswith("lib.correlation."):
        del sys.modules[_mod]

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from lib.theme import inject_css, FAVICON
from lib.correlation import RollingCorr, PartialCorr

st.set_page_config(page_title="Correlation · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Correlation Analysis")
with st.popover("ℹ"):
    st.markdown("""Measures how much our assets move together on a rolling basis. Useful for spotting when supposed diversifiers are no longer diversifying, or when a macro regime shift is causing previously uncorrelated assets to move in lockstep.""")

# ── Controls ──────────────────────────────────────────────────────────────────

_c1, _c2, _c3, _c4, _c5 = st.columns([1.2, 1.2, 1.2, 1.2, 0.8])
date_start    = _c1.date_input("Start date", value=pd.Timestamp("2020-01-01").date())
date_end      = _c2.date_input("End date",   value=pd.Timestamp.today().date())
window_corr   = _c3.slider("Rolling window (days)",  10, 120, 30)
window_smooth = _c4.slider("Smoothing window (days)",  5,  40, 10)
run_btn       = _c5.button("Run", type="primary", use_container_width=True)

# ── Fixed pairs ───────────────────────────────────────────────────────────────

FIXED_PAIRS = [
    ("SPY", "IWM",  None,  "Breadth",  "S&P 500 vs Russell 2000 — market internals"),
    ("SPY", "TLT",  None,  "Macro",    "Equities vs Long Treasury — risk-on/off"),
    ("SPY", "VIX",  None,  "Stress",   "S&P 500 vs VIX — fear gauge"),
    ("HYG", "TLT",  None,  "Liquidity","High-Yield vs Treasury — credit/duration spread"),
    ("IWM", "HYG", "SPY", "Partial: Small-cap ↔ Credit",
     "IWM vs HYG controlling for SPY — idiosyncratic credit sensitivity"),
]

LSEG_MAP = {
    "SPY": "SPY.P",
    "IWM": "IWM.P",
    "TLT": "TLT.P",
    "HYG": "HYG.P",
    "VIX": ".VIX",
    "XLU": "XLU.P",
}

# ── Data loading via LSEG ────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices from LSEG (TR.PriceClose) + yfinance for VIX."""
    import lseg.data as ld
    import yfinance as yf
    from dotenv import load_dotenv
    import pathlib as _pl
    load_dotenv(_pl.Path(__file__).resolve().parent.parent.parent / ".env")

    cols = {}

    # VIX via yfinance (LSEG doesn't carry it reliably)
    if "VIX" in tickers:
        try:
            vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
            if isinstance(vix, pd.DataFrame):
                vix = vix.iloc[:, 0]
            vix.index = pd.to_datetime(vix.index)
            cols["VIX"] = vix.rename("VIX")
        except Exception as e:
            st.warning(f"VIX fetch failed: {e}")

    # All other tickers via LSEG (one call per ticker)
    lseg_tickers = [t for t in tickers if t != "VIX"]
    if lseg_tickers:
        try:
            ld.open_session()
            for ticker in lseg_tickers:
                ric = LSEG_MAP.get(ticker, ticker)
                try:
                    raw = ld.get_history(
                        universe=[ric],
                        fields=["TR.PriceClose"],
                        parameters={"Adjusted": 1},
                        start=start,
                        end=end,
                        interval="1D",
                    )
                    if raw is not None and not raw.empty:
                        s = raw.copy()
                        s.index = pd.to_datetime(s.index)
                        s = s.apply(pd.to_numeric, errors="coerce").dropna(how="all")
                        cols[ticker] = s.iloc[:, 0].rename(ticker)
                except Exception as e:
                    st.warning(f"{ticker} fetch failed: {e}")
            try:
                ld.close_session()
            except Exception:
                pass
        except Exception as e:
            st.error(f"LSEG session error: {e}")
            return pd.DataFrame()

    if not cols:
        return pd.DataFrame()

    df = pd.DataFrame(cols).apply(pd.to_numeric, errors="coerce").dropna()
    return df


def _line(series: pd.Series, title: str, subtitle: str,
          yrange=(-1.05, 1.05), zero_line: bool = True, height: int = 300) -> go.Figure:
    s = series.dropna()
    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(width=2, color="#3B82F6"),
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ))
    if zero_line:
        fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
    # Shade positive / negative regions
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s.values >= 0, s.values, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(16,185,129,0.15)", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=s.index, y=np.where(s.values < 0, s.values, 0),
        fill="tozeroy", mode="none",
        fillcolor="rgba(239,68,68,0.15)", showlegend=False,
    ))
    title_text = (
        f"<b>{title}</b>"
        f"<br><span style='font-size:11px;color:#94A3B8'>{subtitle}</span>"
    )
    fig.update_layout(
        template="capital",
        title=dict(text=title_text, x=0, xanchor="left"),
        height=height,
        yaxis=dict(range=list(yrange), tickformat=".2f", zeroline=False),
        xaxis=dict(showgrid=False),
        margin=dict(l=50, r=20, t=72, b=40),
        showlegend=False,
    )
    return fig


# ── Session state ─────────────────────────────────────────────────────────────

if "corr_result" not in st.session_state:
    st.session_state["corr_result"] = None

if run_btn:
    all_tickers = tuple(sorted({t for pair in FIXED_PAIRS for t in pair[:3] if t}))
    with st.spinner("Fetching prices and computing correlations…"):
        prices = _load_prices(all_tickers, str(date_start), str(date_end))
        if prices.empty:
            st.error("No price data returned. Check LSEG connection.")
            st.stop()

        log_ret = np.log(prices / prices.shift(1)).dropna()

        results = {}
        for x, y, ctrl, label, desc in FIXED_PAIRS:
            if x not in prices.columns or y not in prices.columns:
                continue
            if ctrl is None:
                rc  = RollingCorr(prices[x], prices[y], window_corr, window_smooth)
                results[label] = {
                    "series": rc.run_rho(),
                    "desc":   desc,
                    "type":   "rolling",
                    "x": x, "y": y, "ctrl": None,
                }
            else:
                if ctrl not in prices.columns:
                    continue
                pc = PartialCorr(prices[x], prices[y], prices[ctrl])
                results[label] = {
                    "series": pc.rolling(window=window_corr),
                    "desc":   desc,
                    "type":   "partial",
                    "x": x, "y": y, "ctrl": ctrl,
                }

        st.session_state["corr_result"] = {
            "results":  results,
            "prices":   prices,
            "log_ret":  log_ret,
            "start":    str(date_start),
            "end":      str(date_end),
        }

res = st.session_state["corr_result"]
if res is None:
    st.info("Set a date range and click **Run**.")
    st.stop()

results  = res["results"]
prices   = res["prices"]
log_ret  = res["log_ret"]

# ─────────────────────────────────────────────────────────────────────────────
# FIXED CHARTS
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Market Correlation Monitor")
st.caption(f"{res['start']} → {res['end']}  ·  rolling {window_corr}d, smooth {window_smooth}d")
st.write("")

CHART_ORDER = ["Macro", "Stress", "Breadth", "Liquidity", "Partial: Small-cap ↔ Credit"]

# Row 1: Macro + Stress
c1, c2 = st.columns(2, gap="large")
pairs_row1 = [k for k in CHART_ORDER[:2] if k in results]
for col, label in zip([c1, c2], pairs_row1):
    r = results[label]
    s = r["series"]
    tag = "(partial)" if r["type"] == "partial" else ""
    ctrl_note = f" | control: {r['ctrl']}" if r["ctrl"] else ""
    with col:
        st.plotly_chart(
            _line(s, f"{r['x']} ↔ {r['y']} {tag}",
                  r["desc"] + ctrl_note, height=320),
            use_container_width=True,
        )

st.write("")

# Row 2: Breadth + Liquidity
c3, c4 = st.columns(2, gap="large")
pairs_row2 = [k for k in CHART_ORDER[2:4] if k in results]
for col, label in zip([c3, c4], pairs_row2):
    r = results[label]
    tag = "(partial)" if r["type"] == "partial" else ""
    ctrl_note = f" | control: {r['ctrl']}" if r["ctrl"] else ""
    with col:
        st.plotly_chart(
            _line(r["series"], f"{r['x']} ↔ {r['y']} {tag}",
                  r["desc"] + ctrl_note, height=320),
            use_container_width=True,
        )

st.write("")

# Row 3: Partial (full width)
if "Partial: Small-cap ↔ Credit" in results:
    r = results["Partial: Small-cap ↔ Credit"]
    st.plotly_chart(
        _line(r["series"], f"{r['x']} ↔ {r['y']}  (partial, controlling for {r['ctrl']})",
              r["desc"], height=280),
        use_container_width=True,
    )



# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM PARTIAL CORRELATION EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
with st.expander("Custom partial correlation explorer", expanded=False):
    st.caption(
        "Enter any three tickers available in the data. "
        "Computes partial correlation of X and Y after removing the influence of Control."
    )

    available = sorted(prices.columns.tolist())
    ce1, ce2, ce3, ce4 = st.columns([1, 1, 1, 1])
    with ce1:
        custom_x = st.selectbox("X", available, index=available.index("SPY") if "SPY" in available else 0)
    with ce2:
        custom_y = st.selectbox("Y", available, index=available.index("TLT") if "TLT" in available else 1)
    with ce3:
        ctrl_opts = ["(none)"] + available
        custom_ctrl = st.selectbox("Control", ctrl_opts,
                                   index=ctrl_opts.index("(none)"))
    with ce4:
        custom_window = st.number_input("Window (days)", min_value=10, max_value=252, value=60)

    custom_label = st.text_input(
        "Custom tickers to add (comma-separated LSEG RICs)",
        placeholder="e.g. XLU.P, GLD.P",
        help="These will be fetched from LSEG and added to the available list.",
    )

    if custom_label.strip():
        extra_rics = [r.strip() for r in custom_label.split(",") if r.strip()]
        extra_map  = {r.replace(".P", "").replace(".", "_"): r for r in extra_rics}
        with st.spinner("Fetching extra tickers…"):
            extra_prices = _load_prices(
                tuple(extra_map.values()),
                res["start"], res["end"],
            )
            if not extra_prices.empty:
                prices   = pd.concat([prices, extra_prices], axis=1).dropna()
                available = sorted(prices.columns.tolist())
                st.success(f"Added: {', '.join(extra_prices.columns.tolist())}")

    if st.button("Compute", key="custom_run"):
        if custom_x not in prices.columns or custom_y not in prices.columns:
            st.warning("One or both tickers not available in loaded data.")
        elif custom_x == custom_y:
            st.warning("X and Y must be different.")
        else:
            if custom_ctrl == "(none)":
                rc_custom = RollingCorr(prices[custom_x], prices[custom_y],
                                        int(custom_window), window_smooth)
                s_custom  = rc_custom.run_rho()
                title_c   = f"{custom_x} ↔ {custom_y}  (rolling)"
                sub_c     = f"Rolling {custom_window}d correlation"
            else:
                if custom_ctrl not in prices.columns:
                    st.warning(f"Control ticker '{custom_ctrl}' not in data.")
                    st.stop()
                pc_custom = PartialCorr(prices[custom_x], prices[custom_y],
                                        prices[custom_ctrl])
                s_custom  = pc_custom.rolling(window=int(custom_window))
                title_c   = f"{custom_x} ↔ {custom_y}  (partial, control: {custom_ctrl})"
                sub_c     = f"Partial correlation removing {custom_ctrl}'s influence · window={custom_window}d"

            scalar = float(s_custom.dropna().iloc[-1]) if not s_custom.dropna().empty else float("nan")
            st.metric("Latest correlation", f"{scalar:+.3f}")
            st.plotly_chart(
                _line(s_custom, title_c, sub_c, height=340),
                use_container_width=True,
            )

