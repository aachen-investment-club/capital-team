import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date

from lib.theme import inject_css, FAVICON
from lib.data import (
    get_security_master,
    get_eod_prices,
    get_portfolio_and_benchmarks,
    _eod_data_version,
)
from lib.volatility import (
    ALL_MODELS,
    log_returns_pct,
    ann_vol_from_var,
    run_backtest,
    evaluate_models,
    forecast_ahead,
)

st.set_page_config(page_title="Volatility · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Volatility")
with st.popover("ℹ"):
    st.markdown("""Fits GARCH family models to estimate how volatile each asset currently is, compares model forecasts against realised vol, and projects volatility forward over your chosen horizon.""")

_TODAY    = date.today() - pd.Timedelta(days=1)
_DATE_MIN = date(2015, 1, 1)

# Benchmark tickers stored in portfolio_and_benchmarks (fund-inception history only)
_BENCH_TICKERS = {"SPX", "MSCI_WORLD", "MSCI_EUROPE", "60_40"}

# ── Sidebar ───────────────────────────────────────────────────────────────────

# ── Controls ──────────────────────────────────────────────────────────────────

sm = get_security_master()
_equities    = sm[sm["asset_type"] != "INDEX"]["ticker"].tolist() if not sm.empty else []
_idx_tickers = sm[sm["asset_type"] == "INDEX"]["ticker"].tolist() if not sm.empty else []
asset_options = ["Portfolio"] + _equities + _idx_tickers

_c1, _c2, _c3, _c4, _c5 = st.columns([1.2, 1.8, 1.5, 1.5, 0.8])
selected_asset  = _c1.selectbox("Asset", asset_options)
selected_models = _c2.multiselect(
    "Models", options=list(ALL_MODELS), default=list(ALL_MODELS),
    help="Select one or more GARCH-family models to compare.",
)
backtest_from = _c3.date_input(
    "Backtest from", value=date(2025, 1, 1),
    min_value=_DATE_MIN, max_value=_TODAY,
)
forecast_horizon = _c4.slider("Forecast horizon (days)", min_value=5, max_value=90, value=30, step=5)
run_btn          = _c5.button("Run", type="primary", use_container_width=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _get_returns(asset: str, cache_version: str) -> pd.Series | None:
    """
    Load log returns (%) for any asset.

    Priority for benchmarks / index tickers:
      1. EOD parquet (if ingested via ingest_eod.py — full history)
      2. portfolio_and_benchmarks.json fallback (fund inception only)
    """
    if asset == "Portfolio":
        df = get_portfolio_and_benchmarks()
        port = df[df["ticker"] == "PORTFOLIO"][["date", "daily_return"]].dropna()
        if port.empty:
            return None
        s = port.set_index("date")["daily_return"] * 100
        s.index = pd.to_datetime(s.index)
        return s.sort_index().dropna()

    master = get_security_master()
    row = master[master["ticker"] == asset]

    if not row.empty:
        sid = row.iloc[0]["security_id"]
        eod = get_eod_prices(sid, cache_version)
        if not eod.empty:
            price_col = "adj_close" if "adj_close" in eod.columns else "close"
            eod["date"] = pd.to_datetime(eod["date"])
            prices = eod.set_index("date")[price_col].sort_index()
            rets = log_returns_pct(prices)
            if len(rets) >= 10:
                return rets

    # Fallback: portfolio_and_benchmarks for benchmark tickers (fund-inception only)
    if asset in _BENCH_TICKERS:
        df = get_portfolio_and_benchmarks()
        sub = df[df["ticker"] == asset][["date", "daily_return"]].dropna()
        if not sub.empty:
            s = sub.set_index("date")["daily_return"] * 100
            s.index = pd.to_datetime(s.index)
            return s.sort_index().dropna()

    return None

if not selected_models:
    st.info("Select at least one model in the sidebar.")
    st.stop()

version  = _eod_data_version()
rets_raw = _get_returns(selected_asset, version)

if rets_raw is None or len(rets_raw) < 20:
    st.warning(
        f"Not enough price history for **{selected_asset}** "
        f"(need ≥ 20 trading days). Try a different asset or ingest more data."
    )
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────

if "vol_result" not in st.session_state:
    st.session_state["vol_result"] = None

if run_btn:
    with st.spinner("Fitting models…"):
        try:
            bt_data  = run_backtest(rets_raw, backtest_from, selected_models)
            eval_df  = evaluate_models(bt_data["variances"], bt_data["proxy_rv"])
            fc_data  = forecast_ahead(rets_raw, selected_models, steps=forecast_horizon)
            st.session_state["vol_result"] = {
                "bt":       bt_data,
                "eval":     eval_df,
                "fc":       fc_data,
                "asset":    selected_asset,
                "bt_from":  backtest_from,
                "models":   list(selected_models),
            }
        except Exception as e:
            st.error(str(e))
            st.session_state["vol_result"] = None

res = st.session_state["vol_result"]

if res is None:
    st.info("Configure settings in the sidebar and click **Run** to fit the models.")
    st.stop()


# ── Helper: consistent colour per model ──────────────────────────────────────

_MODEL_COLOR = {"GARCH": "#2563EB", "GJR-GARCH": "#10B981"}

def _model_color(name: str) -> str:
    return _MODEL_COLOR.get(name, "#94A3B8")


# ── 1. BACKTEST ───────────────────────────────────────────────────────────────

st.subheader(f"Backtest — {res['asset']}  ·  from {res['bt_from']}")

bt = res["bt"]
test_idx  = bt["test_returns"].index
proxy_ann = ann_vol_from_var(bt["proxy_rv"])  # |daily return| × √252 proxy

fig_bt = go.Figure()

# Proxy "realised" vol — grey area
fig_bt.add_trace(go.Scatter(
    x=test_idx, y=proxy_ann,
    fill="tozeroy",
    fillcolor="rgba(148,163,184,0.18)",
    line=dict(color="rgba(148,163,184,0.5)", width=1),
    name="|Daily Return| × √252 (proxy)",
    hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>Proxy RV</extra>",
))

for name, h_series in bt["variances"].items():
    ann_v = ann_vol_from_var(h_series.reindex(test_idx).ffill())
    fig_bt.add_trace(go.Scatter(
        x=test_idx, y=ann_v,
        mode="lines",
        name=name,
        line=dict(color=_model_color(name), width=2),
        hovertemplate=f"%{{x|%d %b %Y}}  %{{y:.1f}}<extra>{name}</extra>",
    ))

fig_bt.update_layout(
    template="capital",
    yaxis_title="Annualised Volatility (%)",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(l=60, r=20, t=40, b=60),
)
fig_bt.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
st.plotly_chart(fig_bt, width="stretch")

st.caption(
    "Proxy RV is the absolute daily log-return scaled to annualised vol. "
    "Model lines show conditional vol propagated through the test period "
    "using parameters estimated on training data only."
)


# ── 2. MODEL RANKINGS ─────────────────────────────────────────────────────────

st.divider()
st.subheader("Model Rankings")

eval_df = res["eval"].copy()

# Format for display
disp_eval = eval_df.copy()
disp_eval["MSE"]   = disp_eval["MSE"].map("{:.6f}".format)
disp_eval["QLIKE"] = disp_eval["QLIKE"].map("{:.4f}".format)
if "DM stat" in disp_eval.columns:
    disp_eval["DM stat"]  = disp_eval["DM stat"].map(
        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
    )
    disp_eval["DM p-val"] = disp_eval["DM p-val"].map(
        lambda v: f"{v:.3f}" if pd.notna(v) else "—"
    )

st.dataframe(disp_eval, hide_index=True, width="stretch")

if "DM stat" in eval_df.columns:
    dm_stat = eval_df["DM stat"].iloc[0]
    dm_p    = eval_df["DM p-val"].iloc[0]
    best    = eval_df["Model"].iloc[0]
    second  = eval_df["Model"].iloc[1]
    if pd.notna(dm_stat):
        sig = dm_p < 0.05
        verdict = (
            f"The Diebold-Mariano test (QLIKE loss, H₀: equal accuracy) gives "
            f"stat = {dm_stat:.3f}, p = {dm_p:.3f}. "
            + (f"**{best}** is significantly better than {second} at the 5% level."
               if sig else
               f"No statistically significant difference between {best} and {second}.")
        )
        st.caption(verdict)

# QLIKE bar chart
bar_fig = go.Figure()
for _, row in eval_df.iterrows():
    bar_fig.add_trace(go.Bar(
        x=[row["Model"]],
        y=[row["QLIKE"]],
        name=row["Model"],
        marker_color=_model_color(row["Model"]),
        text=[f"{row['QLIKE']:.4f}"],
        textposition="outside",
    ))
bar_fig.update_layout(
    template="capital",
    yaxis_title="Mean QLIKE (lower = better)",
    showlegend=False,
    height=260,
    margin=dict(l=60, r=20, t=20, b=40),
)
st.plotly_chart(bar_fig, width="stretch")


# ── 3. VOLATILITY FORECAST ────────────────────────────────────────────────────

st.divider()
st.subheader(f"Volatility Forecast — next {res['fc']['forecast'][list(res['fc']['forecast'].keys())[0]].__len__()} days")

fc   = res["fc"]
fig_fc = go.Figure()

for name in res["models"]:
    hist = fc["history"].get(name)
    fore = fc["forecast"].get(name)
    if hist is None or fore is None:
        continue

    hist_ann = ann_vol_from_var(hist)
    fore_ann = ann_vol_from_var(fore)

    # Historical segment
    fig_fc.add_trace(go.Scatter(
        x=hist_ann.index, y=hist_ann,
        mode="lines",
        name=name,
        line=dict(color=_model_color(name), width=2),
        legendgroup=name,
        hovertemplate=f"%{{x|%d %b %Y}}  %{{y:.1f}}<extra>{name}</extra>",
    ))

    # Forecast segment — dashed, connected to last historical point
    join_x = [hist_ann.index[-1]] + list(fore_ann.index)
    join_y = [float(hist_ann.iloc[-1])] + list(fore_ann.values)
    fig_fc.add_trace(go.Scatter(
        x=join_x, y=join_y,
        mode="lines",
        name=f"{name} forecast",
        line=dict(color=_model_color(name), width=2, dash="dash"),
        legendgroup=name,
        showlegend=True,
        hovertemplate=f"%{{x|%d %b %Y}}  %{{y:.1f}}<extra>{name} forecast</extra>",
    ))

# Vertical line at "today"
last_hist = max(fc["history"][m].index[-1] for m in res["models"] if m in fc["history"])
fig_fc.add_vline(
    x=last_hist.timestamp() * 1000,
    line_dash="dot", line_color="#BFDBFE",
    annotation_text="Today", annotation_position="top right",
)

fig_fc.update_layout(
    template="capital",
    yaxis_title="Annualised Volatility (%)",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    margin=dict(l=60, r=20, t=40, b=60),
)
fig_fc.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
st.plotly_chart(fig_fc, width="stretch")

st.caption(
    "Solid lines: in-sample conditional vol (last 90 trading days, fit on full history). "
    "Dashed lines: multi-step-ahead forecast. "
    "GARCH-family forecasts converge toward long-run volatility as the horizon increases."
)
