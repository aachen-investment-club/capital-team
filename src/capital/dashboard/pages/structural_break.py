"""Structural Break Detection, CUSUM and Student-t BOCPD on a composite
macro stress level signal, run against benchmarks / positions / NAV / custom."""
import traceback

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc
from plotly.subplots import make_subplots

from capital.analytics.credit_liquidity import CreditFilter, LiquidityFilter, StressScore
from capital.analytics.structural_break import CUSUMDetector, StudentTBOCPD, build_level_signal
from capital.analytics.trend import GJRGarch
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG

dash.register_page(
    __name__, path="/structural-break", name="Structural Break", order=9,
    description="CUSUM and BOCPD regime-shift detection on a macro stress signal.",
)

BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "TLT": "20Y Treasury",
    "GLD": "Gold",
    "HYG": "High-Yield Bond",
}

_C_UP, _C_DOWN, _C_LINE, _C_GREY = "#10B981", "#EF4444", "#3B82F6", "#64748B"

HAZARD_OPTIONS = {
    "1/50: short (~10 wk)": 1 / 50,
    "1/150: medium (~30 wk)": 1 / 150,
    "1/300: long  (~60 wk)": 1 / 300,
}


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ── Data loaders ──────────────────────────────────────────────────────────────

@cached_by_version
def _load_macro(start: str, end: str) -> dict:
    result = {}
    try:
        result["hy_oas"] = loaders.get_fred_series("BAMLH0A0HYM2").loc[start:end]
    except Exception:
        result["hy_oas"] = pd.Series(dtype=float)
    for key, ticker in [("spy_close", "SPY"), ("tlt_close", "TLT"), ("vix_close", "VIX")]:
        try:
            result[key] = loaders.get_market_prices(ticker).loc[start:end]
        except Exception:
            result[key] = pd.Series(dtype=float)
    try:
        result["spy_vol"] = loaders.get_market_ohlcv("SPY")["volume"].loc[start:end].dropna()
    except Exception:
        result["spy_vol"] = pd.Series(dtype=float)
    return result


@cached_by_version
def _load_benchmark_prices(start: str, end: str) -> pd.DataFrame:
    cols = {}
    for ticker in BENCHMARKS:
        s = loaders.get_market_prices(ticker).loc[start:end]
        if not s.empty:
            cols[ticker] = s
    if not cols:
        return pd.DataFrame()
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    return df.apply(pd.to_numeric, errors="coerce").dropna(how="all")


@cached_by_version
def _load_portfolio_positions(start: str, end: str) -> dict:
    sm = loaders.get_security_master()
    wh = loaders.get_daily_weightings_history()
    if not wh.empty:
        syms = set(wh[wh["date"] == wh["date"].max()]["symbol"].tolist())
        sm = sm[sm["ticker"].isin(syms)]
    sm = sm[sm["asset_type"] != "INDEX"]
    out = {}
    for _, row in sm.iterrows():
        eod = loaders.get_eod_prices(row["security_id"])
        if eod.empty:
            continue
        eod = eod.copy()
        eod["date"] = pd.to_datetime(eod["date"])
        col = "adj_close" if "adj_close" in eod.columns else "close"
        s = eod.set_index("date")[col].loc[start:end].dropna().sort_index()
        if len(s) >= 60:
            out[row["ticker"]] = s.rename(row["ticker"])
    return out


@cached_by_version
def _load_portfolio_aggregate(start: str, end: str) -> pd.Series:
    pb = loaders.get_portfolio_and_benchmarks()
    port = pb[pb["ticker"] == "PORTFOLIO"].set_index("date")["index_value"]
    port.index = pd.to_datetime(port.index)
    return port.loc[start:end].sort_index().rename("Portfolio NAV")


@cached_by_version
def _build_signal_bus(start: str, end: str) -> dict:
    """gjr_vol, credit/liquidity levels, stress index, and the composite level signal."""
    macro = _load_macro(start, end)
    hy_oas = macro.get("hy_oas", pd.Series(dtype=float))
    spy_close = macro.get("spy_close", pd.Series(dtype=float))
    spy_vol = macro.get("spy_vol", pd.Series(dtype=float))

    out = {"spy_close": spy_close}
    try:
        gjr = GJRGarch()
        gjr.fit(spy_close)
        out["gjr_vol"] = gjr.zscore_vol
        out["gjr_vol_ann"] = gjr.annualised_vol
    except Exception:
        out["gjr_vol"] = pd.Series(0.0, index=spy_close.index)
        out["gjr_vol_ann"] = pd.Series(dtype=float)
    try:
        cf = CreditFilter(hy_oas)
        out["c_level"], out["c_shock"] = cf.level(), cf.shock()
    except Exception:
        out["c_level"] = out["c_shock"] = pd.Series(dtype=float)
    try:
        lf = LiquidityFilter(spy_close, spy_vol)
        out["liq_level"] = lf.level()
        out["liq_shock"] = lf.shock(out["liq_level"])
    except Exception:
        out["liq_level"] = out["liq_shock"] = pd.Series(dtype=float)
    try:
        ss = StressScore({"credit_shock": out["c_shock"], "liquidity_shock": out["liq_shock"]})
        out["stress_idx"], _ = ss.run()
    except Exception:
        out["stress_idx"] = pd.Series(dtype=float)
    try:
        out["level_signal"] = build_level_signal(out["gjr_vol"], out["c_level"], out["liq_level"])
    except Exception:
        out["level_signal"] = pd.Series(dtype=float)
    return out


# ── Charts ────────────────────────────────────────────────────────────────────

def _cusum_chart(price, level, cusum_sig, cusum_stats, title, k, h, height=600):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.25, 0.30], vertical_spacing=0.04)

    price_clean = price.dropna()
    if not price_clean.empty:
        aligned_sig = cusum_sig.reindex(price_clean.index, method="ffill").fillna(0)
        prev_state, seg_start = 0, price_clean.index[0]

        def _backdrop(x0, x1, state):
            if state:
                fig.add_vrect(x0=x0, x1=x1,
                              fillcolor=_rgba(_C_UP if state == 1 else _C_DOWN, 0.12),
                              line_width=0, row=1, col=1)

        for dt, sig in aligned_sig.items():
            if sig != prev_state:
                _backdrop(seg_start, dt, prev_state)
                seg_start, prev_state = dt, int(sig)
        _backdrop(seg_start, price_clean.index[-1], prev_state)

        fig.add_trace(go.Scatter(
            x=price_clean.index, y=price_clean.values, mode="lines", name="Price",
            line=dict(color="#94A3B8", width=1.5), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=1, col=1)

        for bx, label, color, sym in [
            (cusum_sig[cusum_sig == 1], "Upward break", _C_UP, "triangle-up"),
            (cusum_sig[cusum_sig == -1], "Downward break", _C_DOWN, "triangle-down"),
        ]:
            if not bx.empty:
                prices_at = price_clean.reindex(bx.index, method="nearest")
                fig.add_trace(go.Scatter(
                    x=bx.index, y=prices_at.values, mode="markers", name=label,
                    marker=dict(symbol=sym, size=10, color=color,
                                line=dict(width=1, color="white")),
                    showlegend=False,
                    hovertemplate=f"%{{x|%Y-%m-%d}}, {label}<extra></extra>"), row=1, col=1)

    lev = level.dropna()
    if not lev.empty:
        fig.add_trace(go.Scatter(
            x=lev.index, y=np.where(lev.values >= 0, lev.values, 0),
            fill="tozeroy", mode="none", fillcolor=_rgba(_C_DOWN, 0.25),
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=lev.index, y=np.where(lev.values < 0, lev.values, 0),
            fill="tozeroy", mode="none", fillcolor=_rgba(_C_UP, 0.20),
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=lev.index, y=lev.values, mode="lines",
            line=dict(color=_C_LINE, width=1.2), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)", row=2, col=1)

    if not cusum_stats.empty:
        fig.add_trace(go.Scatter(
            x=cusum_stats.index, y=cusum_stats["S_pos"].values, mode="lines", name="S+",
            line=dict(color=_C_UP, width=1.5), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d} S+: %{y:.3f}<extra></extra>"), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=cusum_stats.index, y=cusum_stats["S_neg"].values, mode="lines", name="S−",
            line=dict(color=_C_DOWN, width=1.5), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d} S−: %{y:.3f}<extra></extra>"), row=3, col=1)
        fig.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)", row=3, col=1)

    title_text = (f"<b>{title}, CUSUM</b>"
                  f"<br><span style='font-size:11px;color:#94A3B8'>"
                  f"k={k:.2f} · h={h:.1f} · green backdrop = upward regime · "
                  f"red backdrop = downward regime</span>")
    fig.update_layout(
        title=dict(text=title_text, x=0, xanchor="left"),
        height=height, showlegend=False,
        margin=dict(l=50, r=20, t=72, b=40))
    fig.update_yaxes(zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Level signal", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="S+ / S−", row=3, col=1, title_font=dict(size=10))
    return fig


def _bocpd_chart(price, bocpd_norm, bocpd_raw, title, hazard, nu, height=580):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.30, 0.25], vertical_spacing=0.04)
    price_clean, norm_clean, raw_clean = price.dropna(), bocpd_norm.dropna(), bocpd_raw.dropna()

    if not price_clean.empty and not norm_clean.empty:
        threshold = norm_clean.quantile(0.80)
        aligned = norm_clean.reindex(price_clean.index, method="ffill").fillna(0)
        in_anom, seg_start = False, price_clean.index[0]

        def _anom(x0, x1, active):
            if active:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=_rgba(_C_DOWN, 0.14),
                              line_width=0, row=1, col=1)

        for dt, val in aligned.items():
            now = val >= threshold
            if now != in_anom:
                _anom(seg_start, dt, in_anom)
                seg_start, in_anom = dt, now
        _anom(seg_start, price_clean.index[-1], in_anom)

        fig.add_trace(go.Scatter(
            x=price_clean.index, y=price_clean.values, mode="lines",
            line=dict(color="#94A3B8", width=1.5), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=1, col=1)

    if not norm_clean.empty:
        fig.add_trace(go.Scatter(
            x=norm_clean.index, y=norm_clean.values, fill="tozeroy", mode="lines",
            fillcolor=_rgba(_C_DOWN, 0.22), line=dict(color=_C_DOWN, width=1.2),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>"), row=2, col=1)
        fig.add_hline(y=norm_clean.quantile(0.80), line_width=1, line_dash="dot",
                      line_color="rgba(239,68,68,0.5)", row=2, col=1)

    if not raw_clean.empty:
        fig.add_trace(go.Scatter(
            x=raw_clean.index, y=raw_clean.values, fill="tozeroy", mode="lines",
            fillcolor=_rgba(_C_GREY, 0.25), line=dict(color=_C_GREY, width=1.0),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra></extra>"), row=3, col=1)

    hazard_disp = f"hazard=1/{round(1 / hazard)}" if hazard > 0 else "hazard=0"
    title_text = (f"<b>{title}, BOCPD</b>"
                  f"<br><span style='font-size:11px;color:#94A3B8'>"
                  f"{hazard_disp} · ν={nu} · red backdrop = top-20% anomaly · "
                  f"dotted line = 80th-percentile threshold</span>")
    fig.update_layout(
        title=dict(text=title_text, x=0, xanchor="left"),
        height=height, showlegend=False,
        margin=dict(l=50, r=20, t=72, b=40))
    fig.update_yaxes(zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(title_text="Price", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Surprise (0: 1)", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="P(changepoint)", row=3, col=1, title_font=dict(size=10))
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

def layout():
    today = pd.Timestamp.today().date()
    return dmc.Stack([
        ui.page_title(
            "Structural Break Detection",
            "Detects when the market has shifted into a new regime (e.g. from calm "
            "to stressed) using two methods: CUSUM, which flags when a stress signal "
            "persistently drifts above or below its baseline, and BOCPD, which "
            "estimates the probability that the current observation belongs to a "
            "new distribution."),
        dmc.Group([
            dmc.DatePickerInput(id="sb-start", label="Start date", value="2019-01-01", w=150),
            dmc.DatePickerInput(id="sb-end", label="End date", value=today.isoformat(), w=150),
            dmc.CheckboxGroup(
                id="sb-sources", label="Sources", value=["bench"],
                children=dmc.Group([
                    dmc.Checkbox(label="Benchmarks", value="bench"),
                    dmc.Checkbox(label="Positions", value="pos"),
                    dmc.Checkbox(label="Portfolio NAV", value="agg"),
                    dmc.Checkbox(label="Custom", value="custom"),
                ], mt=6)),
            dmc.TextInput(id="sb-custom-ric", label="Custom ticker / RIC",
                          placeholder="e.g. NVDA", w=180),
        ], align="end", gap="lg"),
        dmc.Group([
            dmc.Box(dmc.Stack([
                dmc.Text("CUSUM k (allowance)", size="sm", fw=500),
                dmc.Slider(id="sb-cusum-k", min=0.1, max=2.0, value=0.3, step=0.05,
                           marks=[{"value": v, "label": str(v)} for v in (0.5, 1.0, 1.5, 2.0)],
                           w=200),
            ], gap=4)),
            dmc.Box(dmc.Stack([
                dmc.Text("CUSUM h (threshold)", size="sm", fw=500),
                dmc.Slider(id="sb-cusum-h", min=1.0, max=15.0, value=6.0, step=0.5,
                           marks=[{"value": v, "label": str(v)} for v in (5, 10, 15)], w=200),
            ], gap=4)),
            dmc.Select(id="sb-hazard", label="BOCPD hazard rate",
                       data=list(HAZARD_OPTIONS.keys()),
                       value=list(HAZARD_OPTIONS.keys())[1], w=230),
            dmc.Box(dmc.Stack([
                dmc.Text("BOCPD ν (d.o.f.)", size="sm", fw=500),
                dmc.Slider(id="sb-nu", min=3, max=20, value=5,
                           marks=[{"value": v, "label": str(v)} for v in (5, 10, 15, 20)], w=180),
            ], gap=4)),
            dmc.Button("Run", id="sb-run", mt=22),
        ], align="end", gap="lg"),
        dcc.Loading(dmc.Box(
            dmc.Alert("Choose sources and click Run.", color="blue", variant="light"),
            id="sb-results", mt="md")),
    ])


# ── Results assembly ──────────────────────────────────────────────────────────

def _metric(label, value):
    return dmc.Paper([dmc.Text(label, className="metric-label"),
                      dmc.Text(value, className="metric-value")], className="metric-card")


def _summary_table(results: dict) -> dmc.Table:
    STATE_ICON = {1: "🟢", -1: "🔴", 0: "⚪"}
    STATE_LABEL = {1: "Upward break", -1: "Downward break", 0: "Stable"}
    rows = []
    for name, r in results.items():
        sig = r["cusum_sig"]
        last_breaks = sig[sig != 0]
        last_date = last_breaks.index[-1].strftime("%d %b %Y") if not last_breaks.empty else ": "
        state = int(last_breaks.iloc[-1]) if not last_breaks.empty else 0
        norm = r["bocpd_norm"].dropna()
        surprise = float(norm.iloc[-1]) if not norm.empty else 0.0
        surprise_pct = float((norm <= norm.iloc[-1]).mean() * 100) if not norm.empty else 0.0
        bar = "█" * round(surprise_pct / 10) + "░" * (10 - round(surprise_pct / 10))
        rows.append({
            "Asset": name,
            "Regime": f"{STATE_ICON[state]} {STATE_LABEL[state]}",
            "Last break": last_date,
            "CUSUM ↑ / ↓": f"{int((sig == 1).sum())} / {int((sig == -1).sum())}",
            "Surprise": f"{surprise:.2f}",
            "Anomaly level": f"{bar}  {surprise_pct:.0f}th pct",
        })
    return ui.df_table(pd.DataFrame(rows))


def _results_block(start, end, sources, custom_ric, k, h, hazard_label, nu) -> dmc.Stack:
    s_str, e_str = str(start), str(end)
    hazard = HAZARD_OPTIONS[hazard_label]

    bus = _build_signal_bus(s_str, e_str)
    if bus.get("spy_close", pd.Series(dtype=float)).empty:
        return dmc.Alert("SPY data unavailable: cannot build signal bus.",
                         color="red", variant="light")
    level_signal = bus.get("level_signal", pd.Series(dtype=float))
    if level_signal.empty:
        return dmc.Alert("Level signal is empty: check FRED and market-data ingest.",
                         color="red", variant="light")

    # Targets
    targets: dict[str, pd.Series] = {}
    warnings = []
    if "bench" in sources:
        bench_df = _load_benchmark_prices(s_str, e_str)
        for col in bench_df.columns:
            targets[f"{col} ({BENCHMARKS.get(col, col)})"] = bench_df[col].dropna()
    if "pos" in sources:
        targets.update(_load_portfolio_positions(s_str, e_str))
    if "agg" in sources:
        port_nav = _load_portfolio_aggregate(s_str, e_str)
        if not port_nav.empty:
            targets["Portfolio NAV"] = port_nav
    if "custom" in sources and (custom_ric or "").strip():
        ric = custom_ric.strip()
        s = loaders.get_market_prices(ric).loc[s_str:e_str]
        if s.empty:
            sm = loaders.get_security_master()
            match = sm[(sm["ric"] == ric) | (sm["ticker"] == ric)]
            if not match.empty:
                eod = loaders.get_eod_prices(match.iloc[0]["security_id"])
                if not eod.empty:
                    eod = eod.copy()
                    eod["date"] = pd.to_datetime(eod["date"])
                    col = "adj_close" if "adj_close" in eod.columns else "close"
                    s = eod.set_index("date")[col].loc[s_str:e_str].dropna().sort_index()
        if not s.empty:
            targets[ric] = s.rename(ric)
        else:
            warnings.append(f"No data found for '{ric}'.")

    if not targets:
        return dmc.Alert("No price series loaded. Enable at least one source.",
                         color="yellow", variant="light")

    # Detectors
    results = {}
    for name, price_series in targets.items():
        aligned_level = level_signal.reindex(
            level_signal.index.union(price_series.index)
        ).interpolate("time").reindex(level_signal.index)
        cusum = CUSUMDetector(k=k, h=h)
        bocpd = StudentTBOCPD(hazard=hazard, nu=nu)
        bocpd_norm = bocpd.run(aligned_level)
        results[name] = {
            "price": price_series,
            "level": aligned_level,
            "cusum_sig": cusum.detect(aligned_level),
            "cusum_stats": cusum.stat_series(aligned_level),
            "bocpd_norm": bocpd_norm,
            "bocpd_raw": bocpd.raw_probs,
        }

    # ── Assemble ──
    children = [dmc.Alert(w, color="yellow", variant="light") for w in warnings]
    children.append(dmc.Text(
        f"{s_str} → {e_str}  ·  CUSUM k={k:.2f} h={h:.1f}  ·  "
        f"BOCPD hazard=1/{round(1 / hazard)} ν={nu}", size="sm", c="dimmed"))

    # Macro bus panel
    stress = bus.get("stress_idx", pd.Series(dtype=float)).dropna()
    gjr_ann = bus.get("gjr_vol_ann", pd.Series(dtype=float)).dropna()
    lev_s = level_signal.dropna()
    bus_children = [dmc.SimpleGrid([
        _metric("Stress index (latest)", f"{stress.iloc[-1]:.3f}" if not stress.empty else ": "),
        _metric("GJR-GARCH vol (ann.)", f"{gjr_ann.iloc[-1]:.1%}" if not gjr_ann.empty else ": "),
        _metric("Level signal (latest)", f"{lev_s.iloc[-1]:+.3f}σ" if not lev_s.empty else ": "),
    ], cols={"base": 1, "sm": 3})]
    if not lev_s.empty:
        fig_bus = go.Figure()
        fig_bus.add_trace(go.Scatter(
            x=lev_s.index, y=np.where(lev_s.values >= 0, lev_s.values, 0),
            fill="tozeroy", mode="none", fillcolor=_rgba(_C_DOWN, 0.20), showlegend=False))
        fig_bus.add_trace(go.Scatter(
            x=lev_s.index, y=np.where(lev_s.values < 0, lev_s.values, 0),
            fill="tozeroy", mode="none", fillcolor=_rgba(_C_UP, 0.15), showlegend=False))
        fig_bus.add_trace(go.Scatter(
            x=lev_s.index, y=lev_s.values, mode="lines",
            line=dict(color=_C_LINE, width=1.5), showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}σ<extra></extra>"))
        fig_bus.add_hline(y=0, line_width=1, line_color="rgba(255,255,255,0.2)")
        fig_bus.update_layout(
            title=dict(text="<b>Composite level signal</b>"
                            "<br><span style='font-size:11px;color:#94A3B8'>"
                            "Expanding z-score of 10d-smoothed (GJR-vol + credit level "
                            "+ liquidity level)</span>", x=0, xanchor="left"),
            height=220, showlegend=False,
            margin=dict(l=50, r=20, t=64, b=30),
            yaxis=dict(zeroline=False), xaxis=dict(showgrid=False))
        bus_children.append(dcc.Graph(figure=fig_bus, config=GRAPH_CONFIG))
    children.append(dmc.Accordion([dmc.AccordionItem(
        [dmc.AccordionControl("Macro signal bus"),
         dmc.AccordionPanel(dmc.Stack(bus_children, gap="md"))], value="bus")],
        value="bus"))

    children.append(ui.section("Regime summary"))
    children.append(_summary_table(results))
    children.append(dmc.Divider(mt="md"))

    for name, r in results.items():
        children.append(ui.section(name))
        children.append(dmc.Tabs([
            dmc.TabsList([dmc.TabsTab("CUSUM", value="cusum"),
                          dmc.TabsTab("BOCPD", value="bocpd")]),
            dmc.TabsPanel(dmc.Stack([
                dcc.Graph(figure=_cusum_chart(r["price"], r["level"], r["cusum_sig"],
                                              r["cusum_stats"], name, k, h, 580),
                          config=GRAPH_CONFIG),
                dmc.Text("Green backdrop = CUSUM detected upward stress shift (S+ > h, "
                         "then reset). Red backdrop = downward stress shift. Triangles "
                         "mark break events. Panel 3 shows the running S+ (green) and "
                         "S− (red) statistics.", size="xs", c="dimmed"),
            ]), value="cusum"),
            dmc.TabsPanel(dmc.Stack([
                dcc.Graph(figure=_bocpd_chart(r["price"], r["bocpd_norm"], r["bocpd_raw"],
                                              name, hazard, nu, 560),
                          config=GRAPH_CONFIG),
                dmc.Text("Red backdrop = normalised changepoint surprise above 80th "
                         "percentile. Dotted line = 80th-percentile threshold. Panel 2: "
                         "normalised surprise (0: 1) smoothed over 20 days. Panel 3: raw "
                         "P(changepoint) from Student-t run-length distribution.",
                         size="xs", c="dimmed"),
            ]), value="bocpd"),
        ], value="cusum"))

    return dmc.Stack(children, gap="md")


@callback(
    Output("sb-results", "children"),
    Input("sb-run", "n_clicks"),
    State("sb-start", "value"),
    State("sb-end", "value"),
    State("sb-sources", "value"),
    State("sb-custom-ric", "value"),
    State("sb-cusum-k", "value"),
    State("sb-cusum-h", "value"),
    State("sb-hazard", "value"),
    State("sb-nu", "value"),
    prevent_initial_call=True,
)
def run_structural_break(n, start, end, sources, custom_ric, k, h, hazard_label, nu):
    try:
        return _results_block(start, end, sources or [], custom_ric,
                              float(k), float(h), hazard_label, int(nu))
    except Exception:
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True),
                         color="red", variant="light", title="Run failed")
