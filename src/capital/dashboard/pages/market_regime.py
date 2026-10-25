"""Market Regime: compare and configure regime models (the champion, HMM
variants, an SMA baseline), standing on any past date and looking forward.

The as-of date is the point you stand on. Everything left of it is what each
model could have said then; everything right of it is the future relative to
that date: the price path that followed, and each model's labels as they would
have come in live (HMMs fitted point-in-time keep their as-of parameters, so
that stretch is genuinely out of sample).

Model runs are cached per (asset, model, settings, as-of) on the data version,
so the champion (~1s) and each HMM (~0.5-5s) are fitted once; flipping between
dates and models already run is instant.
"""
import traceback
from dataclasses import replace

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
from plotly.subplots import make_subplots

from capital.analytics.regime import champion as ch
from capital.analytics.regime import compare as cmp
from capital.analytics.regime.hmm import STATE_COLORS, STATES
from capital.dashboard import components as ui
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.theme import GRAPH_CONFIG, NAVY, TEXT_MUTED

dash.register_page(
    __name__, path="/market-regime", name="Market Regime", order=0.6,
    description="Compare and configure regime models (champion, HMM variants, SMA) "
                "on any date, and see what followed.",
)

ASSETS = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "TLT": "20Y Treasury",
    "HYG": "High-Yield Bond", "GLD": "Gold",
    "XLK": "Technology", "XLC": "Communication", "XLY": "Cons. Discretionary", "XLF": "Financials",
    "XLV": "Health Care", "XLI": "Industrials", "XLP": "Cons. Staples", "XLE": "Energy",
    "XLU": "Utilities", "XLB": "Materials", "XLRE": "Real Estate",
}
LOOKBACK = {"1Y": 252, "3Y": 756, "5Y": 1260, "All": 0}
LOOKAHEAD = {"None": 0, "3M": 63, "6M": 126, "1Y": 252, "All": -1}

#: The champion with the page's settings applied. Listed next to the champion so
#: a configuration is always judged against the reference it departs from.
CUSTOM = "champion_custom"
MODEL_LABELS = {k: m.label for k, m in cmp.MODELS.items()} | {CUSTOM: "Champion · custom"}
MODEL_OPTIONS = [{"value": k, "label": MODEL_LABELS[k]} for k in ["champion", CUSTOM, *list(cmp.MODELS)[1:]]]

#: Ribbon colour codes: direction labels, then the champion's volatility labels.
_CODES = {"Bear": 0, "Sideways": 1, "Bull": 2, "Calm": 3, "Choppy": 4, "Turbulent": 5}
_CODE_COLORS = [STATE_COLORS["Bear"], STATE_COLORS["Sideways"], STATE_COLORS["Bull"],
                ch.VOL_COLORS["Calm"], ch.VOL_COLORS["Choppy"], ch.VOL_COLORS["Turbulent"]]
_RIBBON_SCALE = [pair for i, c in enumerate(_CODE_COLORS)
                 for pair in ([i / len(_CODE_COLORS), c], [(i + 1) / len(_CODE_COLORS), c])]
_FUTURE = "rgba(148,163,184,0.12)"
_BADGE = {"Bull": "teal", "Sideways": "blue", "Bear": "red",
          "Calm": "teal", "Choppy": "yellow", "Turbulent": "grape"}


def _pct(x, digits: int = 1, sign: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


# ── Cached runs ───────────────────────────────────────────────────────────────

@cached_by_version
def _prices(ticker: str) -> pd.Series:
    return loaders.get_market_prices(ticker).dropna().astype(float).sort_index()


@cached_by_version
def _run(ticker: str, key: str, settings: tuple, asof: str | None) -> dict:
    """One model on one asset. `settings` is a sorted tuple of (name, value);
    `asof` is None for models whose output does not depend on it."""
    px, cfg = _prices(ticker), dict(settings)
    if key in ("champion", CUSTOM):
        return {"frame": cmp.champion_regime(px, ch.ChampionConfig(**cfg))}
    m = cmp.MODELS[key]
    if m.kind == "sma":
        return {"frame": cmp.sma_regime(px, **cfg)}
    return cmp.fit_hmm(px, **m.params, **cfg, asof=pd.Timestamp(asof) if asof else None)


def _model_settings(key: str, champion_cfg: dict, hmm_cfg: dict, sma_cfg: dict) -> tuple[tuple, bool]:
    """(settings tuple, depends on as-of) for one model."""
    if key == "champion":
        return tuple(sorted(ch.CHAMPION.to_dict().items())), False
    if key == CUSTOM:
        return tuple(sorted(champion_cfg.items())), False
    kind = cmp.MODELS[key].kind
    if kind == "sma":
        return tuple(sorted(sma_cfg.items())), False
    return tuple(sorted(hmm_cfg.items())), hmm_cfg["fit"] == "asof"


# ── Layout ────────────────────────────────────────────────────────────────────

def _num(id_, label, value, step, w=120, **kw):
    return dmc.NumberInput(id=id_, label=label, value=value, step=step, w=w, size="xs", **kw)


def _settings_panel() -> dmc.Accordion:
    c = ch.CHAMPION
    pct = {"suffix": "%", "decimalScale": 1}
    champion = dmc.Stack([
        dmc.Text("Applied to “Champion · custom” (selected automatically when you change a value); "
                 "“Champion” always runs the defaults.", size="xs", c=TEXT_MUTED),
        dmc.Group([
            dmc.Select(id="mr-c-q", label="Kalman q (slope speed)", size="xs", w=170,
                       data=[{"value": str(q), "label": f"{q:.0e}" + (" (champion)" if q == c.kalman_q else "")}
                             for q in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4)], value=str(c.kalman_q)),
            _num("mr-c-bull", "Bull line /yr", c.bull_slope * 100, 1, **pct),
            _num("mr-c-bear", "Bear line /yr", c.bear_slope * 100, 1, **pct),
            _num("mr-c-band", "Hysteresis band", c.band * 100, 0.5, min=0, **pct),
            _num("mr-c-burn", "Burn-in (days)", c.burn_in, 21, min=21),
        ], gap="sm", align="end"),
        dmc.Group([
            _num("mr-c-calm", "Calm below", c.calm_below * 100, 1, min=1, **pct),
            _num("mr-c-turb", "Turbulent above", c.turbulent_above * 100, 1, min=1, **pct),
            _num("mr-c-past", "Realised days", c.past_days, 1, min=2, max=63),
            _num("mr-c-ahead", "GARCH days ahead", c.ahead_days, 1, min=1, max=63),
            _num("mr-c-train", "GARCH warm-up (yrs)", c.train_years, 1, min=1, max=10, w=140),
            dmc.Button("Reset to champion", id="mr-c-reset", variant="subtle", size="xs"),
        ], gap="sm", align="end"),
    ], gap="xs")
    hmm = dmc.Group([
        dmc.Stack([dmc.Text("Fit window", size="xs", fw=500),
                   dmc.SegmentedControl(id="mr-h-fit", size="xs", value="asof", data=[
                       {"value": "asof", "label": "Up to as-of (point-in-time)"},
                       {"value": "full", "label": "Full sample (look-ahead)"}])], gap=2),
        dmc.Stack([dmc.Text("Probabilities", size="xs", fw=500),
                   dmc.SegmentedControl(id="mr-h-probs", size="xs", value="filtered", data=[
                       {"value": "filtered", "label": "Filtered"},
                       {"value": "smoothed", "label": "Smoothed (hindsight)"}])], gap=2),
        dmc.Select(id="mr-h-cov", label="Covariance", size="xs", w=110, value="full",
                   data=["full", "diag"]),
        _num("mr-h-restarts", "EM restarts", 3, 1, min=1, max=8),
        _num("mr-h-rv", "Realised-vol window", 21, 1, min=5, max=126, w=140),
    ], gap="sm", align="end")
    sma = dmc.Group([
        _num("mr-s-window", "SMA window (days)", 200, 10, min=20, max=400, w=140),
        _num("mr-s-band", "Neutral band", 2.0, 0.5, min=0, suffix="%", decimalScale=1),
    ], gap="sm", align="end")
    item = lambda v, title, body: dmc.AccordionItem(value=v, children=[  # noqa: E731
        dmc.AccordionControl(dmc.Text(title, size="sm", fw=600, c=NAVY)), dmc.AccordionPanel(body)])
    return dmc.Accordion(multiple=True, variant="contained", radius="md", children=[
        item("champion", "Champion settings", champion),
        item("hmm", "HMM settings (all HMM variants)", hmm),
        item("sma", "SMA baseline settings", sma),
    ])


def layout():
    px = _prices("SPY")
    last = px.index.max() if len(px) else pd.Timestamp.today()
    first = px.index.min() + pd.Timedelta(days=400) if len(px) else last
    return dmc.Stack([
        ui.page_title(
            "Market Regime",
            "Compare regime models on any asset: the champion (Kalman trend × GJR-GARCH), HMM "
            "variants and an SMA baseline. Pick an as-of date to see what each model said then, "
            "and how the market (and the models) went on from there."),
        dmc.Paper(dmc.Stack([
            dmc.Group([
                dmc.Select(id="mr-asset", label="Asset", w=210, value="SPY", searchable=True,
                           data=[{"value": k, "label": f"{k} · {v}"} for k, v in ASSETS.items()]),
                dmc.DatePickerInput(id="mr-asof", label="As-of date", w=150,
                                    value=last.date().isoformat(), minDate=first.date().isoformat(),
                                    maxDate=last.date().isoformat()),
                dmc.Stack([dmc.Text("Look back", size="sm", fw=500),
                           dmc.SegmentedControl(id="mr-back", data=list(LOOKBACK), value="3Y", size="xs")], gap=3),
                dmc.Stack([dmc.Text("Look ahead", size="sm", fw=500),
                           dmc.SegmentedControl(id="mr-ahead", data=list(LOOKAHEAD), value="1Y", size="xs")], gap=3),
                dmc.Select(id="mr-horizon", label="Scoring horizon", w=140, value="21",
                           data=[{"value": "5", "label": "1 week"}, {"value": "21", "label": "1 month"},
                                 {"value": "63", "label": "3 months"}]),
            ], gap="md", align="end"),
            dmc.Group([
                dmc.MultiSelect(id="mr-models", label="Models", data=MODEL_OPTIONS,
                                value=list(cmp.DEFAULT_MODELS), w=560, clearable=False),
                dmc.Select(id="mr-shade", label="Shade price by", w=220, value="champion",
                           data=[o for o in MODEL_OPTIONS if o["value"] in cmp.DEFAULT_MODELS]),
                dmc.Button("Run", id="mr-run", mt=22),
            ], gap="md", align="end"),
        ], gap="sm"), withBorder=True, radius="md", p="sm"),
        _settings_panel(),
        dcc.Loading(html.Div(id="mr-results"), delay_show=300),
        ui.explain("How the models and scores work", _METHOD_TEXT),
    ], gap="sm")


# ── Small callbacks: keep the controls consistent ─────────────────────────────

_CHAMPION_INPUTS = ["mr-c-q", "mr-c-bull", "mr-c-bear", "mr-c-band", "mr-c-burn",
                    "mr-c-calm", "mr-c-turb", "mr-c-past", "mr-c-ahead", "mr-c-train"]


def _champion_cfg(q, bull, bear, band, burn, calm, turb, past, ahead, train) -> dict:
    """Champion settings from the inputs (percent fields -> fractions), falling
    back to the default for any field left empty."""
    c = ch.CHAMPION
    f = lambda v, d, scale=1.0: d if v in (None, "") else float(v) / scale  # noqa: E731
    i = lambda v, d: d if v in (None, "") else int(v)  # noqa: E731
    return replace(c, kalman_q=f(q, c.kalman_q), bull_slope=f(bull, c.bull_slope, 100),
                   bear_slope=f(bear, c.bear_slope, 100), band=f(band, c.band, 100),
                   burn_in=i(burn, c.burn_in), calm_below=f(calm, c.calm_below, 100),
                   turbulent_above=f(turb, c.turbulent_above, 100), past_days=i(past, c.past_days),
                   ahead_days=i(ahead, c.ahead_days), train_years=i(train, c.train_years)).to_dict()


@callback([Output(i, "value") for i in _CHAMPION_INPUTS], Input("mr-c-reset", "n_clicks"),
          prevent_initial_call=True)
def _reset_champion(_):
    c = ch.CHAMPION
    return [str(c.kalman_q), c.bull_slope * 100, c.bear_slope * 100, c.band * 100, c.burn_in,
            c.calm_below * 100, c.turbulent_above * 100, c.past_days, c.ahead_days, c.train_years]


@callback(Output("mr-models", "value"), [Input(i, "value") for i in _CHAMPION_INPUTS],
          State("mr-models", "value"), prevent_initial_call=True)
def _select_custom(*args):
    """Changing a champion setting adds the custom champion to the comparison."""
    *vals, models = args
    if _champion_cfg(*vals) == ch.CHAMPION.to_dict() or CUSTOM in (models or []):
        return no_update
    return [*(models or []), CUSTOM]


@callback(Output("mr-shade", "data"), Output("mr-shade", "value"), Input("mr-models", "value"),
          State("mr-shade", "value"))
def _shade_options(models, current):
    models = models or []
    data = [o for o in MODEL_OPTIONS if o["value"] in models]
    return data, current if current in models else (models[0] if models else None)


@callback(Output("mr-asof", "maxDate"), Output("mr-asof", "minDate"), Output("mr-asof", "value"),
          Input("mr-asset", "value"), State("mr-asof", "value"), prevent_initial_call=True)
def _asof_bounds(asset, asof):
    px = _prices(asset)
    if px.empty:
        return no_update, no_update, no_update
    lo, hi = px.index.min() + pd.Timedelta(days=400), px.index.max()
    a = min(max(pd.Timestamp(asof or hi), lo), hi)
    return hi.date().isoformat(), lo.date().isoformat(), a.date().isoformat()


# ── Results ───────────────────────────────────────────────────────────────────

@callback(
    Output("mr-results", "children"),
    Input("mr-run", "n_clicks"),
    State("mr-asset", "value"), State("mr-asof", "value"), State("mr-back", "value"),
    State("mr-ahead", "value"), State("mr-horizon", "value"), State("mr-models", "value"),
    State("mr-shade", "value"),
    [State(i, "value") for i in _CHAMPION_INPUTS],
    State("mr-h-fit", "value"), State("mr-h-probs", "value"), State("mr-h-cov", "value"),
    State("mr-h-restarts", "value"), State("mr-h-rv", "value"),
    State("mr-s-window", "value"), State("mr-s-band", "value"),
)
def _results(_, asset, asof, back, ahead, horizon, models, shade, *rest):
    try:
        champion_vals, (fit, probs, cov, restarts, rv, sma_w, sma_b) = rest[:10], rest[10:]
        if not models:
            return ui.alert("Pick at least one model.", "yellow")
        px = _prices(asset)
        if px.empty:
            return ui.alert(f"No prices stored for {asset}.", "red")
        asof = px.index[max(px.index.searchsorted(pd.Timestamp(asof or px.index.max()), side="right") - 1, 0)]
        horizon = int(horizon or 21)
        champion_cfg = _champion_cfg(*champion_vals)
        hmm_cfg = {"fit": fit, "probs": probs, "covariance": cov, "restarts": int(restarts or 3),
                   "rv_window": int(rv or 21)}
        sma_cfg = {"window": int(sma_w or 200), "band": float(sma_b if sma_b is not None else 2) / 100}

        runs, failed = {}, []
        for key in models:
            settings, dated = _model_settings(key, champion_cfg, hmm_cfg, sma_cfg)
            try:
                runs[key] = _run(asset, key, settings, asof.date().isoformat() if dated else None)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{MODEL_LABELS[key]}: {exc}")
        if not runs:
            return ui.alert("Every model failed. " + " · ".join(failed), "red")

        # The displayed window: look back from the as-of date, then look ahead.
        pos = px.index.get_loc(asof)
        nb, na = LOOKBACK[back], LOOKAHEAD[ahead]
        lo = px.index[max(pos - nb, 0)] if nb else px.index[0]
        hi = px.index[-1] if na < 0 else px.index[min(pos + na, len(px) - 1)]
        window = px.loc[lo:hi].index
        states = {MODEL_LABELS[k]: r["frame"]["state"] for k, r in runs.items()}

        children = [ui.alert(" · ".join(failed), "orange", "Some models failed")] if failed else []
        children += [
            ui.section(f"As of {asof:%d %b %Y}"),
            _asof_cards(runs, asof, champion_cfg),
            _next_section(px, asof),
            ui.section("Regimes across the as-of date"),
            _graph(_price_ribbons(px, runs, window, asof, shade if shade in runs else next(iter(runs)), asset)),
            dmc.Text("Shaded grey: the future relative to the as-of date. Point-in-time HMMs keep their "
                     "as-of parameters there, so their labels are out of sample.", size="xs", c=TEXT_MUTED),
            ui.section(f"Scorecard: what followed each label ({horizon}-day forward window)"),
            *_scorecards(runs, px, asof, horizon, states),
        ]
        champs = {k: r for k, r in runs.items() if k in ("champion", CUSTOM)}
        if champs:
            children += [ui.section("Champion readings"),
                         _graph(_champion_fig(champs, window, asof, champion_cfg))]
        hmms = {k: r for k, r in runs.items() if "params" in r}
        if hmms:
            children += [ui.section("HMM states"), _hmm_tables(hmms)]
        return dmc.Stack(children, gap="xs")
    except Exception:  # noqa: BLE001
        return dmc.Alert(dmc.Code(traceback.format_exc(), block=True), color="red", variant="light",
                         title="Market regime failed")


def _graph(fig: go.Figure) -> dcc.Graph:
    return dcc.Graph(figure=fig, config=GRAPH_CONFIG)


def _asof_cards(runs: dict, asof: pd.Timestamp, champion_cfg: dict) -> dmc.SimpleGrid:
    cards = []
    for key, r in runs.items():
        f = r["frame"].loc[:asof].dropna(subset=["state"])
        if f.empty:
            body = [dmc.Text("Not warmed up by this date.", size="xs", c=TEXT_MUTED)]
        else:
            row, state = f.iloc[-1], f["state"].iloc[-1]
            days = cmp.days_in_state(f["state"], asof)
            badges = [dmc.Badge(state, color=_BADGE[state], variant="filled")]
            if key in ("champion", CUSTOM):
                cfg = ch.ChampionConfig(**(champion_cfg if key == CUSTOM else ch.CHAMPION.to_dict()))
                vs = f["vol_state"].iloc[-1]
                badges.append(dmc.Badge(vs, color=_BADGE[vs], variant="light"))
                lines = [f"Slope {row['reading']:+.0%}/yr (Bull > {cfg.bull_slope:+.0%}, Bear < {cfg.bear_slope:+.0%})",
                         f"Vol nowcast {row['vol_nowcast']:.0%} · {cmp.days_in_state(f['vol_state'], asof)}d {vs}"]
            elif cmp.MODELS[key].kind == "sma":
                lines = [f"Price {row['reading']:+.1%} vs its moving average"]
            else:
                p = " / ".join(f"{s[0]} {row.get(f'p_{s}', np.nan):.0%}" for s in STATES if f"p_{s}" in row)
                lines = [f"P({state}) {row['confidence']:.0%} · {p}",
                         f"Fitted on data to {r['train_end']:%d %b %Y}"]
            body = [dmc.Group(badges, gap=4),
                    dmc.Text(f"{days}d in regime", size="xs", fw=600, c=NAVY),
                    *[dmc.Text(x, size="xs", c=TEXT_MUTED) for x in lines]]
        cards.append(dmc.Paper(dmc.Stack([dmc.Text(MODEL_LABELS[key], fw=700, size="sm", c=NAVY), *body], gap=4),
                               withBorder=True, radius="md", p="sm"))
    return dmc.SimpleGrid(cards, cols={"base": 1, "sm": 2, "lg": 4}, spacing="sm")


def _next_section(px: pd.Series, asof: pd.Timestamp):
    nxt = cmp.what_happened_next(px, asof)
    if nxt.empty:
        return dmc.Text("The as-of date is the latest close: there is no future to look at yet. "
                        "Pick an earlier date to see what followed.", size="sm", c=TEXT_MUTED)
    names = {21: "1 month", 63: "3 months", 126: "6 months", 252: "1 year"}
    return dmc.Stack([
        dmc.Text("What happened next", fw=600, size="sm", c=NAVY),
        ui.metric_row([(f"Next {names.get(int(r.horizon_d), f'{int(r.horizon_d)}d')}", _pct(r["return"]),
                        f"max DD {_pct(r.max_drawdown, 1)} · vol {r.ann_vol:.0%} · to {r.end:%d %b %Y}")
                       for _, r in nxt.iterrows()], cols={"base": 2, "md": 4}),
    ], gap=4)


def _ribbon_rows(runs: dict) -> list[tuple[str, pd.Series]]:
    rows = []
    for key, r in runs.items():
        rows.append((MODEL_LABELS[key], r["frame"]["state"]))
        if "vol_state" in r["frame"]:
            rows.append((f"{MODEL_LABELS[key]} · vol", r["frame"]["vol_state"]))
    return rows


def _price_ribbons(px, runs, window, asof, shade_key, asset) -> go.Figure:
    rows = _ribbon_rows(runs)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.62, 0.38], specs=[[{"secondary_y": True}], [{}]])
    p = px.reindex(window)
    shade = runs[shade_key]["frame"]["state"].reindex(window)
    for s in STATES:
        on = (shade == s).astype(float).replace(0.0, np.nan)
        fig.add_trace(go.Bar(x=window, y=on, name=s, marker_color=STATE_COLORS[s], opacity=0.25,
                             hoverinfo="skip", width=86_400_000 * 3), row=1, col=1, secondary_y=False)
    past, fut = p.loc[:asof], p.loc[asof:]
    hover = "%{x|%d %b %Y}  " + asset + " %{y:.2f}<extra></extra>"
    fig.add_trace(go.Scatter(x=past.index, y=past, name=asset, line=dict(color=NAVY, width=1.6),
                             hovertemplate=hover, showlegend=False), row=1, col=1, secondary_y=True)
    if len(fut) > 1:
        fig.add_trace(go.Scatter(x=fut.index, y=fut, name="after as-of", line=dict(color=NAVY, width=1.4, dash="dot"),
                                 hovertemplate=hover, showlegend=False), row=1, col=1, secondary_y=True)

    labels = [lab for lab, _ in rows][::-1]  # first model on top
    series = [s.reindex(window) for _, s in rows][::-1]
    z = [[_CODES.get(v, np.nan) if isinstance(v, str) else np.nan for v in s] for s in series]
    text = [[v if isinstance(v, str) else "–" for v in s] for s in series]
    fig.add_trace(go.Heatmap(x=window, y=labels, z=z, text=text, zmin=-0.5, zmax=len(_CODES) - 0.5,
                             colorscale=_RIBBON_SCALE, showscale=False, ygap=3,
                             hovertemplate="%{x|%d %b %Y}  %{y}: %{text}<extra></extra>"), row=2, col=1)

    if window[-1] > asof:
        for r in (1, 2):
            fig.add_vrect(x0=asof, x1=window[-1], fillcolor=_FUTURE, line_width=0, layer="above", row=r, col=1)
    fig.add_vline(x=asof, line_color=NAVY, line_width=1.2, line_dash="dash")
    fig.add_annotation(x=asof, y=1.0, yref="paper", text=f"as of {asof:%d %b %Y}", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=10, color=NAVY))

    lo, hi = float(p.min()), float(p.max())
    pad = (hi - lo) * 0.05
    fig.update_yaxes(range=[0, 1], visible=False, row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[lo - pad, hi + pad], side="left", row=1, col=1, secondary_y=True)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11), row=2, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(height=380 + 26 * len(rows), barmode="overlay", bargap=0,
                      margin=dict(l=ui.axis_margin(labels, font_size=11), r=16, t=36, b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1,
                                  title_text=f"Price shaded by {MODEL_LABELS[shade_key]}: "))
    return fig


def _scorecards(runs, px, asof, horizon, states) -> list:
    cards = {MODEL_LABELS[k]: cmp.scorecard(r["frame"]["state"], px, asof, horizon) for k, r in runs.items()}
    rows, bars = [], []
    for name, sc in cards.items():
        b, a = sc["before"], sc["after"]
        by = b["by_state"] if b else None
        rows.append([
            name,
            f"{b['switches_per_year']:.1f}" if b else "–",
            *[_pct(by.loc[s, "ann_ret"], 0) if by is not None else "–" for s in STATES],
            _pct(b["spread"], 0) if b else "–",
            _pct(b["vol_spread"], 0) if b else "–",
            _pct(a["spread"], 0) if a else "–",
            _pct(a["vol_spread"], 0) if a else "–",
        ])
        if by is not None:
            bars.append((name, by))
    head = ["Model", "Switches / yr", *[f"{s}: fwd ret" for s in STATES],
            "Bull − Bear ret", "Bear − Bull vol", "After as-of: ret spread", "After as-of: vol spread"]
    table = ui.df_table(pd.DataFrame(rows, columns=head))

    out = [table, dmc.Text(
        f"Before = days up to the as-of date whose {horizon}-day forward window had closed by then; after = "
        "days after the as-of date. Forward returns are annualised. A useful direction model has a "
        "positive Bull − Bear return spread; a useful volatility read a positive Bear − Bull vol spread.",
        size="xs", c=TEXT_MUTED)]
    if bars:
        ret_fig, vol_fig = go.Figure(), go.Figure()
        for s in STATES:
            names = [n for n, _ in bars]
            ret_fig.add_trace(go.Bar(x=names, y=[by.loc[s, "ann_ret"] for _, by in bars], name=s,
                                     marker_color=STATE_COLORS[s],
                                     hovertemplate=f"%{{x}} · {s}: %{{y:+.1%}} / yr<extra></extra>"))
            vol_fig.add_trace(go.Bar(x=names, y=[by.loc[s, "fwd_vol"] for _, by in bars], name=s,
                                     marker_color=STATE_COLORS[s], showlegend=False,
                                     hovertemplate=f"%{{x}} · {s}: %{{y:.1%}}<extra></extra>"))
        ret_fig.update_layout(title=f"Forward return by label (to {asof:%b %Y}, annualised)", barmode="group",
                              height=320, yaxis_tickformat="+.0%", margin=dict(t=50, b=40))
        vol_fig.update_layout(title="Forward realised vol by label", barmode="group",
                              height=320, yaxis_tickformat=".0%", margin=dict(t=50, b=40))

        agree = cmp.agreement(states, end=asof)
        agree_fig = go.Figure(go.Heatmap(
            z=agree.to_numpy(), x=list(agree.columns), y=list(agree.index), zmin=0, zmax=1,
            colorscale=[[0, "#FFFFFF"], [1, NAVY]], xgap=2, ygap=2,
            text=[[f"{v:.0%}" if np.isfinite(v) else "" for v in row] for row in agree.to_numpy()],
            texttemplate="%{text}", hovertemplate="%{y} vs %{x}: same label %{z:.0%} of days<extra></extra>",
            colorbar=dict(tickformat=".0%", thickness=8)))
        agree_fig.update_yaxes(autorange="reversed", showgrid=False)
        agree_fig.update_xaxes(showgrid=False)
        agree_fig.update_layout(title=f"Agreement: share of days with the same label (to {asof:%b %Y})",
                                height=320, margin=dict(l=ui.axis_margin(agree.index), t=50, b=60))
        out.append(dmc.SimpleGrid([_graph(ret_fig), _graph(vol_fig), _graph(agree_fig)],
                                  cols={"base": 1, "lg": 3}, spacing="sm"))
    return out


def _champion_fig(champs: dict, window, asof, champion_cfg: dict) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Trend slope (annualised)", "Volatility nowcast (annualised)"))
    dash_by = {"champion": "solid", CUSTOM: "dot"}
    for key, r in champs.items():
        f = r["frame"].reindex(window)
        name = MODEL_LABELS[key]
        fig.add_trace(go.Scatter(x=window, y=f["reading"], name=name, line=dict(color=NAVY, dash=dash_by[key]),
                                 hovertemplate=f"%{{x|%d %b %Y}}  {name}: %{{y:+.1%}}/yr<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=window, y=f["vol_nowcast"], name=name, showlegend=False,
                                 line=dict(color=NAVY, dash=dash_by[key]),
                                 hovertemplate=f"%{{x|%d %b %Y}}  {name}: %{{y:.1%}}<extra></extra>"),
                      row=2, col=1)
    for key in champs:
        cfg = ch.ChampionConfig(**(champion_cfg if key == CUSTOM else ch.CHAMPION.to_dict()))
        d = dash_by[key]
        for y, color in ((cfg.bull_slope, STATE_COLORS["Bull"]), (cfg.bear_slope, STATE_COLORS["Bear"])):
            fig.add_hline(y=y, line_color=color, line_dash="dash" if d == "solid" else "dot", line_width=1,
                          row=1, col=1)
        for y, color in ((cfg.calm_below, ch.VOL_COLORS["Calm"]), (cfg.turbulent_above, ch.VOL_COLORS["Turbulent"])):
            fig.add_hline(y=y, line_color=color, line_dash="dash" if d == "solid" else "dot", line_width=1,
                          row=2, col=1)
    fig.add_vline(x=asof, line_color=NAVY, line_width=1.2, line_dash="dash")
    if window[-1] > asof:
        for r in (1, 2):
            fig.add_vrect(x0=asof, x1=window[-1], fillcolor=_FUTURE, line_width=0, layer="below", row=r, col=1)
    fig.update_yaxes(tickformat="+.0%", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=2, col=1)
    fig.update_layout(height=460, margin=dict(t=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1))
    return fig


def _hmm_tables(hmms: dict) -> dmc.SimpleGrid:
    cards = []
    for key, r in hmms.items():
        p = r["params"]
        df = pd.DataFrame({
            "State": p.index,
            "Ann. return": [f"{x:+.0f}%" for x in p["ann_ret_pct"]],
            "Ann. vol": [f"{x:.0f}%" for x in p["ann_vol_pct"]],
            "Typical length": [f"{x:.0f}d" for x in p["exp_duration_d"]],
            "Share": [f"{x:.0%}" for x in p["share"]],
        })
        cards.append(dmc.Paper(dmc.Stack([
            dmc.Group([dmc.Text(MODEL_LABELS[key], fw=700, size="sm", c=NAVY),
                       dmc.Text(f"fitted to {r['train_end']:%d %b %Y} · log-lik {r['loglik']:,.0f}",
                                size="xs", c=TEXT_MUTED)], justify="space-between"),
            dmc.Text(cmp.MODELS[key].description, size="xs", c=TEXT_MUTED),
            ui.df_table(df),
        ], gap=4), withBorder=True, radius="md", p="sm"))
    return dmc.SimpleGrid(cards, cols={"base": 1, "lg": 2}, spacing="sm")


_METHOD_TEXT = r"""
**The as-of date.** Stand on any past close. Left of the dashed line is what each
model could have told you then; right of it (shaded grey) is the future relative to
that date: the price that followed and each model's labels as they would have come
in live. *What happened next* gives the realised return, worst drawdown and vol over
the following 1, 3, 6 and 12 months.

**Models.**

- **Champion**: a local-linear-trend Kalman filter on log price gives the drift;
  annualised above the Bull line = Bull, below the Bear line = Bear, with a hysteresis
  band so labels do not flicker. Volatility: realised variance of the last few days
  blended with a GJR-GARCH(1,1)-t forecast of the next few (refitted each January on
  earlier data), cut into Calm / Choppy / Turbulent. Causal by construction.
  **Champion · custom** runs your settings next to it.
- **HMM variants**: Gaussian hidden Markov models on daily return and log 21-day realised
  vol (or returns only). States are ranked by mean return (or by vol for the
  *vol-labelled* variant) and mapped to Bull / Sideways / Bear; with four states the two
  middle ones are both Sideways and their probabilities are summed.
- **SMA 200 trend**: price against its moving average with a neutral band. The baseline
  any regime model should beat.

**Point in time vs hindsight (HMM settings).** *Up to as-of* fits the HMM on data up to
the as-of date and freezes its parameters, so everything after is out of sample.
*Full sample* fits on all of history (parameter look-ahead). *Filtered* probabilities use
data up to each day; *smoothed* ones also use later days (state look-ahead). Switch both
to hindsight and watch the bands sharpen: that sharpness is not available in real time.

**Scorecard.** For every day the forward $h$-day log return and realised vol are grouped
by that day's label:

$$\text{fwd ret}_{\text{state}} = \tfrac{252}{h}\,\overline{\ln P_{t+h} - \ln P_t}, \qquad
\text{spread} = \text{fwd ret}_{\text{Bull}} - \text{fwd ret}_{\text{Bear}}$$

*Before* uses days whose forward window had closed by the as-of date (what you could have
checked then); *after* uses the days that followed it. Equity bear regimes are often
followed by sharp rebounds, so a negative return spread is common; the vol spread
(Bear − Bull forward vol) is usually the more reliable sign that a model reads risk.
*Switches / yr* measures stability: many switches means a label that is expensive to
trade on. *Agreement* is the share of days on which two models give the same label.
"""
