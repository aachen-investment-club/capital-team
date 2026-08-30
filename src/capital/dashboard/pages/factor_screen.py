"""
Factor Screen - the professional-grade cross-sectional factor model.

Shape of this page:
- A **run bar**: pick a finished run, or configure and queue a new one. Runs are
  background jobs (capital.jobs), so the page never blocks - the job queue below
  the bar is live.
- Seven tabs, each fed by its own callback so switching tabs does not recompute
  the others: Portfolio, What-if, Screen, Factors, Securities, Methodology,
  Run details.

Every chart carries a collapsed explanation (ui.explain); the Methodology tab is
the long-form reference behind them. Copy for both comes from
capital.dashboard.factor_text.

Charts: brand `capital` template. Diverging quantities (exposures, factor
returns, risk contributions) use a steel-blue/antique-gold pair with a neutral
midpoint - three times the colour-vision separation of the red/green scale, and
a neutral rather than yellow midpoint so zero reads as zero.
"""

import json
import re
from datetime import date
from functools import lru_cache

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, dash_table, dcc, html, no_update

from capital.analytics.factors import risk as riskmod
from capital.analytics.factors.spec import ALL_STYLES, DESCRIPTORS, STYLES, ModelSpec
from capital.dashboard import components as ui
from capital.dashboard import factor_text as txt
from capital.data import factor_store as fstore
from capital.data import loaders
from capital.data.cache import cached_by_version
from capital.jobs import queue as jobs
from capital.theme import BORDER, GRAPH_CONFIG, NAVY, TEXT_MUTED

dash.register_page(
    __name__,
    path="/factor-screen",
    name="Factor Screen",
    order=6,
    description="Cross-sectional factor model over the full universe: portfolio "
    "exposures, what-if trades, factor robustness, and the "
    "methodology behind it.",
)

# ── Palette ───────────────────────────────────────────────────────────────────
# Diverging pair (validated): CVD separation dE 24.6 vs 8.1 for red/green.
NEG, POS, MID = "#3A6B9C", "#B8962E", "#F1F5F9"
DIVERGING = [[0.0, "#1E3A5F"], [0.25, NEG], [0.5, MID], [0.75, POS], [1.0, "#7A6014"]]
#: Fixed categorical order, never cycled, capped at 5 series per chart.
SERIES = ["#0C1E40", "#B8962E", "#2E7D6B", "#8B2E4A", "#3A6B9C"]
MAX_SERIES = 5

_FREQ_LABELS = {"B": "Daily", "W-FRI": "Weekly", "ME": "Monthly"}
#: Poll cadence: brisk while something is running, near-idle otherwise.
_POLL_FAST, _POLL_IDLE = 1500, 20000
_GROUP_ORDER = ["Market", "Style", "Industry", "Country"]


# ── Cached inputs ─────────────────────────────────────────────────────────────


@cached_by_version
def _portfolio_weights() -> pd.DataFrame:
    """Latest portfolio weights mapped onto security_ids.

    Position symbols are tickers; the model is keyed on security_id, so anything
    the master does not know (or that never made the run) is reported as
    uncovered rather than dropped silently.
    """
    empty = pd.DataFrame(
        columns=["security_id", "symbol", "name", "weight", "category", "is_cash"]
    )
    try:
        hist = loaders.get_daily_weightings_history()
    except Exception as exc:  # noqa: BLE001
        # Position weights come from S3 (the IBKR Lambda's output). The model
        # itself does not need them, so an S3 hiccup should cost the Portfolio
        # and What-if tabs, not the whole page.
        print(f"[factor-screen] portfolio weights unavailable: {exc}")
        return empty
    master = loaders.get_security_master()
    if hist.empty or master.empty:
        return empty
    latest = hist[hist["date"] == hist["date"].max()].copy()
    ticker_to_sid = dict(zip(master["ticker"], master["security_id"]))
    latest["security_id"] = latest["symbol"].map(ticker_to_sid)
    latest["is_cash"] = latest["symbol"].str.startswith("CASH_")
    total = latest["pct_nav"].sum()
    latest["weight"] = latest["pct_nav"] / total if total else latest["pct_nav"]
    return latest[["security_id", "symbol", "name", "weight", "category", "is_cash"]]


@lru_cache(maxsize=8)
def _bundle(run_id: str) -> dict:
    """All frames one run needs, loaded once. Runs are immutable, so this is safe
    to memoise on run_id alone (unlike loader caches, which key on data_version)."""
    manifest = fstore.load_manifest(run_id) or {}
    exposures = fstore.load_exposure_matrix(run_id)
    meta = fstore.load_security_meta(run_id)
    return {
        "manifest": manifest,
        "spec": manifest.get("spec", {}),
        "summary": manifest.get("summary", {}),
        "coverage": manifest.get("coverage", {}),
        "groups": manifest.get("factor_groups", {}),
        "styles": manifest.get("styles", []),
        "exposures": exposures,
        "meta": meta.set_index("security_id") if not meta.empty else meta,
        "cov": fstore.load_covariance(run_id),
        "corr": fstore.load_correlation(run_id),
        "spec_risk": fstore.load_specific_risk(run_id),
        "factor_returns": fstore.load_factor_returns(run_id),
        "fit": fstore.load_frame(run_id, "fit_stats", index="date"),
    }


def _label(factor: str, styles: list[str]) -> str:
    if factor in STYLES:
        return STYLES[factor].label
    return factor.replace("IND_", "").replace("CTY_", "")


def _weights_for_run(run_id: str) -> tuple[pd.Series, dict]:
    """Portfolio weights restricted to what this run covers, plus a coverage note."""
    port = _portfolio_weights()
    bundle = _bundle(run_id)
    if port.empty or bundle["exposures"].empty:
        return pd.Series(dtype=float), {
            "covered": 0.0,
            "cash": 0.0,
            "missing": [],
            "n": 0,
            "total": 0,
            "names": {},
        }
    cash = float(port.loc[port["is_cash"], "weight"].sum())
    live = port[~port["is_cash"]].dropna(subset=["security_id"])
    covered = live[live["security_id"].isin(bundle["exposures"].index)]
    missing = sorted(set(port[~port["is_cash"]]["symbol"]) - set(covered["symbol"]))
    w = covered.set_index("security_id")["weight"]
    return w, {
        "covered": float(w.sum()),
        "cash": cash,
        "missing": missing,
        "n": int(len(w)),
        "total": int(len(port[~port["is_cash"]])),
        "names": dict(zip(covered["security_id"], covered["symbol"])),
    }

    # ── Figure helpers ────────────────────────────────────────────────────────────


def _fig(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 420,
    left: int = 60,
    right: int = 90,
    bottom: int = 52,
) -> go.Figure:
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:12px;color:{TEXT_MUTED}'>{subtitle}</span>"
    fig.update_layout(
        template="capital",
        height=height,
        title=dict(text=text, x=0, xanchor="left", pad=dict(b=10)),
        margin=dict(l=left, r=right, t=80 if subtitle else 62, b=bottom),
    )
    # The brand template ticks at 15px; ui.axis_margin budgets for
    # ui.AXIS_FONT_SIZE. Pin both axes to that size, or long category labels get
    # clipped by exactly the difference.
    fig.update_yaxes(tickfont=dict(size=ui.AXIS_FONT_SIZE))
    fig.update_xaxes(tickfont=dict(size=ui.AXIS_FONT_SIZE))
    return fig


def _legend_below(fig: go.Figure, has_xtitle: bool = False) -> go.Figure:
    """Legends go under the plot.

    Above it they collide with the two-line title (every title on this page
    carries a subtitle); directly below they collide with the x-axis title, so
    charts that have one get a deeper drop and a taller bottom margin.
    """
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.30 if has_xtitle else -0.16,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        )
    )
    fig.update_layout(margin=dict(b=fig.layout.margin.b + (86 if has_xtitle else 44)))
    return fig


def _pad_x(fig: go.Figure, values) -> go.Figure:
    """Room for the outside value labels, which Plotly's autorange ignores."""
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return fig
    lo, hi = min(min(finite), 0.0), max(max(finite), 0.0)
    pad = 0.22 * max(hi - lo, 1e-9)
    fig.update_xaxes(range=[lo - pad, hi + pad])
    return fig


def _slug(text: str) -> str:
    """Stable element id from a chart title."""
    plain = re.sub(r"<[^>]+>", " ", str(text or "chart"))
    return "fs-fig-" + re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")[:60]


def _graph(fig: go.Figure, id: str | None = None) -> dcc.Graph:
    """A figure with a *stable* id derived from its title.

    Without one, every re-render mounts a brand-new Graph: Plotly rebuilds the
    canvas, the element briefly has zero height, and the page scroll jumps. With
    a stable id React updates the existing figure in place.
    """
    return dcc.Graph(
        figure=fig,
        config=GRAPH_CONFIG,
        responsive=True,
        id=id or _slug(fig.layout.title.text),
    )


def _hbar(
    series: pd.Series,
    title: str,
    subtitle: str = "",
    xtitle: str = "Exposure (z-score)",
    height: int | None = None,
    fmt: str = "{:+.2f}",
) -> go.Figure:
    """Diverging horizontal bars with direct labels: the workhorse of this page."""
    s = series.dropna().sort_values()
    if s.empty:
        return _fig(go.Figure(), title, "No data")
    fig = go.Figure(
        go.Bar(
            y=s.index.tolist(),
            x=s.values,
            orientation="h",
            marker=dict(
                color=[POS if v >= 0 else NEG for v in s.values],
                line=dict(width=2, color="#FFFFFF"),
            ),
            text=[fmt.format(v) for v in s.values],
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=BORDER)
    fig = _fig(
        fig,
        title,
        subtitle,
        height or max(300, 30 * len(s) + 150),
        left=ui.axis_margin(s.index),
        right=70,
    )
    percent = fmt.endswith("%}")
    fig.update_layout(
        xaxis=dict(title=xtitle, zeroline=False, tickformat=".1%" if percent else None),
        bargap=0.3,
        uniformtext=dict(mode="hide", minsize=9),
    )
    fig.update_traces(cliponaxis=False)
    return _pad_x(fig, s.values)


def _grouped_hbar(
    frame: pd.DataFrame, title: str, subtitle: str = "", xtitle: str = "Exposure (z-score)"
) -> go.Figure:
    """Two-series comparison (e.g. current vs proposed). Legend always present."""
    fig = go.Figure()
    for i, col in enumerate(frame.columns[:MAX_SERIES]):
        fig.add_trace(
            go.Bar(
                y=frame.index.tolist(),
                x=frame[col].values,
                orientation="h",
                name=str(col),
                marker=dict(color=SERIES[i], line=dict(width=2, color="#FFFFFF")),
                hovertemplate=f"{col} · %{{y}}: %{{x:.3f}}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_width=1, line_color=BORDER)
    fig = _fig(
        fig,
        title,
        subtitle,
        max(340, 42 * len(frame) + 170),
        left=ui.axis_margin(frame.index),
        right=60,
    )
    fig.update_layout(
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        xaxis=dict(title=xtitle, zeroline=False),
    )
    return _pad_x(_legend_below(fig, has_xtitle=True), frame.to_numpy(dtype=float).ravel())


def _lines(
    frame: pd.DataFrame,
    title: str,
    subtitle: str = "",
    ytitle: str = "",
    height: int = 420,
    percent: bool = True,
) -> go.Figure:
    """Up to five series with a legend and direct end-labels."""
    cols = list(frame.columns)[:MAX_SERIES]
    # Direct end-labels only while they cannot collide; beyond three series the
    # legend carries identity and stacked end-labels become unreadable.
    label_ends = len(cols) <= 3
    fig = go.Figure()
    for i, col in enumerate(cols):
        s = frame[col].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                name=str(col),
                mode="lines",
                line=dict(color=SERIES[i % len(SERIES)], width=2),
                hovertemplate=f"{col}<br>%{{x|%d %b %Y}}: %{{y:.2%}}<extra></extra>"
                if percent
                else f"{col}<br>%{{x|%d %b %Y}}: %{{y:.3f}}<extra></extra>",
            )
        )
        if label_ends:
            fig.add_annotation(
                x=s.index[-1],
                y=s.iloc[-1],
                text=f"  {col}",
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color=TEXT_MUTED),
            )
    fig = _fig(fig, title, subtitle, height, left=68, right=110 if label_ends else 40)
    fig.update_layout(
        hovermode="x unified",
        yaxis=dict(title=ytitle, tickformat=".0%" if percent else None),
    )
    return _legend_below(fig)


def _heatmap(
    frame: pd.DataFrame,
    title: str,
    subtitle: str = "",
    fmt: str = "{:+.2f}",
    height: int | None = None,
    zmax: float | None = None,
    colorbar_title: str = "",
) -> go.Figure:
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    lim = zmax or (float(np.abs(finite).max()) if finite.size else 1.0) or 1.0
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=[str(c) for c in frame.columns],
            y=[str(i) for i in frame.index],
            colorscale=DIVERGING,
            zmid=0,
            zmin=-lim,
            zmax=lim,
            xgap=2,
            ygap=2,
            text=[
                [fmt.format(v) if np.isfinite(v) else " - " for v in row] for row in values
            ],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
            colorbar=dict(title=colorbar_title, thickness=12, len=0.75, outlinewidth=0),
        )
    )
    # Column labels sit above the plot, so the top margin has to clear both the
    # two-line title and the rotated tick text.
    # Rotated -45 degrees, so the vertical space a label needs is ~0.7 of its length.
    top_pad = min(150, 26 + 4.6 * max((len(str(c)) for c in frame.columns), default=0))
    fig = _fig(
        fig,
        title,
        subtitle,
        height or max(340, 30 * len(frame) + 200),
        left=ui.axis_margin(frame.index),
        right=80,
    )
    fig.update_layout(
        margin=dict(t=int(fig.layout.margin.t + top_pad)),
        xaxis=dict(side="top", tickfont=dict(size=11), showgrid=False, tickangle=-45),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12), showgrid=False),
    )
    return fig


def _table(
    df: pd.DataFrame, id: str | None = None, page_size: int = 15, height: int = 460
) -> html.Div:
    """Sortable, filterable table. Used where the reader needs to explore rather
    than read a fixed ranking: and as the accessible view of every chart."""
    return html.Div(
        dash_table.DataTable(
            id=id or f"{abs(hash(tuple(df.columns)))}",
            data=df.to_dict("records"),
            columns=[
                {
                    "name": c.replace("_", " ").title(),
                    "id": c,
                    "type": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "text",
                    "format": {"specifier": ".3f"}
                    if pd.api.types.is_float_dtype(df[c])
                    else None,
                }
                for c in df.columns
            ],
            sort_action="native",
            filter_action="native",
            page_size=page_size,
            style_table={"overflowX": "auto", "maxHeight": height, "overflowY": "auto"},
            style_cell={
                "fontFamily": "Inter, system-ui, sans-serif",
                "fontSize": "13px",
                "padding": "6px 10px",
                "border": f"1px solid {BORDER}",
            },
            style_header={
                "backgroundColor": "#EFF6FF",
                "fontWeight": 600,
                "color": NAVY,
                "border": f"1px solid {BORDER}",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#F8FAFC"}
            ],
        ),
        className="factor-table-wrap",
    )

    # ── Layout ────────────────────────────────────────────────────────────────────


def _style_options() -> list[dict]:
    return [{"value": k, "label": STYLES[k].label} for k in ALL_STYLES]


def _config_form() -> dmc.Stack:
    today = date.today()
    return dmc.Stack(
        [
            dmc.Alert(
                "A run reads the whole store and estimates the model on every "
                "date in the window. It runs in the background: you can close "
                "this and keep working.",
                color="blue",
                variant="light",
            ),
            dmc.TextInput(id="fs-name", label="Run name", value="Factor model"),
            dmc.Group(
                [
                    dmc.DatePickerInput(
                        id="fs-start",
                        label="Window start",
                        value=(today - pd.Timedelta(days=365 * 3)).isoformat(),
                        w=170,
                    ),
                    dmc.DatePickerInput(
                        id="fs-asof",
                        label="As of",
                        value=None,
                        placeholder="Latest in store",
                        w=170,
                        description="Blank = latest",
                    ),
                    dmc.Select(
                        id="fs-freq",
                        label="Frequency",
                        w=150,
                        data=[{"value": k, "label": v} for k, v in _FREQ_LABELS.items()],
                        value="W-FRI",
                    ),
                ],
                grow=True,
            ),
            dmc.MultiSelect(
                id="fs-styles",
                label="Style factors",
                data=_style_options(),
                value=list(ALL_STYLES),
                searchable=True,
                description="Every style by default. Any whose descriptors "
                "are missing from the store is dropped and "
                "reported in Run details, never faked.",
            ),
            dmc.Group(
                [
                    dmc.MultiSelect(
                        id="fs-assets",
                        label="Estimation universe",
                        data=["COMMON", "ETF"],
                        value=["COMMON"],
                        w=210,
                        description="Single stocks only",
                    ),
                    dmc.NumberInput(
                        id="fs-maxsec",
                        label="Max securities",
                        value=0,
                        min=0,
                        max=20000,
                        step=250,
                        w=160,
                        description="0 = the whole master",
                    ),
                    dmc.NumberInput(
                        id="fs-minhist",
                        label="Min history (days)",
                        value=250,
                        min=40,
                        max=2000,
                        step=10,
                        w=170,
                    ),
                ],
                grow=True,
            ),
            dmc.Group(
                [
                    dmc.Switch(
                        id="fs-etfs", label="Price ETFs off the model", checked=True
                    ),
                    dmc.Switch(id="fs-industry", label="Industry factors", checked=True),
                    dmc.Switch(id="fs-country", label="Country factors", checked=True),
                    dmc.Switch(id="fs-robust", label="Robust regression", checked=True),
                ],
                gap="lg",
            ),
            dmc.Accordion(
                chevronPosition="left",
                variant="subtle",
                children=[
                    dmc.AccordionItem(
                        value="adv",
                        children=[
                            dmc.AccordionControl(
                                dmc.Text(
                                    "Advanced estimation settings", size="sm", c=TEXT_MUTED
                                )
                            ),
                            dmc.AccordionPanel(
                                dmc.Stack(
                                    [
                                        dmc.Group(
                                            [
                                                dmc.Select(
                                                    id="fs-weight",
                                                    label="Regression weights",
                                                    w=190,
                                                    data=[
                                                        {
                                                            "value": "sqrt_cap",
                                                            "label": "Square-root cap",
                                                        },
                                                        {"value": "cap", "label": "Cap"},
                                                        {
                                                            "value": "equal",
                                                            "label": "Equal",
                                                        },
                                                    ],
                                                    value="sqrt_cap",
                                                ),
                                                dmc.NumberInput(
                                                    id="fs-winsor",
                                                    label="Winsorise at (σ)",
                                                    value=3.0,
                                                    min=1.5,
                                                    max=6.0,
                                                    step=0.5,
                                                    w=160,
                                                ),
                                                dmc.NumberInput(
                                                    id="fs-cover",
                                                    label="Min coverage",
                                                    value=0.5,
                                                    min=0.1,
                                                    max=0.95,
                                                    step=0.05,
                                                    w=150,
                                                ),
                                                dmc.Select(
                                                    id="fs-numeraire",
                                                    label="Numeraire",
                                                    w=170,
                                                    data=[
                                                        {
                                                            "value": "local",
                                                            "label": "Local currency",
                                                        },
                                                        {"value": "EUR", "label": "EUR"},
                                                        {"value": "USD", "label": "USD"},
                                                    ],
                                                    value="local",
                                                    description="Falls back to local if FX is missing",
                                                ),
                                            ],
                                            grow=True,
                                        ),
                                        dmc.Group(
                                            [
                                                dmc.NumberInput(
                                                    id="fs-hl-beta",
                                                    label="Beta half-life",
                                                    value=63,
                                                    min=10,
                                                    max=500,
                                                    w=150,
                                                ),
                                                dmc.NumberInput(
                                                    id="fs-hl-var",
                                                    label="Cov variance half-life",
                                                    value=90,
                                                    min=10,
                                                    max=500,
                                                    w=180,
                                                ),
                                                dmc.NumberInput(
                                                    id="fs-hl-corr",
                                                    label="Cov correlation half-life",
                                                    value=252,
                                                    min=20,
                                                    max=1000,
                                                    w=190,
                                                ),
                                                dmc.NumberInput(
                                                    id="fs-nw",
                                                    label="Newey-West lags",
                                                    value=5,
                                                    min=0,
                                                    max=20,
                                                    w=150,
                                                ),
                                            ],
                                            grow=True,
                                        ),
                                        dcc.Markdown(
                                            "Half-lives are in **periods of the chosen frequency's "
                                            "underlying daily series** for descriptors, and in "
                                            "**estimation periods** for the covariance matrix. "
                                            "See the methodology page for what each one controls.",
                                            className="explain-body",
                                        ),
                                    ],
                                    gap="sm",
                                )
                            ),
                        ],
                    ),
                ],
            ),
            dmc.Group(
                [
                    dmc.Button("Queue run", id="fs-submit", leftSection="▶"),
                    dmc.Button(
                        "Cancel", id="fs-close-drawer", variant="subtle", color="gray"
                    ),
                ],
                justify="flex-end",
                mt="md",
            ),
        ],
        gap="sm",
    )


def _tab_panel(
    value: str, explain_title: str, explain_body: str, extra=None
) -> dmc.TabsPanel:
    children = [ui.explain(explain_title, explain_body)]
    if extra:
        children.extend(extra)
        # overlay_style keeps the previous content on screen (dimmed) while the new
        # one is computed. The default behaviour swaps it for a spinner, which
        # collapses the page height and makes the browser jump to the top.
    children.append(
        dcc.Loading(
            html.Div(id=f"fs-{value}-content", style={"minHeight": "320px"}),
            delay_show=350,
            overlay_style={"visibility": "visible", "opacity": 0.45},
        )
    )
    return dmc.TabsPanel(dmc.Stack(children, gap="sm", mt="md"), value=value)


def layout():
    return dmc.Stack(
        [
            ui.page_title("Factor Screen", txt.PAGE_INFO),
            dmc.Paper(
                [
                    dmc.Group(
                        [
                            dmc.Select(
                                id="fs-run",
                                label="Model run",
                                data=[],
                                value=None,
                                searchable=True,
                                w=420,
                                placeholder="No runs yet: queue one to start",
                            ),
                            html.Div(id="fs-run-badges", style={"paddingTop": "26px"}),
                            dmc.Button(
                                "New run", id="fs-open-drawer", mt=24, leftSection="＋"
                            ),
                        ],
                        align="start",
                        gap="md",
                    ),
                    html.Div(id="fs-jobs", style={"marginTop": "12px"}),
                ],
                withBorder=True,
                radius="md",
                p="md",
            ),
            dmc.Drawer(
                id="fs-drawer",
                title="Configure a factor-model run",
                opened=False,
                position="right",
                size="xl",
                padding="lg",
                children=_config_form(),
            ),
            dcc.Interval(id="fs-poll", interval=_POLL_FAST),
            dcc.Store(id="fs-trades", data=[]),
            dcc.Store(id="fs-queue-sig", data=None),
            html.Div(id="fs-notify"),
            dmc.Tabs(
                id="fs-tabs",
                value="portfolio",
                mt="md",
                children=[
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Portfolio", value="portfolio"),
                            dmc.TabsTab("What-if", value="whatif"),
                            dmc.TabsTab("Screen", value="screen"),
                            dmc.TabsTab("Factors", value="factors"),
                            dmc.TabsTab("Securities", value="securities"),
                            dmc.TabsTab("Methodology", value="method"),
                            dmc.TabsTab("Run details", value="run"),
                        ]
                    ),
                    _tab_panel(
                        "portfolio",
                        "How to read the portfolio exposures",
                        txt.READING_EXPOSURES + "\n\n---\n\n" + txt.RISK_DECOMPOSITION,
                    ),
                    _tab_panel(
                        "whatif",
                        "How the trade simulation works",
                        txt.WHAT_IF,
                        extra=[
                            dmc.Paper(
                                dmc.Stack(
                                    [
                                        dmc.Group(
                                            [
                                                dmc.Select(
                                                    id="fs-trade-sec",
                                                    label="Security",
                                                    data=[],
                                                    searchable=True,
                                                    w=340,
                                                    placeholder="Search by ticker or name",
                                                    nothingFoundMessage="No matching security",
                                                    maxDropdownHeight=340,
                                                    limit=100,
                                                    leftSection=ui.SEARCH_ICON,
                                                ),
                                                dmc.NumberInput(
                                                    id="fs-trade-w",
                                                    label="Weight change (%)",
                                                    value=2.0,
                                                    step=0.5,
                                                    decimalScale=2,
                                                    w=170,
                                                ),
                                                dmc.Button(
                                                    "Add trade", id="fs-trade-add", mt=24
                                                ),
                                                dmc.Button(
                                                    "Clear all",
                                                    id="fs-trade-clear",
                                                    mt=24,
                                                    variant="subtle",
                                                    color="gray",
                                                ),
                                                dmc.SegmentedControl(
                                                    id="fs-funding",
                                                    mt=24,
                                                    value="pro_rata",
                                                    data=[
                                                        {
                                                            "value": "pro_rata",
                                                            "label": "Funded pro rata",
                                                        },
                                                        {
                                                            "value": "cash",
                                                            "label": "Funded from cash",
                                                        },
                                                    ],
                                                ),
                                            ],
                                            align="start",
                                            gap="md",
                                        ),
                                        html.Div(id="fs-trade-list"),
                                    ],
                                    gap="sm",
                                ),
                                withBorder=True,
                                radius="md",
                                p="md",
                            ),
                        ],
                    ),
                    _tab_panel(
                        "screen",
                        "What is in this table",
                        txt.SCREEN_EXPLAIN,
                        extra=[
                            dmc.Group(
                                [
                                    dmc.Select(
                                        id="fs-screen-x",
                                        label="X axis",
                                        data=[],
                                        w=200,
                                        searchable=True,
                                    ),
                                    dmc.Select(
                                        id="fs-screen-y",
                                        label="Y axis",
                                        data=[],
                                        w=200,
                                        searchable=True,
                                    ),
                                    dmc.MultiSelect(
                                        id="fs-screen-sector",
                                        label="Sectors",
                                        data=[],
                                        w=320,
                                        searchable=True,
                                        placeholder="All sectors",
                                    ),
                                ],
                                align="end",
                                gap="md",
                            ),
                        ],
                    ),
                    _tab_panel(
                        "factors",
                        "How to judge whether a factor is real",
                        txt.IC_EXPLAIN,
                        extra=[
                            dmc.MultiSelect(
                                id="fs-factor-pick",
                                label="Factors to chart",
                                data=[],
                                value=[],
                                searchable=True,
                                maw=640,
                                description="Up to five at a time: beyond that a line "
                                "chart stops being readable.",
                            ),
                        ],
                    ),
                    _tab_panel(
                        "securities",
                        "Reading a single security",
                        txt.SECURITY_EXPLAIN,
                        extra=[
                            dmc.Select(
                                id="fs-sec-pick",
                                label="Security",
                                data=[],
                                searchable=True,
                                w=380,
                                placeholder="Search by ticker or name",
                                nothingFoundMessage="No matching security",
                                maxDropdownHeight=340,
                                limit=100,
                                leftSection=ui.SEARCH_ICON,
                            ),
                        ],
                    ),
                    _tab_panel("method", "How to use this reference", txt.METHOD_HOWTO),
                    _tab_panel(
                        "run", "What the coverage report tells you", txt.COVERAGE_EXPLAIN
                    ),
                ],
            ),
        ],
        gap="xs",
    )

    # ── Job queue callbacks ───────────────────────────────────────────────────────


def _job_row(job: dict) -> dmc.Paper:
    colour = {
        "queued": "gray",
        "running": "blue",
        "done": "teal",
        "failed": "red",
        "cancelled": "gray",
    }.get(job["status"], "gray")
    right = []
    if job["status"] in ("queued", "running"):
        right.append(
            dmc.Button(
                "Cancel",
                id={"type": "fs-cancel", "job": job["id"]},
                size="compact-xs",
                variant="subtle",
                color="red",
            )
        )
    elif job["status"] == "done" and (job.get("result") or {}).get("run_id"):
        # Finishing does not switch anyone's selected run, the queue is shared,
        # and yanking a colleague off the run they are reading would be worse
        # than an extra click. So offer the click instead.
        right.append(
            dmc.Button(
                "Open run",
                id={"type": "fs-open-run", "run": job["result"]["run_id"]},
                size="compact-xs",
                variant="light",
            )
        )
    elif job["status"] == "failed" and job.get("error"):
        right.append(
            dmc.HoverCard(
                [
                    dmc.HoverCardTarget(
                        dmc.Button(
                            "Details", size="compact-xs", variant="subtle", color="gray"
                        )
                    ),
                    dmc.HoverCardDropdown(
                        dmc.Code(
                            job["error"][-1500:],
                            block=True,
                            style={"maxWidth": "640px", "fontSize": "11px"},
                        )
                    ),
                ],
                position="left",
                shadow="md",
                width=680,
            )
        )
    return dmc.Paper(
        dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.Badge(job["status"], color=colour, variant="light", size="sm"),
                        dmc.Text(job.get("label") or job["id"], size="sm", fw=500),
                        dmc.Text(job.get("message") or "", size="xs", c=TEXT_MUTED),
                        dmc.Group(right, gap="xs", ml="auto"),
                    ],
                    gap="sm",
                    wrap="nowrap",
                ),
                dmc.Progress(
                    value=100 * float(job.get("progress") or 0),
                    size="sm",
                    color=colour,
                    animated=job["status"] == "running",
                    striped=job["status"] == "running",
                )
                if job["status"] in ("queued", "running")
                else None,
            ],
            gap=6,
        ),
        withBorder=True,
        radius="sm",
        p="xs",
    )


@callback(
    Output("fs-jobs", "children"),
    Output("fs-run", "data"),
    Output("fs-run", "value"),
    Output("fs-run-badges", "children"),
    Output("fs-queue-sig", "data"),
    Output("fs-poll", "interval"),
    Input("fs-poll", "n_intervals"),
    State("fs-run", "value"),
    State("fs-queue-sig", "data"),
)
def _poll(_n, current_run, previous):
    """Advance the queue and refresh the run list - but only write what changed.

    This callback fires on a timer, and three of its outputs feed the rest of the
    page (`fs-run.value` drives every tab). Writing them unconditionally re-ran
    the whole page on every tick: figures remounted and the browser lost the
    scroll position. So each output is compared against a signature of what was
    last written and returns `no_update` when nothing moved.

    `fs-run.value` is read as State, never Input: an Output and an Input on the
    same prop makes the callback re-trigger itself, which is what turned a 3s
    timer into a continuous re-render.

    It is also the queue's backstop - if the pump thread died, this keeps jobs
    moving for as long as anyone has the page open.
    """
    jobs.pump()
    active = [
        j
        for j in jobs.list_jobs("factor_model", limit=12)
        if j["status"] != "done" or _recent(j)
    ]
    runs = fstore.list_runs(limit=40)
    run_ids = [r["run_id"] for r in runs]
    busy = any(j["status"] in (jobs.QUEUED, jobs.RUNNING) for j in active)

    # Keep the user's selection unless it has gone away; a colleague's finished
    # run must not move the run someone else is reading.
    value = (
        current_run if current_run in set(run_ids) else (run_ids[0] if run_ids else None)
    )

    sig = {
        "jobs": [
            [
                j["id"],
                j["status"],
                round(float(j.get("progress") or 0), 2),
                j.get("message"),
                bool((j.get("result") or {}).get("run_id")),
            ]
            for j in active[:6]
        ],
        "runs": run_ids,
        "value": value,
        "interval": _POLL_FAST if busy else _POLL_IDLE,
    }
    previous = previous or {}
    if sig == previous:
        return (no_update,) * 6

    changed = lambda key: sig.get(key) != previous.get(key)  # noqa: E731
    return (
        (
            [_job_row(j) for j in active[:6]]
            or dmc.Text("No jobs running.", size="xs", c=TEXT_MUTED)
        )
        if changed("jobs")
        else no_update,
        [{"value": r["run_id"], "label": _run_label(r)} for r in runs]
        if changed("runs")
        else no_update,
        value if changed("value") else no_update,
        _run_badges(value) if (changed("value") or changed("runs")) else no_update,
        sig,
        sig["interval"] if changed("interval") else no_update,
    )


def _recent(job: dict) -> bool:
    """Keep a finished job on screen briefly so the user sees it complete."""
    finished = job.get("finished_at")
    if not finished:
        return True
    try:
        age = (pd.Timestamp.utcnow() - pd.Timestamp(finished)).total_seconds()
    except (ValueError, TypeError):
        return False
    return age < 120


def _run_label(manifest: dict) -> str:
    spec, summary = manifest.get("spec", {}), manifest.get("summary", {})
    created = str(manifest.get("created_at", ""))[:16].replace("T", " ")
    return (
        f"{spec.get('name', 'run')} · {summary.get('n_securities', '?')} securities · "
        f"{_FREQ_LABELS.get(spec.get('frequency'), spec.get('frequency', ''))} · {created}"
    )


def _run_badges(run_id: str | None) -> list:
    if not run_id:
        return []
    manifest = fstore.load_manifest(run_id) or {}
    summary = manifest.get("summary", {})
    out = [
        dmc.Badge(f"{summary.get('n_periods', '?')} periods", variant="light", color="gray")
    ]
    r2 = summary.get("mean_r2")
    if r2 is not None:
        out.append(dmc.Badge(f"mean R² {r2:.2f}", variant="light", color="gray"))
    opf = summary.get("obs_per_factor")
    if opf is not None:
        out.append(
            dmc.Badge(
                f"{opf} obs/factor", variant="light", color="yellow" if opf < 10 else "teal"
            )
        )
    try:
        from capital.data.cache import data_version

        if manifest.get("data_version") and manifest["data_version"] != data_version():
            out.append(
                dmc.Badge("stale: store has newer data", variant="light", color="orange")
            )
    except Exception:
        pass
    return [dmc.Group(out, gap="xs")]


@callback(
    Output("fs-drawer", "opened"),
    Input("fs-open-drawer", "n_clicks"),
    Input("fs-close-drawer", "n_clicks"),
    Input("fs-submit", "n_clicks"),
    prevent_initial_call=True,
)
def _toggle_drawer(open_clicks, close_clicks, submit_clicks):
    return dash.ctx.triggered_id == "fs-open-drawer"


@callback(
    Output("fs-notify", "children"),
    Output("fs-queue-sig", "data", allow_duplicate=True),
    Output("fs-poll", "interval", allow_duplicate=True),
    Input("fs-submit", "n_clicks"),
    State("fs-name", "value"),
    State("fs-start", "value"),
    State("fs-asof", "value"),
    State("fs-freq", "value"),
    State("fs-styles", "value"),
    State("fs-assets", "value"),
    State("fs-etfs", "checked"),
    State("fs-industry", "checked"),
    State("fs-country", "checked"),
    State("fs-robust", "checked"),
    State("fs-maxsec", "value"),
    State("fs-minhist", "value"),
    State("fs-weight", "value"),
    State("fs-winsor", "value"),
    State("fs-cover", "value"),
    State("fs-hl-beta", "value"),
    State("fs-hl-var", "value"),
    State("fs-hl-corr", "value"),
    State("fs-nw", "value"),
    State("fs-numeraire", "value"),
    prevent_initial_call=True,
)
def _submit(
    n,
    name,
    start,
    asof,
    freq,
    styles,
    assets,
    etfs,
    industry,
    country,
    robust,
    maxsec,
    minhist,
    weight,
    winsor,
    cover,
    hl_beta,
    hl_var,
    hl_corr,
    nw,
    numeraire,
):
    if not n:
        return no_update, no_update, no_update
    if not styles:
        return ui.alert("Pick at least one style factor.", "red"), no_update, no_update
    spec = ModelSpec(
        name=name or "Factor model",
        start=start or "",
        as_of=asof or "",
        frequency=freq or "W-FRI",
        styles=tuple(styles),
        asset_types=tuple(assets or ["COMMON"]),
        include_etfs=bool(etfs),
        industry_factors=bool(industry),
        country_factors=bool(country),
        robust=bool(robust),
        max_securities=int(maxsec or 0),
        min_history_days=int(minhist or 250),
        regression_weight=weight or "sqrt_cap",
        winsor_sigma=float(winsor or 3.0),
        min_coverage=float(cover or 0.5),
        beta_halflife=int(hl_beta or 63),
        cov_var_halflife=int(hl_var or 90),
        cov_corr_halflife=int(hl_corr or 252),
        newey_west_lags=int(nw or 5),
        numeraire=numeraire or "local",
    )
    params = spec.to_dict()
    label = (
        f"{spec.name} · {_FREQ_LABELS.get(spec.frequency, spec.frequency)} · "
        f"{len(spec.styles)} styles"
    )
    job_id = jobs.submit(
        "factor_model", params, label=label, dedupe_key=jobs.params_fingerprint(params)
    )
    # Clearing the signature makes the next tick redraw the queue immediately.
    return (
        ui.alert(
            f"Queued as job {job_id}. It runs in the background, this page "
            f"updates as it progresses.",
            "teal",
            title="Run queued",
        ),
        None,
        _POLL_FAST,
    )


@callback(
    Output("fs-queue-sig", "data", allow_duplicate=True),
    Output("fs-poll", "interval", allow_duplicate=True),
    Input({"type": "fs-cancel", "job": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _cancel(clicks):
    trigger = dash.ctx.triggered_id
    if isinstance(trigger, dict) and any(clicks or []):
        jobs.cancel(trigger["job"])
        return None, _POLL_FAST  # clearing the signature forces a redraw
    return no_update, no_update


@callback(
    Output("fs-run", "value", allow_duplicate=True),
    Input({"type": "fs-open-run", "run": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _open_run(clicks):
    trigger = dash.ctx.triggered_id
    if isinstance(trigger, dict) and any(clicks or []):
        return trigger["run"]
    return no_update

    # ── Run-dependent option lists ────────────────────────────────────────────────


def _security_options(bundle: dict) -> list:
    """Options for this run's universe, formatted exactly like every other
    securities picker in the dashboard (see components.security_options)."""
    meta = bundle["meta"]
    if meta.empty:
        return []
    return ui.security_options(meta.reset_index())


def _factor_options(bundle: dict) -> list[dict]:
    styles = bundle["styles"]
    groups = bundle["groups"]
    ordered = [f for f in bundle["factor_returns"].columns]
    return [
        {
            "value": f,
            "label": f"{_label(f, styles)}"
            f"{'' if f in styles or f == 'Market' else ' · ' + groups.get(f, '')}",
        }
        for f in ordered
    ]


@callback(
    Output("fs-trade-sec", "data"),
    Output("fs-sec-pick", "data"),
    Output("fs-sec-pick", "value"),
    Output("fs-screen-x", "data"),
    Output("fs-screen-x", "value"),
    Output("fs-screen-y", "data"),
    Output("fs-screen-y", "value"),
    Output("fs-screen-sector", "data"),
    Output("fs-factor-pick", "data"),
    Output("fs-factor-pick", "value"),
    Input("fs-run", "value"),
)
def _run_options(run_id):
    if not run_id:
        return [], [], None, [], None, [], None, [], [], []
    bundle = _bundle(run_id)
    secs = _security_options(bundle)
    styles = bundle["styles"]
    style_opts = [
        {"value": k, "label": STYLES[k].label if k in STYLES else k} for k in styles
    ]
    sectors = (
        sorted(bundle["meta"]["sector"].dropna().unique())
        if not bundle["meta"].empty
        else []
    )

    weights, _ = _weights_for_run(run_id)
    default_sec = weights.idxmax() if len(weights) else (secs[0]["value"] if secs else None)
    x_default = styles[0] if styles else None
    y_default = styles[1] if len(styles) > 1 else x_default
    return (
        secs,
        secs,
        default_sec,
        style_opts,
        x_default,
        style_opts,
        y_default,
        sectors,
        _factor_options(bundle),
        styles[:MAX_SERIES],
    )

    # ── Portfolio tab ─────────────────────────────────────────────────────────────


def _pct(x, digits: int = 1) -> str:
    return " - " if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def _styles_only(series: pd.Series, bundle: dict) -> pd.Series:
    keys = [k for k in bundle["styles"] if k in series.index]
    out = series.loc[keys]
    out.index = [STYLES[k].label if k in STYLES else k for k in keys]
    return out


@callback(
    Output("fs-portfolio-content", "children"),
    Input("fs-run", "value"),
    Input("fs-tabs", "value"),
)
def _portfolio_tab(run_id, tab):
    if tab != "portfolio":
        return no_update
    if not run_id:
        return ui.alert("Queue a model run to see the portfolio's factor exposures.")
    bundle = _bundle(run_id)
    weights, cov_note = _weights_for_run(run_id)
    if weights.empty:
        return ui.alert(
            "None of the current positions are covered by this run. "
            "Check that these positions are part of the covered universe "
            "and have enough price history.",
            "yellow",
        )

    decomp = riskmod.decompose_risk(
        weights,
        bundle["exposures"],
        bundle["cov"],
        bundle["spec_risk"]["sigma"],
        bundle["groups"],
    )
    if not decomp:
        return ui.alert("Could not decompose risk for this run.", "yellow")

    contrib = riskmod.security_contributions(
        weights, bundle["exposures"], bundle["cov"], bundle["spec_risk"]["sigma"]
    )
    names = cov_note["names"]

    metrics = ui.metric_row(
        [
            (
                "Portfolio volatility",
                _pct(decomp["total_vol"]),
                "annualised, model-implied",
            ),
            (
                "From factors",
                _pct(decomp["factor_share"]),
                f"{_pct(decomp['factor_vol'])} vol",
            ),
            (
                "Specific (diversifiable)",
                _pct(decomp["specific_share"]),
                f"{_pct(decomp['specific_vol'])} vol",
            ),
            (
                "Positions covered",
                f"{cov_note['n']} / {cov_note['total']}",
                f"{_pct(cov_note['covered'])} of NAV",
            ),
            ("Cash", _pct(cov_note["cash"]), "excluded from exposures"),
        ]
    )

    style_exp = _styles_only(decomp["exposure"], bundle)
    top_contrib = decomp["contribution"].reindex(
        decomp["contribution"].abs().sort_values(ascending=False).index[:16]
    )
    top_contrib.index = [_label(f, bundle["styles"]) for f in top_contrib.index]

    group_series = (
        decomp["by_group"]
        .reindex([g for g in _GROUP_ORDER if g in decomp["by_group"].index])
        .dropna()
    )
    group_series["Specific"] = decomp["specific_vol"]

    table = contrib.join(bundle["meta"][["ticker", "name", "sector"]], how="left")
    table["ticker"] = table["ticker"].fillna(pd.Series(names))
    table = table.reset_index(names="security_id")
    table = table[
        [
            "ticker",
            "name",
            "sector",
            "weight",
            "factor_ctr",
            "specific_ctr",
            "total_ctr",
            "pct_of_risk",
            "specific_vol",
        ]
    ].rename(columns={"weight": "weight_of_covered"})

    warning = []
    if cov_note["missing"]:
        warning.append(
            ui.alert(
                "Not covered by this run: "
                + ", ".join(cov_note["missing"][:20])
                + ("…" if len(cov_note["missing"]) > 20 else "")
                + ". Exposures below are for the covered part of the book only.",
                "yellow",
            )
        )

    return dmc.Stack(
        [
            metrics,
            *warning,
            ui.section("Factor tilts"),
            dcc.Markdown(
                "The market has exposure 0 to every style by construction, so "
                "these bars *are* the portfolio's active bet against the market.",
                className="explain-body",
            ),
            _graph(
                _hbar(
                    style_exp,
                    "Portfolio style exposures",
                    f"Cap-weighted market = 0 · {cov_note['n']} positions · "
                    f"{_pct(cov_note['covered'])} of NAV covered",
                )
            ),
            ui.section("Where the risk comes from"),
            dmc.SimpleGrid(
                [
                    _graph(
                        _hbar(
                            top_contrib,
                            "Risk contribution by factor",
                            "Annualised volatility points · sums to factor volatility",
                            xtitle="Contribution to volatility",
                            fmt="{:+.2%}",
                        )
                    ),
                    _graph(
                        _hbar(
                            group_series,
                            "Risk by factor group",
                            "Specific risk is the diversifiable remainder",
                            xtitle="Contribution to volatility",
                            fmt="{:+.2%}",
                            height=420,
                        )
                    ),
                ],
                cols={"base": 1, "lg": 2},
            ),
            ui.explain(
                "Why a small tilt can dominate the risk chart",
                "A factor's risk contribution is its exposure multiplied by how "
                "that factor covaries with the rest of the portfolio: "
                "$x_k (Fx)_k / \\sigma_P$. A modest tilt to a volatile factor that "
                "correlates with your other tilts costs more than a large tilt to a "
                "quiet, uncorrelated one. That is why this chart and the exposure "
                "chart above rank factors differently, and why this one is the "
                "sizing chart.",
            ),
            ui.section("Contribution by position"),
            _table(table.round(4), id="fs-positions-table", page_size=20),
        ],
        gap="sm",
    )

    # ── What-if tab ───────────────────────────────────────────────────────────────


@callback(
    Output("fs-trades", "data"),
    Input("fs-trade-add", "n_clicks"),
    Input("fs-trade-clear", "n_clicks"),
    Input({"type": "fs-trade-del", "sid": ALL}, "n_clicks"),
    State("fs-trade-sec", "value"),
    State("fs-trade-w", "value"),
    State("fs-trades", "data"),
    prevent_initial_call=True,
)
def _edit_trades(add, clear, dels, sid, delta, trades):
    trigger = dash.ctx.triggered_id
    trades = list(trades or [])
    if trigger == "fs-trade-clear":
        return []
    if isinstance(trigger, dict) and trigger.get("type") == "fs-trade-del":
        if any(dels or []):
            return [t for t in trades if t["sid"] != trigger["sid"]]
        return no_update
    if trigger == "fs-trade-add" and sid and delta:
        trades = [t for t in trades if t["sid"] != sid]
        trades.append({"sid": sid, "delta": float(delta) / 100.0})
    return trades


@callback(
    Output("fs-trade-list", "children"),
    Input("fs-trades", "data"),
    State("fs-run", "value"),
)
def _render_trades(trades, run_id):
    if not trades:
        return dmc.Text(
            "No hypothetical trades yet: add one above to see its effect.",
            size="sm",
            c=TEXT_MUTED,
        )
    meta = _bundle(run_id)["meta"] if run_id else pd.DataFrame()
    chips = []
    for t in trades:
        ticker = meta.loc[t["sid"], "ticker"] if t["sid"] in meta.index else t["sid"]
        sign = "+" if t["delta"] >= 0 else "−"
        chips.append(
            dmc.Badge(
                dmc.Group(
                    [
                        dmc.Text(
                            f"{ticker}  {sign}{abs(t['delta']):.2%}", size="xs", fw=600
                        ),
                        dmc.ActionIcon(
                            "×",
                            id={"type": "fs-trade-del", "sid": t["sid"]},
                            size="xs",
                            variant="transparent",
                            color="gray",
                        ),
                    ],
                    gap=4,
                    wrap="nowrap",
                ),
                variant="light",
                color="navy" if t["delta"] >= 0 else "gray",
                size="lg",
                radius="sm",
            )
        )
    return dmc.Group(chips, gap="xs")


@callback(
    Output("fs-whatif-content", "children"),
    Input("fs-run", "value"),
    Input("fs-tabs", "value"),
    Input("fs-trades", "data"),
    Input("fs-funding", "value"),
)
def _whatif_tab(run_id, tab, trades, funding):
    if tab != "whatif":
        return no_update
    if not run_id:
        return ui.alert("Queue a model run first.")
    bundle = _bundle(run_id)
    before, cov_note = _weights_for_run(run_id)
    if before.empty:
        return ui.alert("No covered positions to trade against.", "yellow")
    if not trades:
        return ui.alert(
            "Add a hypothetical trade above. The model will show what it "
            "does to the portfolio's exposures and risk: nothing is "
            "executed or saved."
        )

    after = riskmod.apply_trades(
        before, {t["sid"]: t["delta"] for t in trades}, funding=funding or "pro_rata"
    )
    result = riskmod.compare_portfolios(
        before,
        after,
        bundle["exposures"],
        bundle["cov"],
        bundle["spec_risk"]["sigma"],
        bundle["groups"],
    )
    if not result.get("after"):
        return ui.alert("Could not evaluate the proposed portfolio.", "yellow")

    b, a = result["before"], result["after"]
    metrics = ui.metric_row(
        [
            ("Volatility now", _pct(b["total_vol"]), "annualised"),
            (
                "Volatility after",
                _pct(a["total_vol"]),
                f"{'+' if result['vol_delta'] >= 0 else ''}{100 * result['vol_delta']:.2f} pts",
            ),
            ("Factor share", _pct(a["factor_share"]), f"was {_pct(b['factor_share'])}"),
            ("Turnover", _pct(result["turnover"]), "one-way, of NAV"),
            ("Positions after", f"{int((after.abs() > 1e-6).sum())}", f"was {len(before)}"),
        ]
    )

    keys = [k for k in bundle["styles"]]
    labels = [STYLES[k].label if k in STYLES else k for k in keys]
    compare = pd.DataFrame(
        {
            "Current": b["exposure"].reindex(keys).values,
            "Proposed": a["exposure"].reindex(keys).values,
        },
        index=labels,
    )
    delta = pd.Series(result["exposure_delta"].reindex(keys).values, index=labels)

    ctr_delta = result["contribution_delta"]
    ctr_delta = ctr_delta.reindex(ctr_delta.abs().sort_values(ascending=False).index[:14])
    ctr_delta.index = [_label(f, bundle["styles"]) for f in ctr_delta.index]

    return dmc.Stack(
        [
            metrics,
            _graph(
                _grouped_hbar(
                    compare,
                    "Style exposures: current vs proposed",
                    f"Funded {'pro rata across the book' if funding == 'pro_rata' else 'from cash'}",
                )
            ),
            dmc.SimpleGrid(
                [
                    _graph(
                        _hbar(
                            delta,
                            "Exposure change",
                            "Proposed minus current",
                            xtitle="Δ exposure (z-score)",
                        )
                    ),
                    _graph(
                        _hbar(
                            ctr_delta,
                            "Risk contribution change",
                            "Annualised volatility points",
                            xtitle="Δ contribution to volatility",
                            fmt="{:+.2%}",
                        )
                    ),
                ],
                cols={"base": 1, "lg": 2},
            ),
            ui.explain(
                "Reading the two charts together",
                "The left chart is what you *intended*: the tilt you moved. The "
                "right chart is what it *cost*. They disagree whenever the factor "
                "you moved is unusually volatile or correlated with tilts you "
                "already hold, and the disagreement is the point: a trade that "
                "barely changes exposures but sharply raises risk contribution is "
                "concentrating an existing bet, not adding a new one.",
            ),
        ],
        gap="sm",
    )

    # ── Screen tab ────────────────────────────────────────────────────────────────


@callback(
    Output("fs-screen-content", "children"),
    Input("fs-run", "value"),
    Input("fs-tabs", "value"),
    Input("fs-screen-x", "value"),
    Input("fs-screen-y", "value"),
    Input("fs-screen-sector", "value"),
)
def _screen_tab(run_id, tab, x_key, y_key, sectors):
    if tab != "screen":
        return no_update
    if not run_id:
        return ui.alert("Queue a model run first.")
    bundle = _bundle(run_id)
    snap = fstore.exposure_snapshot(run_id)
    if snap.empty:
        return ui.alert("This run has no exposure panel.", "yellow")

    meta = bundle["meta"]
    df = meta.join(snap, how="left")
    if sectors:
        df = df[df["sector"].isin(sectors)]
    weights, _ = _weights_for_run(run_id)
    df["held"] = ["yes" if sid in weights.index else "" for sid in df.index]

    style_keys = [k for k in bundle["styles"] if k in df.columns]
    table = df.reset_index()[
        ["ticker", "name", "sector", "country", "method", "held", "market_cap", *style_keys]
    ]
    table = table.rename(
        columns={k: (STYLES[k].label if k in STYLES else k) for k in style_keys}
    )
    table["market_cap"] = (table["market_cap"] / 1e9).round(2)
    table = table.rename(columns={"market_cap": "mcap_bn"})

    scatter = None
    if x_key and y_key and x_key in df.columns and y_key in df.columns:
        sub = df.dropna(subset=[x_key, y_key])
        # Marker size encodes market-cap rank, but the *range* has to shrink as
        # the universe grows: 34px markers are informative for 100 names and an
        # unreadable smear for 1,500.
        dense = len(sub) > 600
        base, span = (5, 9) if dense else (8, 26)
        size = sub["market_cap"].fillna(sub["market_cap"].median())
        size = base + span * (size.rank(pct=True) if len(size) > 1 else 1.0)
        held = sub["held"] == "yes"
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sub.loc[~held, x_key],
                y=sub.loc[~held, y_key],
                mode="markers",
                name="Universe",
                marker=dict(
                    size=size[~held],
                    color=NEG,
                    opacity=0.30 if dense else 0.45,
                    line=dict(width=0 if dense else 2, color="#FFFFFF"),
                ),
                text=sub.loc[~held, "ticker"],
                customdata=sub.loc[~held, ["name", "sector"]].values,
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}"
                "<br>x %{x:.2f} · y %{y:.2f}<extra></extra>",
            )
        )
        if held.any():
            fig.add_trace(
                go.Scatter(
                    x=sub.loc[held, x_key],
                    y=sub.loc[held, y_key],
                    mode="markers" if dense else "markers+text",
                    name="Held",
                    marker=dict(
                        size=size[held] + (4 if dense else 0),
                        color=POS,
                        line=dict(width=2, color="#FFFFFF"),
                    ),
                    text=sub.loc[held, "ticker"],
                    textposition="top center",
                    textfont=dict(size=10, color="#334155"),
                    customdata=sub.loc[held, ["name", "sector"]].values,
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}"
                    "<br>x %{x:.2f} · y %{y:.2f}<extra></extra>",
                )
            )
        fig.add_hline(y=0, line_width=1, line_color=BORDER)
        fig.add_vline(x=0, line_width=1, line_color=BORDER)
        fig = _fig(
            fig,
            f"{_label(x_key, bundle['styles'])} vs {_label(y_key, bundle['styles'])}",
            f"{len(sub)} securities · marker size = market cap rank · latest cross-section",
            height=560,
        )
        fig.update_layout(
            xaxis=dict(title=_label(x_key, bundle["styles"]) + " (z-score)"),
            yaxis=dict(title=_label(y_key, bundle["styles"]) + " (z-score)"),
        )
        _legend_below(fig, has_xtitle=True)
        scatter = _graph(fig)

    return dmc.Stack(
        [
            ui.metric_row(
                [
                    ("Securities", f"{len(df)}", "in this run"),
                    (
                        "Cross-sectional",
                        f"{int((df['method'] == 'cross-sectional').sum())}",
                        "single stocks",
                    ),
                    (
                        "Returns-based",
                        f"{int((df['method'] == 'returns-based').sum())}",
                        "ETFs and funds",
                    ),
                    ("Held", f"{int((df['held'] == 'yes').sum())}", "current positions"),
                ],
                cols={"base": 2, "sm": 4},
            ),
            # `scatter if ... is not None` deliberately, not `or`: a Dash component
            # with no children is falsy, so `scatter or fallback` silently discards
            # every figure.
            scatter
            if scatter is not None
            else dmc.Text("Pick two factors to plot.", size="sm", c=TEXT_MUTED),
            ui.section("The full screen"),
            _table(table.round(3), id="fs-screen-table", page_size=25, height=620),
        ],
        gap="sm",
    )

    # ── Factors tab ───────────────────────────────────────────────────────────────


@callback(
    Output("fs-factors-content", "children"),
    Input("fs-run", "value"),
    Input("fs-tabs", "value"),
    Input("fs-factor-pick", "value"),
)
def _factors_tab(run_id, tab, picked):
    if tab != "factors":
        return no_update
    if not run_id:
        return ui.alert("Queue a model run first.")
    bundle = _bundle(run_id)
    styles = bundle["styles"]
    picked = [p for p in (picked or styles)][:MAX_SERIES] or styles[:MAX_SERIES]
    freq = _FREQ_LABELS.get(bundle["spec"].get("frequency"), "")

    cum = fstore.load_diagnostic(run_id, "cumulative_returns")
    perf = fstore.load_diagnostic(run_id, "factor_performance")
    ic = fstore.load_diagnostic(run_id, "ic_summary")
    qmeans = fstore.load_diagnostic(run_id, "quantile_means")
    persistence = fstore.load_diagnostic(run_id, "persistence")
    vif = fstore.load_diagnostic(run_id, "vif")
    subperiod = fstore.load_diagnostic(run_id, "subperiod_returns")
    corr = bundle["corr"]
    fit = bundle["fit"]

    children = [
        ui.metric_row(
            [
                (
                    "Periods estimated",
                    f"{bundle['summary'].get('n_periods', ' - ')}",
                    freq.lower(),
                ),
                (
                    "Factors",
                    f"{bundle['summary'].get('n_factors', ' - ')}",
                    "incl. industry & country",
                ),
                (
                    "Mean cross-sectional R²",
                    f"{bundle['summary'].get('mean_r2', float('nan')):.2f}"
                    if bundle["summary"].get("mean_r2") is not None
                    else " - ",
                    "per period",
                ),
                (
                    "Observations per factor",
                    f"{bundle['summary'].get('obs_per_factor', ' - ')}",
                    "below 10 = thin",
                ),
            ],
            cols={"base": 2, "sm": 4},
        )
    ]

    if bundle["summary"].get("obs_per_factor") and bundle["summary"]["obs_per_factor"] < 10:
        children.append(
            ui.alert(
                f"This run fits {bundle['summary'].get('n_factors')} factors to about "
                f"{bundle['summary'].get('obs_per_factor')} securities per factor. Style "
                f"factors are still informative, but industry and country returns are "
                f"fitted on very few names each, read them as indicative. Expanding the "
                f"A broader universe is the fix.",
                "yellow",
                title="Thin cross-section",
            )
        )

        # -- cumulative factor returns --
    if not cum.empty:
        cum = cum.set_index(cum.columns[0])
        cum.index = pd.to_datetime(cum.index)
        cols = [c for c in picked if c in cum.columns]
        if cols:
            children += [
                ui.section("What each factor paid"),
                _graph(
                    _lines(
                        cum[cols].rename(columns={c: _label(c, styles) for c in cols}),
                        "Cumulative factor returns",
                        "One unit of exposure, held continuously",
                        ytitle="Cumulative return",
                    )
                ),
                ui.explain(
                    "What this line actually represents",
                    "Each line is the return to holding exactly one unit of "
                    "exposure to that factor while holding zero exposure to "
                    "every other factor: a long/short portfolio, rebuilt every "
                    "period, that no one could trade frictionlessly. It is the "
                    "*price of the tilt*, not a strategy return: costs, "
                    "borrow and capacity are all absent.",
                ),
            ]

    if not perf.empty:
        # Group order, not alphabetical, otherwise country dummies head a table
        # whose first screenful should be the styles.
        rank = {g: i for i, g in enumerate(_GROUP_ORDER)}
        perf = perf.copy()
        perf["group"] = [bundle["groups"].get(f, "Other") for f in perf["factor"]]
        perf["factor"] = [_label(f, styles) for f in perf["factor"]]
        perf = perf.sort_values(
            ["group", "factor"], key=lambda c: c.map(rank) if c.name == "group" else c
        )
        children += [
            _table(
                perf[
                    [
                        "factor",
                        "group",
                        "ann_return",
                        "ann_vol",
                        "sharpe",
                        "t_stat",
                        "hit_rate",
                        "max_drawdown",
                        "n_periods",
                    ]
                ].round(4),
                id="fs-perf-table",
                page_size=14,
                height=520,
            )
        ]

        # -- information coefficient --
    if not ic.empty:
        horizons = sorted(ic["horizon"].unique())
        base = 21 if 21 in horizons else horizons[min(2, len(horizons) - 1)]
        snap = ic[ic["horizon"] == base].copy()
        snap["label"] = [_label(f, styles) for f in snap["factor"]]
        ir = snap.set_index("label")["ic_ir"].sort_values()
        decay = ic.pivot(index="horizon", columns="factor", values="ic_mean")
        decay = decay[[c for c in picked if c in decay.columns]]
        decay.columns = [_label(c, styles) for c in decay.columns]

        fig_decay = go.Figure()
        for i, col in enumerate(decay.columns[:MAX_SERIES]):
            fig_decay.add_trace(
                go.Scatter(
                    x=decay.index,
                    y=decay[col],
                    mode="lines+markers",
                    name=col,
                    line=dict(color=SERIES[i % len(SERIES)], width=2),
                    marker=dict(size=8, line=dict(width=2, color="#FFFFFF")),
                    hovertemplate=f"{col}<br>%{{x}} days: IC %{{y:.3f}}<extra></extra>",
                )
            )
        fig_decay = _fig(
            fig_decay,
            "IC decay by horizon",
            "Mean rank correlation of exposure with forward return",
            420,
        )
        fig_decay.update_layout(
            xaxis=dict(
                title="Forward horizon (trading days)",
                type="log",
                tickvals=horizons,
                ticktext=[str(h) for h in horizons],
            ),
            yaxis=dict(title="Mean IC"),
        )
        fig_decay.add_hline(y=0, line_width=1, line_color=BORDER)
        _legend_below(fig_decay, has_xtitle=True)

        children += [
            ui.section("Is the factor real?"),
            dmc.SimpleGrid(
                [
                    _graph(
                        _hbar(
                            ir,
                            f"IC information ratio · {base}-day horizon",
                            "Mean IC divided by its own volatility",
                            xtitle="IC IR",
                            fmt="{:+.2f}",
                        )
                    ),
                    _graph(fig_decay),
                ],
                cols={"base": 1, "lg": 2},
            ),
            ui.explain("Information coefficient: the full definition", txt.IC_EXPLAIN),
            ui.explain("Reading the decay curve", txt.DECAY_EXPLAIN),
            _table(
                snap[
                    [
                        "label",
                        "horizon",
                        "ic_mean",
                        "ic_std",
                        "ic_ir",
                        "t_stat",
                        "hit_rate",
                        "n_periods",
                    ]
                ].round(4),
                id="fs-ic-table",
                page_size=12,
            ),
        ]

        # -- quantile spreads --
    if not qmeans.empty:
        sub = qmeans[qmeans["factor"].isin(picked)].copy()
        sub = sub[sub["quantile"] != "Top-Bottom"].sort_values(["factor", "q_order"])
        fig_q = go.Figure()
        for i, (fac, grp) in enumerate(sub.groupby("factor")):
            fig_q.add_trace(
                go.Bar(
                    x=grp["quantile"],
                    y=grp["mean_return"],
                    name=_label(fac, styles),
                    marker=dict(
                        color=SERIES[i % len(SERIES)], line=dict(width=2, color="#FFFFFF")
                    ),
                    hovertemplate=f"{_label(fac, styles)} · %{{x}}: %{{y:.2%}}<extra></extra>",
                )
            )
        fig_q = _fig(
            fig_q,
            "Quantile portfolios",
            "Annualised cap-weighted return of each exposure bucket",
            440,
        )
        fig_q.update_layout(
            barmode="group",
            bargap=0.25,
            bargroupgap=0.06,
            yaxis=dict(title="Annualised return", tickformat=".0%"),
            xaxis=dict(title="Exposure quantile (1 = lowest)"),
        )
        _legend_below(fig_q, has_xtitle=True)
        children += [
            _graph(fig_q),
            ui.explain("Why monotonicity matters", txt.QUANTILE_EXPLAIN),
        ]

        # -- correlation and sub-periods --
    grid = []
    if not corr.empty:
        keys = [f for f in corr.columns if f in styles or f == "Market"]
        if len(keys) > 1:
            c = corr.loc[keys, keys]
            c.index = [_label(f, styles) for f in c.index]
            c.columns = [_label(f, styles) for f in c.columns]
            grid.append(
                _graph(
                    _heatmap(
                        c,
                        "Factor return correlations",
                        "Estimated from the factor returns, not the exposures",
                        zmax=1.0,
                        colorbar_title="ρ",
                    )
                )
            )
    if not subperiod.empty:
        sp = subperiod.set_index(subperiod.columns[0])
        keys = [f for f in sp.columns if f in styles or f == "Market"]
        if keys:
            sp = sp[keys]
            sp.columns = [_label(f, styles) for f in sp.columns]
            grid.append(
                _graph(
                    _heatmap(
                        sp,
                        "Factor returns by calendar year",
                        "Compounded within each year",
                        fmt="{:+.1%}",
                        colorbar_title="Return",
                    )
                )
            )
    if grid:
        children += [
            ui.section("Stability"),
            dmc.SimpleGrid(grid, cols={"base": 1, "lg": 2}),
            ui.explain(
                "Correlations are of returns, not exposures", txt.CORRELATION_EXPLAIN
            ),
            ui.explain("Read the row, not the average", txt.SUBPERIOD_EXPLAIN),
        ]

        # -- persistence, VIF, fit --
    tables = []
    if not persistence.empty:
        p = persistence.copy()
        p["factor"] = [_label(f, styles) for f in p["factor"]]
        tables.append(
            dmc.Stack(
                [
                    dmc.Text("Exposure persistence", fw=600, size="sm", c=NAVY),
                    _table(p.round(3), id="fs-persist-table", page_size=14, height=420),
                ],
                gap=4,
            )
        )
    if not vif.empty:
        v = vif.copy()
        v["factor"] = [_label(f, styles) for f in v["factor"]]
        tables.append(
            dmc.Stack(
                [
                    dmc.Text("Multicollinearity (VIF)", fw=600, size="sm", c=NAVY),
                    _table(v.round(2), id="fs-vif-table", page_size=14, height=420),
                ],
                gap=4,
            )
        )
    if tables:
        children += [
            ui.section("Diagnostics"),
            dmc.SimpleGrid(tables, cols={"base": 1, "lg": 2}),
            ui.explain("How much a security's exposure moves", txt.PERSISTENCE_EXPLAIN),
            ui.explain("When a factor return stops meaning anything", txt.VIF_EXPLAIN),
        ]

    if not fit.empty:
        fig_fit = go.Figure(
            go.Scatter(
                x=fit.index,
                y=fit["r2"],
                mode="lines",
                line=dict(color=SERIES[0], width=2),
                hovertemplate="%{x|%d %b %Y}: R² %{y:.2f}<extra></extra>",
                name="R²",
            )
        )
        fig_fit = _fig(
            fig_fit,
            "Cross-sectional fit over time",
            "Share of each period's return dispersion the factors explain",
            340,
        )
        fig_fit.update_layout(yaxis=dict(title="R²", tickformat=".0%"), showlegend=False)
        children += [
            _graph(fig_fit),
            ui.explain("What a high R² does and does not mean", txt.FIT_EXPLAIN),
        ]

    return dmc.Stack(children, gap="sm")

    # ── Securities tab ────────────────────────────────────────────────────────────


@callback(
    Output("fs-securities-content", "children"),
    Input("fs-run", "value"),
    Input("fs-tabs", "value"),
    Input("fs-sec-pick", "value"),
    Input("fs-factor-pick", "value"),
)
def _securities_tab(run_id, tab, sid, picked):
    if tab != "securities":
        return no_update
    if not run_id:
        return ui.alert("Queue a model run first.")
    if not sid:
        return ui.alert("Pick a security above.")
    bundle = _bundle(run_id)
    styles = bundle["styles"]
    picked = [p for p in (picked or styles) if p in styles][:MAX_SERIES] or styles[
        :MAX_SERIES
    ]

    meta = bundle["meta"]
    row = meta.loc[sid] if sid in meta.index else None
    if row is None:
        return ui.alert("That security is not in this run.", "yellow")

    long = fstore.load_exposures(run_id, security_ids=(sid,))
    if long.empty:
        return ui.alert("No exposure history stored for this security.", "yellow")
    history = long.pivot(index="date", columns="factor", values="value").sort_index()

    stability = fstore.load_diagnostic(run_id, "security_stability")
    stability = (
        stability[stability["security_id"] == sid] if not stability.empty else stability
    )
    fit = fstore.load_diagnostic(run_id, "security_fit")
    fit_row = fit[fit["security_id"] == sid] if not fit.empty else pd.DataFrame()
    sigma = bundle["spec_risk"]["sigma"].get(sid, np.nan)
    weights, _ = _weights_for_run(run_id)

    metrics = ui.metric_row(
        [
            (
                "Model R²",
                _pct(float(fit_row["r2"].iloc[0])) if len(fit_row) else " - ",
                "return variance explained",
            ),
            (
                "Total volatility",
                _pct(float(fit_row["total_vol"].iloc[0])) if len(fit_row) else " - ",
                "annualised",
            ),
            (
                "Specific volatility",
                _pct(float(sigma)) if np.isfinite(sigma) else " - ",
                "idiosyncratic, diversifiable",
            ),
            ("Portfolio weight", _pct(float(weights.get(sid, 0.0)), 2), "of covered NAV"),
            (
                "Exposure method",
                str(row.get("method", " - ")),
                f"{row.get('sector', '')} · {row.get('country', '')}",
            ),
        ]
    )

    cols = [c for c in picked if c in history.columns]
    charts = []
    if cols:
        named = history[cols].rename(columns={c: _label(c, styles) for c in cols})
        fig = _lines(
            named,
            f"{row['ticker']}, exposure history",
            "Z-scores against the cap-weighted market",
            ytitle="Exposure (z-score)",
            height=460,
            percent=False,
        )
        fig.add_hline(y=0, line_width=1, line_color=BORDER)
        for band, opacity in ((1.0, 0.06), (2.0, 0.03)):
            fig.add_hrect(
                y0=-band,
                y1=band,
                fillcolor=NEG,
                opacity=opacity,
                line_width=0,
                layer="below",
            )
        charts.append(_graph(fig))

    latest = history.iloc[-1].reindex([k for k in styles if k in history.columns])
    latest.index = [_label(k, styles) for k in latest.index]
    charts.append(
        _graph(
            _hbar(
                latest,
                f"{row['ticker']}, latest exposures",
                f"As of {history.index[-1].date()}",
            )
        )
    )

    tables = []
    if not stability.empty:
        st = stability.copy()
        st["factor"] = [_label(f, styles) for f in st["factor"]]
        tables.append(
            _table(
                st[
                    ["factor", "last", "mean", "std", "min", "max", "autocorr", "n_obs"]
                ].round(3),
                id="fs-stability-table",
                page_size=14,
                height=460,
            )
        )

    return dmc.Stack(
        [
            dmc.Group(
                [
                    dmc.Title(f"{row['ticker']}, {row['name']}", order=3, c=NAVY),
                    dmc.Badge(str(row.get("method", "")), variant="light", color="gray"),
                ],
                gap="sm",
                align="baseline",
            ),
            metrics,
            dmc.SimpleGrid(charts, cols={"base": 1, "lg": 2}),
            ui.explain(
                "Why the shaded bands are there",
                "The inner band is ±1 exposure standard deviation and the outer ±2. "
                "An exposure that stays inside the inner band is, for practical "
                "purposes, market-neutral on that factor no matter what the sign "
                "says. A line that crosses the bands repeatedly is the signature of "
                "an unstable characteristic: see the autocorrelation column below.",
            ),
            ui.section("Exposure stability"),
            *tables,
        ],
        gap="sm",
    )

    # ── Methodology tab ───────────────────────────────────────────────────────────


_METHOD_SECTIONS = [
    ("What this model answers", txt.METHOD_WHAT),
    ("How a run is executed", txt.METHOD_JOBS),
    ("Overview", txt.METHOD_OVERVIEW),
    ("1 · Descriptors", txt.METHOD_DESCRIPTORS),
    ("2 · Standardisation", txt.METHOD_STANDARDISATION),
    ("3 · Style factors", txt.METHOD_STYLES),
    ("4 · Cross-sectional regression", txt.METHOD_REGRESSION),
    ("5 · Risk model", txt.METHOD_RISK),
    ("6 · ETFs and funds", txt.METHOD_RETURNS_BASED),
    ("7 · Limitations", txt.METHOD_LIMITS),
    ("Symbols", txt.METHOD_GLOSSARY),
]


def _md(body: str) -> dcc.Markdown:
    return dcc.Markdown(body, mathjax=True, className="explain-body")


def _factor_reference() -> dmc.Accordion:
    """One expandable card per style factor: what it measures, how it is built,
    and how to read a positive exposure."""
    items = []
    for style in STYLES.values():
        descs = [DESCRIPTORS[k] for k in style.weights if k in DESCRIPTORS]
        rows = "\n".join(
            f"| `{d.key}` | {d.label} | {style.weights[d.key]:.2f} | {d.detail} |"
            for d in descs
        )
        orth = (
            f"\n\n**Orthogonalised against:** "
            f"{', '.join(STYLES[o].label for o in style.orthogonalise_to if o in STYLES)}."
            if style.orthogonalise_to
            else ""
        )
        body = (
            f"{style.summary}\n\n**Reading it.** {style.interpretation}\n\n"
            f"**Built from**\n\n| Descriptor | Name | Weight | Definition |\n"
            f"| --- | --- | --- | --- |\n{rows}{orth}\n\n"
            "Descriptor weights are renormalised over whichever inputs the "
            "store actually has, so a missing input shrinks the blend rather "
            "than voiding the factor."
        )
        items.append(
            dmc.AccordionItem(
                value=style.key,
                children=[
                    dmc.AccordionControl(
                        dmc.Group(
                            [
                                dmc.Text(style.label, fw=600, c=NAVY, size="sm"),
                                dmc.Text(style.summary, size="xs", c=TEXT_MUTED),
                            ],
                            gap="sm",
                            wrap="nowrap",
                        )
                    ),
                    dmc.AccordionPanel(_md(body)),
                ],
            )
        )
    return dmc.Accordion(items, variant="separated", radius="sm", chevronPosition="left")


@callback(Output("fs-method-content", "children"), Input("fs-tabs", "value"))
def _method_tab(tab):
    """Rendered on demand rather than in the layout: it is ~70 MathJax blocks,
    and typesetting them on every page load would slow the tabs people actually
    open first."""
    if tab != "method":
        return no_update
    sections = []
    for title, body in _METHOD_SECTIONS:
        sections.append(dmc.Title(title, order=3, c=NAVY, mt="xl", mb="xs"))
        sections.append(_md(body))
    return dmc.Stack(
        [
            ui.section("The factor set"),
            _md(
                "Every style factor the model can estimate. Click one to see the "
                "descriptors it is built from and how to read its exposure."
            ),
            _factor_reference(),
            html.Div(sections),
        ],
        gap="xs",
    )

    # ── Run details tab ───────────────────────────────────────────────────────────


def _kv_table(data: dict) -> dmc.Table:
    rows = [
        [str(k).replace("_", " "), json.dumps(v) if isinstance(v, (list, dict)) else str(v)]
        for k, v in data.items()
    ]
    return dmc.Table(
        data={"head": ["Setting", "Value"], "body": rows},
        striped=True,
        withTableBorder=True,
        fz="sm",
        verticalSpacing=5,
    )


@callback(
    Output("fs-run-content", "children"),
    Input("fs-run", "value"),
    Input("fs-tabs", "value"),
)
def _run_tab(run_id, tab):
    if tab != "run":
        return no_update
    if not run_id:
        return ui.alert("Queue a model run first.")
    bundle = _bundle(run_id)
    coverage, spec = bundle["coverage"], bundle["spec"]

    style_rows = []
    for key, info in (coverage.get("styles") or {}).items():
        style_rows.append(
            [
                STYLES[key].label if key in STYLES else key,
                ", ".join(info.get("descriptors_used", [])) or " - ",
                ", ".join(info.get("descriptors_missing", [])) or " - ",
                ", ".join(
                    STYLES[o].label
                    for o in info.get("orthogonalised_to", [])
                    if o in STYLES
                )
                or " - ",
            ]
        )

    dropped_styles = coverage.get("dropped_styles") or {}
    dropped_desc = coverage.get("dropped_descriptors") or []

    notes = []
    if dropped_styles:
        notes.append(
            ui.alert(
                dmc.Stack(
                    [dmc.Text("Style factors dropped from this run:", size="sm", fw=600)]
                    + [
                        dmc.Text(
                            f"· {STYLES[k].label if k in STYLES else k}, {v}", size="sm"
                        )
                        for k, v in dropped_styles.items()
                    ],
                    gap=2,
                ),
                "yellow",
            )
        )
    if dropped_desc:
        notes.append(
            ui.alert(
                "Descriptors unavailable: "
                + "; ".join(f"{d['descriptor']} ({d['reason']})" for d in dropped_desc),
                "gray",
            )
        )

    desc_cov = coverage.get("descriptors") or {}
    cov_df = pd.DataFrame(
        [
            {
                "descriptor": k,
                "mean_coverage": v.get("mean_coverage"),
                "dates_usable": v.get("dates_usable"),
                "dates_total": v.get("dates_total"),
            }
            for k, v in desc_cov.items()
        ]
    )

    return dmc.Stack(
        [
            ui.metric_row(
                [
                    (
                        "Run id",
                        run_id[-8:],
                        str(bundle["manifest"].get("created_at", ""))[:16],
                    ),
                    (
                        "Securities",
                        f"{bundle['summary'].get('n_securities', ' - ')}",
                        f"{coverage.get('universe', {}).get('n_estimation', ' - ')} in estimation",
                    ),
                    (
                        "Periods",
                        f"{bundle['summary'].get('n_periods', ' - ')}",
                        f"{spec.get('start', '')} → {spec.get('as_of', '')}",
                    ),
                    (
                        "Runtime",
                        f"{bundle['summary'].get('runtime_seconds', ' - ')}s",
                        "background job",
                    ),
                ],
                cols={"base": 2, "sm": 4},
            ),
            *notes,
            ui.section("What the model could measure"),
            dmc.Table(
                data={
                    "head": [
                        "Style factor",
                        "Descriptors used",
                        "Missing",
                        "Orthogonalised against",
                    ],
                    "body": style_rows,
                },
                striped=True,
                withTableBorder=True,
                fz="sm",
                verticalSpacing=5,
            )
            if style_rows
            else dmc.Text("Not available", size="sm", c=TEXT_MUTED),
            ui.explain(
                "What to do about a dropped factor",
                "A dropped style means there is no usable input for it: almost "
                "always a company fundamental that has not been loaded yet. That "
                "is a data question, not a model one. Once the input is available, "
                "the same settings pick the factor up on the next run "
                "automatically, and it stops appearing in this list.",
            ),
            _table(cov_df.round(3), id="fs-coverage-table", page_size=15)
            if not cov_df.empty
            else dmc.Text("No descriptor coverage recorded.", size="sm", c=TEXT_MUTED),
            ui.section("Run configuration"),
            _kv_table(spec),
            ui.section("Universe"),
            _kv_table(coverage.get("universe") or {}),
        ],
        gap="sm",
    )
