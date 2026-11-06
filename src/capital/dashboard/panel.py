"""Compact building blocks for dense monitoring pages (Control Panel, Position
Regimes): stat strips, titled boxes, small charts, the regime chart, the alert
list. Shared here because a page module cannot be imported by another page.
"""
import traceback

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from capital.analytics.regime import STATE_COLORS
from capital.analytics.regime.champion import VOL_COLORS
from capital.theme import GRAPH_CONFIG, NAVY, TEXT_MUTED

H = 170  # default chart height inside a box, so boxes in one row line up
ALERT_COLOR = {"red": "#C0392B", "orange": "#E67E22", "green": "#2E7D6B"}
_BADGE_COLOR = {"red": "red", "orange": "orange", "green": "teal"}
DIRECTION_BADGE = {"Bull": "teal", "Sideways": "blue", "Bear": "red"}


def stat(label: str, value: str, hint: str = "", color: str = NAVY) -> html.Div:
    """A small stat: muted label, bold value, optional hint. Much denser than
    ui.metric, so a box can carry eight of them in two short rows."""
    return html.Div([
        dmc.Text(label, size="9px", c=TEXT_MUTED, fw=600, tt="uppercase", lh=1.15),
        dmc.Text(value, size="sm", fw=700, c=color, lh=1.2),
        dmc.Text(hint, size="9px", c=TEXT_MUTED, lh=1.15) if hint else None,
    ], style={"minWidth": 0})


def stats(items: list, cols: dict | int) -> dmc.SimpleGrid:
    return dmc.SimpleGrid([stat(*i) for i in items], cols=cols, spacing=6, verticalSpacing=3)


def info(md: str) -> dmc.HoverCard:
    """ℹ icon whose hover card holds the box's methodology (markdown + maths)."""
    return dmc.HoverCard([
        dmc.HoverCardTarget(dmc.ThemeIcon("ℹ", variant="light", radius="xl", size="sm",
                                          style={"cursor": "help"})),
        dmc.HoverCardDropdown(dcc.Markdown(md, mathjax=True, className="explain-body"),
                              maw=440, fz="sm"),
    ], position="bottom-end", shadow="md", openDelay=150, withinPortal=True)


def box(title: str, body, subtitle: str = "", explain: str | None = None, right=None) -> dmc.Paper:
    """One theme = one box: a single-line header (title, subtitle, controls, ℹ) over
    the body. `subtitle` may be an empty component filled by a callback, which is
    falsy, hence the explicit comparison."""
    head = dmc.Group([
        dmc.Group([dmc.Text(title, fw=700, c=NAVY, size="xs"),
                   dmc.Text(subtitle, size="10px", c=TEXT_MUTED, truncate="end",
                            style={"minWidth": 0, "flex": 1})] if subtitle != "" else
                  [dmc.Text(title, fw=700, c=NAVY, size="xs")],
                  gap=8, wrap="nowrap", style={"minWidth": 0, "flex": 1}),
        dmc.Group([x for x in (right, info(explain) if explain else None) if x is not None],
                  gap=6, wrap="nowrap"),
    ], justify="space-between", wrap="nowrap", gap="xs")
    return dmc.Paper(dmc.Stack([head, body], gap=3), withBorder=True, radius="md", p=8, h="100%")


def fail(title: str) -> dmc.Paper:
    return box(title, dmc.Alert(dmc.Code(traceback.format_exc(), block=True),
                                color="red", variant="light", title=f"{title} failed"))


def graph(fig: go.Figure, height: int = H, legend: bool = False, left: int = 40,
          right: int = 10, bottom: int = 22) -> dcc.Graph:
    """A compact chart for a box. Size `left` with ui.axis_margin on categorical
    y axes so long labels are not clipped."""
    fig.update_layout(
        height=height, margin=dict(l=left, r=right, t=18 if legend else 4, b=bottom),
        showlegend=legend, font=dict(size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=9), borderwidth=0) if legend else None,
    )
    fig.update_xaxes(tickfont=dict(size=9), title_font=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9), title_font=dict(size=9))
    return dcc.Graph(figure=fig, config={**GRAPH_CONFIG, "displayModeBar": False},
                     style={"height": f"{height}px"})


# ── Regime chart ──────────────────────────────────────────────────────────────

def _regime_bands(fig: go.Figure, labels: pd.Series, colors: dict, row: int) -> None:
    """One full-height bar per day on a hidden 0–1 axis, coloured by that day's
    label. Bars (not shapes) so the shading is exact per day and survives the
    weekend rangebreaks without gaps."""
    for s, color in colors.items():
        on = (labels == s).astype(float).replace(0.0, np.nan)
        fig.add_trace(go.Bar(x=labels.index, y=on, name=s, marker_color=color, opacity=0.28,
                             hoverinfo="skip", width=86_400_000, showlegend=row == 1),
                      row=row, col=1, secondary_y=False)


def _level(fig: go.Figure, index: pd.Index, y, color: str, row: int, **kw) -> None:
    """A dotted threshold, constant or moving: the same call draws SPY's fixed
    lines and a stock's own."""
    fig.add_trace(go.Scatter(x=index, y=pd.Series(y, index=index), mode="lines", showlegend=False,
                             hoverinfo="skip", line=dict(color=color, width=1, dash="dot")),
                  row=row, col=1, **kw)


def regime_figure(reg: pd.DataFrame, px: pd.Series, name: str, bull, bear, calm=None, turbulent=None,
                  tickprefix: str = "") -> go.Figure:
    """Row 1: price over direction shading. Row 2: the trend slope against the Bull
    / Bear lines. Row 3: the vol nowcast over volatility shading, with the Calm and
    Turbulent lines if given. A line is a number or a Series on `reg`'s index."""
    state, vol_state = reg["direction_name"], reg["volatility_name"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.56, 0.22, 0.22],
                        vertical_spacing=0.03, specs=[[{"secondary_y": True}], [{}], [{"secondary_y": True}]])
    _regime_bands(fig, state, STATE_COLORS, 1)
    fig.add_trace(go.Scatter(
        x=px.index, y=px, name=name, line=dict(color=NAVY, width=1.8), showlegend=False,
        customdata=np.column_stack([state.fillna("–"), vol_state.fillna("–")]),
        hovertemplate="%{x|%d %b %Y}  " + name + " %{y:,.2f}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"),
        row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=reg.index, y=reg["slope"], name="Trend slope", showlegend=False,
        line=dict(color=NAVY, width=1.4),
        hovertemplate="%{x|%d %b %Y}  slope %{y:+.0%}/yr<extra></extra>"), row=2, col=1)
    _level(fig, reg.index, bull, STATE_COLORS["Bull"], 2)
    _level(fig, reg.index, bear, STATE_COLORS["Bear"], 2)

    _regime_bands(fig, vol_state, VOL_COLORS, 3)
    fig.add_trace(go.Scatter(
        x=reg.index, y=reg["vol_nowcast"], name="Vol nowcast", showlegend=False,
        line=dict(color=NAVY, width=1.4),
        hovertemplate="%{x|%d %b %Y}  vol %{y:.0%}<extra></extra>"), row=3, col=1, secondary_y=True)
    if calm is not None:
        _level(fig, reg.index, calm, VOL_COLORS["Calm"], 3, secondary_y=True)
    if turbulent is not None:
        _level(fig, reg.index, turbulent, VOL_COLORS["Turbulent"], 3, secondary_y=True)

    fig.update_layout(barmode="overlay", bargap=0, legend=dict(traceorder="normal"))
    fig.update_yaxes(range=[0, 1], visible=False, row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 1], visible=False, row=3, col=1, secondary_y=False)
    lo, hi = float(px.min()), float(px.max())
    pad = (hi - lo) * 0.05
    fig.update_yaxes(range=[lo - pad, hi + pad], side="left", showgrid=True, tickprefix=tickprefix,
                     row=1, col=1, secondary_y=True)
    fig.update_yaxes(tickformat="+.0%", showgrid=False, title_text="slope", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", showgrid=False, side="left", title_text="vol",
                     row=3, col=1, secondary_y=True)
    # Skip weekends and market holidays (weekdays with no close), so the shading
    # runs unbroken from one trading day to the next.
    holidays = pd.bdate_range(reg.index[0], reg.index[-1]).difference(reg.index)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                  dict(values=[d.strftime("%Y-%m-%d") for d in holidays])],
                     showgrid=False, domain=[0, 1])
    return fig


# ── Alert list ────────────────────────────────────────────────────────────────

def _age(bd: int) -> str:
    return "today" if bd == 0 else f"{bd}d ago"


def alert_badges(alerts: list[dict]) -> dmc.Group:
    """How many alerts at each level, for a box header."""
    counts = {lvl: sum(a["level"] == lvl for a in alerts) for lvl in ALERT_COLOR}
    return dmc.Group([dmc.Badge(str(n), color=_BADGE_COLOR[lvl], variant="filled", size="sm", circle=True)
                      for lvl, n in counts.items() if n], gap=3)


def alert_list(alerts: list[dict], height: int = 245) -> dmc.Center | dmc.ScrollArea:
    """The ranked alerts, or an all-clear when there are none."""
    if not alerts:
        return dmc.Center(dmc.Stack([dmc.Text("All clear", fw=700, c=ALERT_COLOR["green"]),
                                     dmc.Text("Every rule is inside its limit.", size="xs", c=TEXT_MUTED)],
                                    gap=0, align="center"), h=150)
    rows = [html.Div([
        html.Span(style={"width": 4, "alignSelf": "stretch", "borderRadius": 2, "flex": "0 0 4px",
                         "background": ALERT_COLOR[a["level"]]}),
        html.Div([
            dmc.Group([dmc.Text(a["title"], size="xs", fw=700, c=NAVY, lh=1.25, truncate="end",
                                style={"flex": 1, "minWidth": 0}),
                       dmc.Text(_age(a["age_bd"]), size="10px", c=TEXT_MUTED, style={"flex": "0 0 auto"})],
                      justify="space-between", gap=4, wrap="nowrap"),
            dmc.Text(a["detail"], size="11px", c=TEXT_MUTED, lh=1.25),
        ], style={"flex": 1, "minWidth": 0}),
    ], style={"display": "flex", "gap": 7, "padding": "4px 0", "borderBottom": "1px solid #EEF2F7"})
        for a in alerts]
    return dmc.ScrollArea(html.Div(rows), h=height, type="auto", offsetScrollbars=True)
