"""Position Regimes: the champion regime model fitted on every position held, three
to a row, under two tables that sort the book by direction and by volatility and
highlight recent transitions.

The holding alerts themselves live in the Control Panel's alert box. Everything
reads `positions.position_book`, cached on the data version: the first load after
an ingest fits every stock (a few seconds), and later loads are cache hits.
"""
import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, callback, dcc, html

from capital.analytics import control_panel as cp
from capital.analytics.regime import STATE_COLORS, STATES
from capital.analytics.regime import champion as ch
from capital.analytics.regime import positions
from capital.dashboard import components as ui
from capital.dashboard import panel
from capital.theme import NAVY, TEXT_MUTED

dash.register_page(
    __name__, path="/position-regimes", name="Position Regimes", order=0.7,
    description="The champion regime model on every position held, in each stock's own volatility "
                "lines, sorted by direction and volatility with recent transitions highlighted.",
)

_C = ch.CHAMPION
_DAYS = cp.LONG  # each chart's lookback; the model runs on the stock's full history
_CARD_SPAN = {"base": 12, "sm": 6, "lg": 4}
_RECENT_DAYS = 10  # a transition within two weeks of trading days is highlighted
_DIRECTIONS = {s: STATE_COLORS[s] for s in ("Bull", "Sideways", "Bear")}

_TEXT = rf"""
The champion regime model, fitted on each position separately: its own Kalman
trend slope and its own yearly GJR-GARCH refits, on the stock's adjusted close in
its trading currency. Cash is left out.

Every line is a **multiple of the stock's own long-run volatility**, because the
champion's lines are sized for SPY. The multiples are its lines over SPY's
long-run vol of {positions.REFERENCE_VOL:.0%}:

| | SPY | Multiple |
|---|---|---|
| Bull above | {_C.bull_slope:+.0%}/yr | {_C.bull_slope / positions.REFERENCE_VOL:+.2f}× |
| Bear below | {_C.bear_slope:+.0%}/yr | {_C.bear_slope / positions.REFERENCE_VOL:+.2f}× |
| Calm below | {_C.calm_below:.0%} | {_C.calm_below / positions.REFERENCE_VOL:.2f}× |
| Turbulent above | {_C.turbulent_above:.0%} | {_C.turbulent_above / positions.REFERENCE_VOL:.2f}× |

Long-run vol uses only data up to each day, so every band is what you could have
known then, and the dotted lines move slowly with it. Calm is rarer on a stock
than on SPY, because long-run vol includes earnings jumps.

**Short histories.** Direction needs {_C.burn_in} days of prices. GARCH needs
{_C.train_years} calendar years; until then the nowcast is realised vol alone and
the card says "realised" in place of "lines" under it (Calm / Turbulent lines).

**Tables.** Each position sits under its current state, with the trading days it
has been there. A highlighted position changed state within the last
{_RECENT_DAYS} trading days and shows where it came from. Alerts on these regimes
appear on the **Control Panel**.

Compare models on the **Market Regime** page, which runs on SPY and the ETFs.
"""

def layout():
    return dmc.Stack([
        dmc.Group([
            dmc.Title("Position Regimes", order=3, c=NAVY),
            panel.info(_TEXT),
        ], gap=6),
        dcc.Loading(html.Div(id="pr-body"), delay_show=300),
        # Fires the callback once on page load; nothing else drives it.
        dcc.Store(id="pr-init", data=0),
    ], gap=4)


def _previous(labels: pd.Series) -> str | None:
    """The state before the current one, or None if the stock has only had one."""
    v = labels.dropna()
    before = v[v != v.iloc[-1]]
    return before.iloc[-1] if len(before) else None


def _came_from(h: dict, axis: str, days_key: str) -> str | None:
    """The previous state if the position changed state recently, else None."""
    return _previous(h["reg"][f"{axis}_name"]) if h["cur"][days_key] <= _RECENT_DAYS else None


def _entry(h: dict, axis: str, days_key: str, color: str) -> html.Div:
    """One position in a state table; highlighted if it arrived there recently."""
    came_from = _came_from(h, axis, days_key)
    line = dmc.Group([
        dmc.Group([dmc.Text(h["symbol"], size="xs", fw=700, c=NAVY),
                   dmc.Text(f"{h['weight']:.1%}", size="10px", c=TEXT_MUTED)], gap=5, wrap="nowrap"),
        dmc.Text(f"{h['cur'][days_key]}d", size="10px", c=NAVY if came_from else TEXT_MUTED,
                 fw=700 if came_from else 400),
    ], justify="space-between", gap=4, wrap="nowrap")
    if came_from is None:
        return html.Div(line, style={"padding": "3px 6px"})
    return html.Div([line, dmc.Text(f"from {came_from}", size="10px", c=color, fw=600, lh=1.2)],
                    style={"padding": "3px 6px", "borderRadius": 4, "background": f"{color}22",
                           "borderLeft": f"3px solid {color}"})


def _state_table(book: list[dict], title: str, axis: str, days_key: str, colors: dict) -> dmc.Paper:
    """One column per state, each listing its positions: recent arrivals first,
    then by weight (the book's order)."""
    held = [h for h in book if h["cur"] and h["cur"][axis]]
    heads, columns = [], []
    for state, color in colors.items():
        here = sorted((h for h in held if h["cur"][axis] == state),
                      key=lambda h: _came_from(h, axis, days_key) is None)
        nav = sum(h["weight"] for h in here)
        heads.append(dmc.TableTh(dmc.Group([dmc.Text(state, size="sm", fw=700, c=color),
                                            dmc.Text(f"{nav:.0%} NAV", size="10px", c=TEXT_MUTED)],
                                           justify="space-between", gap=4, wrap="nowrap"),
                                 style={"padding": "4px 6px", "borderBottom": f"2px solid {color}"}))
        columns.append(dmc.TableTd(dmc.Stack([_entry(h, axis, days_key, color) for h in here], gap=3)
                                   if here else dmc.Text("–", size="xs", c=TEXT_MUTED, px=6),
                                   style={"verticalAlign": "top", "padding": "4px 3px"}))
    table = dmc.Table([dmc.TableThead(dmc.TableTr(heads)), dmc.TableTbody(dmc.TableTr(columns))],
                      withRowBorders=False, style={"tableLayout": "fixed"})
    waiting = [h["symbol"] for h in book if not (h["cur"] and h["cur"][axis])]
    note = dmc.Text(f"Not classified yet: {', '.join(waiting)}", size="10px", c=TEXT_MUTED) if waiting else None
    return panel.box(title, dmc.Stack([x for x in (table, note) if x is not None], gap=2),
                     f"highlighted: changed in the last {_RECENT_DAYS} trading days")


def _key() -> dmc.Group:
    def swatch(label, color):
        return dmc.Group([html.Span(style={"width": 10, "height": 10, "borderRadius": 2, "opacity": 0.45,
                                           "background": color}),
                          dmc.Text(label, size="10px", c=TEXT_MUTED)], gap=4)

    return dmc.Group([
        dmc.Text("Price panel", size="10px", fw=600, c=TEXT_MUTED),
        *[swatch(s, STATE_COLORS[s]) for s in STATES],
        dmc.Text("Vol panel", size="10px", fw=600, c=TEXT_MUTED, ml="sm"),
        *[swatch(s, c) for s, c in ch.VOL_COLORS.items()],
        dmc.Text("Dotted lines: the stock's own thresholds", size="10px", c=TEXT_MUTED, ml="sm"),
    ], gap=8)


def _card(h: dict) -> dmc.Paper:
    title = h["symbol"]
    subtitle = f"{h['weight']:.1%} NAV · {h['theme']} · {h['ccy']} · {h['name']}"
    cur = h["cur"]
    if cur is None:
        return panel.box(title, ui.alert(h["note"], "yellow"), subtitle)

    d, v = cur["direction"], cur["volatility"]
    reg = h["reg"].tail(_DAYS)
    px = h["px"].reindex(reg.index).ffill()
    fig = panel.regime_figure(reg, px, h["symbol"], reg["bull_line"], reg["bear_line"],
                              reg["calm_line"], reg["turbulent_line"])
    fig.update_yaxes(nticks=3, row=2, col=1)
    fig.update_yaxes(nticks=3, row=3, col=1, secondary_y=True)
    if d is None:
        direction = ("Direction", "–", f"{len(h['px'])} of {_C.burn_in} days")
        slope = ("Trend slope", "–", "")
    else:
        direction = ("Direction", d, f"{cur['days_direction']}d in regime", STATE_COLORS[d])
        slope = ("Trend slope", f"{cur['slope']:+.0%} / yr", f"{cur['flip_distance']:+.0%} to {cur['nearest_flip']}")
    stats = panel.stats([
        direction, slope,
        ("Volatility", v, f"{cur['days_volatility']}d in regime", ch.VOL_COLORS[v]),
        ("Vol nowcast", f"{cur['vol_nowcast']:.0%}",
         f"{'lines' if cur['garch'] else 'realised'} {cur['calm_line']:.0%} / {cur['turbulent_line']:.0%}"),
    ], cols=4)
    badge = dmc.Badge(f"{d} · {v}" if d else v, variant="filled", size="sm",
                      color=panel.DIRECTION_BADGE.get(d, "gray"))
    return panel.box(title, dmc.Stack([stats, panel.graph(fig, 300, left=46, right=8)], gap=2),
                     subtitle, right=badge)


def _prices_badge(book: list[dict], today: pd.Timestamp) -> dmc.Badge | None:
    dates = [h["cur"]["date"] for h in book if h["cur"]]
    if not dates:
        return None
    oldest = min(dates)
    lag = max(0, len(pd.bdate_range(oldest, today)) - 1)
    return dmc.Badge(f"Prices {oldest:%d %b %Y}", variant="dot", size="sm",
                     color="orange" if lag > cp.LIMITS["stale_bdays"] else "teal")


@callback(Output("pr-body", "children"), Input("pr-init", "data"))
def _body(_):
    try:
        today = pd.Timestamp.today().normalize()
        book = positions.position_book()
        if not book:
            return ui.alert("No positions in the latest weightings.", "yellow")
        cards = []
        for h in book:
            try:
                card = _card(h)
            except Exception:  # noqa: BLE001
                card = panel.fail(h["symbol"])
            cards.append(dmc.GridCol(card, span=_CARD_SPAN))
        return dmc.Stack([
            dmc.Grid([
                dmc.GridCol(_state_table(book, "Direction", "direction", "days_direction", _DIRECTIONS),
                            span={"base": 12, "lg": 6}),
                dmc.GridCol(_state_table(book, "Volatility", "volatility", "days_volatility", ch.VOL_COLORS),
                            span={"base": 12, "lg": 6}),
            ], gutter=6, align="stretch"),
            dmc.Group([_key(), _prices_badge(book, today)], justify="space-between", wrap="wrap", gap="xs"),
            dmc.Grid(cards, gutter=6, align="stretch"),
        ], gap=6)
    except Exception:  # noqa: BLE001
        return panel.fail("Position Regimes")
