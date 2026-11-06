"""Catalyst Watch: when each position next reports, what analysts expect against
what we paid, how far they agree, and which way their estimates are moving.

Plain tables from the nightly `consensus` snapshot and the trade log, with each
position's regime alongside. Report dates are LSEG's expected dates, not confirmed.
"""
import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import Input, Output, callback, dcc, html

from capital.analytics import catalysts as cat
from capital.analytics.regime import STATE_COLORS, positions
from capital.analytics.regime.champion import VOL_COLORS
from capital.dashboard import components as ui
from capital.dashboard import panel
from capital.data import loaders
from capital.theme import NAVY, TEXT_MUTED

dash.register_page(
    __name__, path="/catalyst-watch", name="Catalyst Watch", order=0.8,
    description="Next earnings dates, upside from our buy price, and how consistent and how "
                "revised analyst estimates are, for every position held.",
)

_SOON_DAYS = 14         # a report this close is highlighted
_PASSED_DAYS = 14       # how long a passed date stays listed, until LSEG publishes the next
_HIGHLIGHT = "#B8962E"

_TEXT = f"""
LSEG's analyst consensus for the positions held, refreshed by the nightly ingest.
Cash is left out, and the gold and copper ETCs have no analyst coverage.

**Report dates** are LSEG's *expected* dates, not confirmed ones, and they can
move. A date that has passed means the company should have reported and LSEG has
not published the next date yet.

**Upside** is the mean price target over the close from the same LSEG pull, and
over our **average buy price**: FIFO on commission-inclusive prices, as IBKR
reports cost basis. Both are left empty when the currencies differ: a depositary
receipt such as HCCD carries the consensus of its underlying (CATL, in CNY). Mean
targets lag a sharp fall in the price, so large upside is often stale targets.

**Agreement.** Three measures, each graded consistent, mixed or contradictory:

| | Consistent | Contradictory |
|---|---|---|
| Target spread (std ÷ mean) | below {cat.TARGET_SPREAD[0]:.0%} | above {cat.TARGET_SPREAD[1]:.0%} |
| EPS spread (std ÷ mean) | below {cat.EPS_SPREAD[0]:.0%} | above {cat.EPS_SPREAD[1]:.0%} |
| Hold or worse, share of analysts | below {cat.NOT_BUY_SHARE[0]:.0%} | above {cat.NOT_BUY_SHARE[1]:.0%} |

The verdict is their average: one mixed measure is still Consistent, two make it
Mixed. These cut-offs are judgement calls, not calibrated. Fewer than
{cat.MIN_ANALYSTS} analysts gets no verdict here or under Revisions.
**Price vs targets** is how many standard deviations of the targets the price sits
from the mean target.

**Revisions** compare today's mean with 30 and 90 days ago. **Rising** or **Falling**
means target and EPS moved the same way over 90 days; changes inside
±{cat.FLAT_REVISION:.0%} count as flat. EPS is empty when it went through zero or the
fiscal year rolled. The rating runs from 1 (Strong Buy) to 5 (Strong Sell), so a
lower number is an upgrade.
"""


def _money(x, ccy=None, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:,.{digits}f}" + (f" {ccy}" if isinstance(ccy, str) else "")


def _pct(x, digits: int = 0) -> str:
    return f"{x:+.{digits}%}" if x is not None and np.isfinite(x) else "–"


def _signed(x, digits: int = 0):
    """A signed percentage, coloured by its sign."""
    if x is None or not np.isfinite(x):
        return "–"
    return _pct(x, digits), panel.ALERT_COLOR["green"] if x > 0 else panel.ALERT_COLOR["red"]


def _count(x) -> str:
    return f"{x:.0f}" if x is not None and np.isfinite(x) else "–"


def _when(days: int) -> str:
    return "today" if days == 0 else f"in {days}d" if days > 0 else f"{-days}d ago"


def _table(heads: list[tuple[str, bool]], rows: list[dmc.TableTr]) -> dmc.TableScrollContainer:
    """`heads` is (label, right-aligned)."""
    head = dmc.TableThead(dmc.TableTr([
        dmc.TableTh(label, style={"fontSize": 10, "padding": "3px 8px", "whiteSpace": "nowrap",
                                  "textAlign": "right" if right else "left"}) for label, right in heads]))
    return dmc.TableScrollContainer(
        dmc.Table([head, dmc.TableTbody(rows)], striped=True, highlightOnHover=True, withRowBorders=False),
        minWidth=640)


def _row(cells: list, heads: list[tuple[str, bool]], highlight: bool = False) -> dmc.TableTr:
    """A cell is text, or (text, colour). The first is the position, in bold."""
    tds = []
    for i, (cell, (_, right)) in enumerate(zip(cells, heads)):
        text, color = cell if isinstance(cell, tuple) else (cell, None)
        tds.append(dmc.TableTd(text, style={
            "fontSize": 12, "padding": "3px 8px", "whiteSpace": "nowrap", "color": color,
            "fontWeight": 700 if i == 0 else 600 if color else None, "textAlign": "right" if right else "left"}))
    return dmc.TableTr(tds, style={"background": f"{_HIGHLIGHT}22"} if highlight else None)


def _regimes() -> dict[str, tuple[str | None, str | None]]:
    return {h["symbol"]: (h["cur"]["direction"], h["cur"]["volatility"]) if h["cur"] else (None, None)
            for h in positions.position_book()}


def _regime_cells(symbol: str, regimes: dict) -> list:
    d, v = regimes.get(symbol, (None, None))
    return [(d, STATE_COLORS[d]) if d else "–", (v, VOL_COLORS[v]) if v else "–"]


def _calendar(df: pd.DataFrame, regimes: dict) -> dmc.Paper:
    heads = [("Position", False), ("% NAV", True), ("Expected", False), ("When", False),
             ("EPS next Q", True), ("Quarter end", False), ("Direction", False), ("Volatility", False)]
    dated = df[df["days_to_report"].notna() & (df["days_to_report"] >= -_PASSED_DAYS)]
    dated = dated.sort_values("expected_report_date")
    rows = [_row([r.symbol, f"{r.pct_nav:.1f}%", f"{r.expected_report_date:%a %d %b %Y}",
                  _when(int(r.days_to_report)) if r.days_to_report >= 0
                  else (f"passed {-int(r.days_to_report)}d ago", TEXT_MUTED),
                  _money(r.eps_next_q, r.eps_ccy) if r.days_to_report >= 0 else "–",
                  f"{r.eps_next_q_end:%b %Y}" if r.days_to_report >= 0 and pd.notna(r.eps_next_q_end) else "–",
                  *_regime_cells(r.symbol, regimes)],
                 heads, highlight=0 <= r.days_to_report <= _SOON_DAYS)
            for r in dated.itertuples()]
    body = _table(heads, rows) if rows else ui.alert("No report dates in the snapshot.", "yellow")
    return panel.box("Earnings calendar", body, f"expected dates · highlighted: within {_SOON_DAYS} days", _TEXT)


def _covered(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["price_target"].notna()]


def _upside(df: pd.DataFrame) -> dmc.Paper:
    heads = [("Position", False), ("% NAV", True), ("Rating", False), ("Avg buy price", True), ("Price", True),
             ("P&L", True), ("Price target", True), ("Upside from price", True), ("Upside from buy", True)]
    rows = [_row([
        r.symbol, f"{r.pct_nav:.1f}%", r.rec_label.title() if isinstance(r.rec_label, str) else "–",
        _money(r.avg_cost, r.cost_ccy), _money(r.price, r.price_ccy), _signed(r.pnl),
        _money(r.price_target, r.target_ccy), _signed(r.upside), _signed(r.upside_cost),
    ], heads) for r in df.itertuples()]
    mixed = _covered(df)[lambda d: d["upside"].isna()]["symbol"].tolist()
    uncovered = df[df["price_target"].isna()]["symbol"].tolist()
    notes = ([f"No upside for {', '.join(mixed)}: target and price are in different currencies."] if mixed else []) \
        + ([f"No analyst coverage: {', '.join(uncovered)}."] if uncovered else [])
    body = dmc.Stack([_table(heads, rows), *[dmc.Text(n, size="10px", c=TEXT_MUTED) for n in notes]], gap=2)
    return panel.box("Upside", body, "mean price target against the price and our average buy price")


_AGREEMENT_COLOR = {"Consistent": panel.ALERT_COLOR["green"], "Mixed": _HIGHLIGHT,
                    "Contradictory": panel.ALERT_COLOR["red"], "Too few analysts": TEXT_MUTED}
_TREND_COLOR = {"Rising": panel.ALERT_COLOR["green"], "Falling": panel.ALERT_COLOR["red"],
                "Mixed": _HIGHLIGHT, "Flat": TEXT_MUTED, "Too few analysts": TEXT_MUTED}


def _label(text, colors: dict):
    return (text, colors[text]) if isinstance(text, str) else "–"


def _agreement(df: pd.DataFrame) -> dmc.Paper:
    heads = [("Position", False), ("Analysts", True), ("Buy / Hold / Sell", True), ("Target range", True),
             ("Target spread", True), ("EPS spread", True), ("Price vs targets", True), ("Agreement", False)]
    rows = [_row([
        r.symbol, _count(r.n_analysts),
        f"{r.n_strong_buy + r.n_buy:.0f} / {r.n_hold:.0f} / {r.n_sell + r.n_strong_sell:.0f}",
        f"{_money(r.target_low, digits=2 if r.target_high < 100 else 0)} – "
        f"{_money(r.target_high, digits=2 if r.target_high < 100 else 0)}",
        f"{r.target_spread:.0%}", f"{r.eps_spread:.0%}", f"{r.price_z:+.1f}σ" if np.isfinite(r.price_z) else "–",
        _label(r.agreement, _AGREEMENT_COLOR),
    ], heads) for r in _covered(df).itertuples()]
    d = _covered(df)
    below = d[(d["target_ccy"] == d["price_ccy"]) & (d["price"] < d["target_low"])]["symbol"].tolist()
    note = dmc.Text(f"Price is below every analyst's target: {', '.join(below)}. After a sharp fall this usually "
                    "means targets have not caught up yet.", size="10px", c=TEXT_MUTED) if below else None
    body = dmc.Stack([x for x in (_table(heads, rows), note) if x is not None], gap=2)
    return panel.box("Analyst agreement", body, "how consistent the estimates are")


def _revisions(df: pd.DataFrame) -> dmc.Paper:
    heads = [("Position", False), ("EPS FY", True), ("EPS 30d", True), ("EPS 90d", True), ("Target 30d", True),
             ("Target 90d", True), ("Rating now", True), ("90d ago", True), ("Trend", False)]
    rows = [_row([
        r.symbol, _money(r.eps_fy, r.eps_ccy), _signed(r.eps_rev_30d, 1), _signed(r.eps_rev_90d, 1),
        _signed(r.target_rev_30d, 1), _signed(r.target_rev_90d, 1),
        f"{r.rec_mean:.2f}", f"{r.rec_mean_90d:.2f}", _label(r.trend, _TREND_COLOR),
    ], heads) for r in _covered(df).itertuples()]
    return panel.box("Estimate revisions", _table(heads, rows),
                     "change in the mean estimate · rating: 1 Strong Buy to 5 Strong Sell")


def layout():
    return dmc.Stack([
        dmc.Group([dmc.Title("Catalyst Watch", order=3, c=NAVY), panel.info(_TEXT)], gap=6),
        dcc.Loading(html.Div(id="cw-body"), delay_show=300),
        # Fires the callback once on page load; nothing else drives it.
        dcc.Store(id="cw-init", data=0),
    ], gap=4)


@callback(Output("cw-body", "children"), Input("cw-init", "data"))
def _body(_):
    try:
        consensus = loaders.get_consensus()
        if consensus.empty:
            return ui.alert("No consensus data yet. It arrives with the next nightly ingest, or run "
                            "`capital-ingest consensus`.", "yellow")
        today = pd.Timestamp.today().normalize()
        df = cat.catalyst_table(loaders.get_held_positions(), consensus, cat.fifo_cost(loaders.get_trade_log()), today)
        as_of = consensus["date"].max()
        badge = dmc.Badge(f"Consensus {as_of:%d %b %Y}", variant="dot", size="sm",
                          color="orange" if (today - as_of).days > 4 else "teal")
        return dmc.Stack([
            dmc.Group(badge, justify="flex-end"),
            _calendar(df, _regimes()),
            _upside(df),
            _agreement(df),
            _revisions(df),
        ], gap=6)
    except Exception:  # noqa: BLE001
        return panel.fail("Catalyst Watch")
