"""
The champion regime model fitted on each position the fund holds.

The filters are the champion's and are fitted per stock: its own Kalman slope,
its own yearly GJR-GARCH refits. The lines cannot be reused as they stand,
because they are sized for SPY: at 12% / 22% most stocks are Turbulent on nearly
every day of their history, and a stock's slope swings with its volatility, so
+10% / -20% a year is noise on a 50%-vol name.

So every line is a multiple of the stock's own long-run vol. The multiples are
the champion's lines over SPY's long-run vol (`REFERENCE_VOL`): Bull +0.56x,
Bear -1.11x, Calm 0.67x, Turbulent 1.22x. Long-run vol is an expanding window, so
the lines on any day use only data up to that day and the labels stay
point-in-time. It is deliberately not current vol: lines that widen in a sell-off
would make Bear hardest to reach when it matters.

Compared on the holdings and a universe sample against percentiles of each
stock's own nowcast and multiples of its median nowcast, this agreed best with
the champion on SPY (88% of days), flipped least, and separated next-month
realised vol as well as either. Its cost: Calm is rarer on stocks than on SPY,
because long-run vol includes earnings jumps that a quiet nowcast does not.

Short histories: GARCH needs three calendar years, so until then the nowcast is
realised vol alone (`garch` False). Direction waits out the champion's burn-in.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from capital.analytics.control_panel import LONG, WINDOW, realised_vol
from capital.analytics.regime import champion as ch
from capital.data import loaders
from capital.data.cache import cached_by_version

REFERENCE_VOL = 0.18    # SPY's long-run annual vol: the scale the champion's lines were set on
MIN_DAYS = 126          # price history before long-run vol, and so any line, is usable


def classify_position(px: pd.Series, config: ch.ChampionConfig = ch.CHAMPION) -> pd.DataFrame:
    """Daily regime for one stock: `champion.classify`'s columns, plus the lines
    in force on each day (`bull_line`, `bear_line`, `calm_line`, `turbulent_line`)
    and `garch` (False where the nowcast is realised vol only)."""
    c = config
    px = px.dropna().astype(float)
    r = np.log(px).diff()
    scale = r.expanding(MIN_DAYS).std() * np.sqrt(252) / REFERENCE_VOL
    slope = ch.kalman_slope(px, c.kalman_q, c.burn_in)
    nowcast = ch.vol_nowcast(px, c.past_days, c.ahead_days, c.train_years)
    vol = nowcast.fillna(np.sqrt((r ** 2).rolling(c.past_days).mean() * 252))

    out = pd.DataFrame({
        "direction": ch.direction(slope / scale, c.bull_slope, c.bear_slope, c.band),
        "volatility": ch.volatility(vol / scale, (c.calm_below, c.turbulent_above)),
        "slope": slope, "vol_nowcast": vol,
        "bull_line": scale * c.bull_slope, "bear_line": scale * c.bear_slope,
        "calm_line": scale * c.calm_below, "turbulent_line": scale * c.turbulent_above,
        "garch": nowcast.notna(),
    })
    out["direction_name"] = out["direction"].map(ch.DIRECTION)
    out["volatility_name"] = out["volatility"].map(ch.VOLATILITY)
    return out


def position_state(reg: pd.DataFrame, config: ch.ChampionConfig = ch.CHAMPION) -> dict:
    """`champion.current_state` in the stock's own lines, plus the lines themselves
    and `vol_pct`, today's nowcast as a percentile of its history. `direction` is
    None until the burn-in has passed; the volatility fields are set before that."""
    r = reg.dropna(subset=["volatility"])
    last = r.iloc[-1]
    state = {
        "date": r.index[-1], "volatility": last["volatility_name"],
        "vol_nowcast": float(last["vol_nowcast"]),
        "days_volatility": ch._days_in(r["volatility_name"]),
        "vol_pct": float((r["vol_nowcast"] <= last["vol_nowcast"]).mean()),
        "garch": bool(last["garch"]),
        "bull_line": float(last["bull_line"]), "bear_line": float(last["bear_line"]),
        "calm_line": float(last["calm_line"]), "turbulent_line": float(last["turbulent_line"]),
    }
    if r["direction"].isna().all():
        return state | {"direction": None, "slope": np.nan, "days_direction": 0,
                        "nearest_flip": None, "flip_distance": np.nan}
    lines = replace(config, bull_slope=state["bull_line"], bear_slope=state["bear_line"],
                    band=config.band * state["bull_line"] / config.bull_slope)
    return ch.current_state(reg, lines) | state


@cached_by_version
def position_book() -> list[dict]:
    """Every non-cash position on the latest weightings date, largest first, each
    with its regime history and current state.

    Keys: symbol, name, theme, ccy, weight, px, reg, cur, rv_window, rv_long,
    note. `reg` and `cur` are None when the position cannot be classified, and
    `note` says why: one bad series costs that position, not the book. Yearly
    GARCH refits cost about 0.5s a stock, so this runs once per data version.
    """
    held = loaders.get_held_positions()
    prices = loaders.get_close_matrix(tuple(held["security_id"].dropna()))

    book = []
    for row in held.itertuples():
        h = {"symbol": row.symbol, "name": row.name, "theme": row.theme, "ccy": row.ccy,
             "weight": row.pct_nav / 100.0, "px": None, "reg": None, "cur": None,
             "rv_window": None, "rv_long": None, "note": None}
        book.append(h)
        if row.security_id not in prices.columns:
            h["note"] = "No price history in the store."
            continue
        px = prices[row.security_id].dropna()
        h["px"] = px
        if len(px) <= MIN_DAYS:
            h["note"] = f"Only {len(px)} days of prices; the model needs {MIN_DAYS + 1}."
            continue
        try:
            h["reg"] = classify_position(px)
            h["cur"] = position_state(h["reg"])
            ret = px.pct_change()
            h["rv_window"], h["rv_long"] = realised_vol(ret, WINDOW), realised_vol(ret, LONG)
        except Exception as exc:  # noqa: BLE001
            print(f"[positions] {row.symbol}: {exc}")
            h["reg"], h["cur"], h["note"] = None, None, f"Model failed: {exc}"
    return book
