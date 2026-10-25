"""
Champion market regime model: Kalman trend slope x GJR-GARCH volatility nowcast.

Classifies every trading day on two axes, using only prices up to that day's
close, so the whole history is point-in-time by construction:

Direction (Bull / Sideways / Bear)
    A local-linear-trend Kalman filter on log price estimates the current drift
    (slope). Annualised: Bull above +10%/yr, Bear below -20%/yr, Sideways in
    between. A state changes only once the slope is 2.5%/yr past a line
    (hysteresis), so the label does not flicker around a threshold.

Volatility (Calm / Choppy / Turbulent)
    GJR-GARCH(1,1) with Student-t errors, refitted every January on all returns
    before that January. Nowcast = realised variance of the last 11 days blended
    with the GARCH forecast of the average variance over the next 10 days
    (weights 11/21 and 10/21), annualised. Calm below 12%, Choppy 12-22%,
    Turbulent above 22%.

Scored against a hand-labelled ground truth: direction macro recall 0.59 (SPY) /
0.58 (STOXX 600), volatility 0.81 / 0.82 (chance 0.33).

Every threshold lives on `ChampionConfig`; the defaults are the champion. The
Market Regime page varies them to compare against other configurations.
"""
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DIRECTION = {0: "Bear", 1: "Sideways", 2: "Bull"}
VOLATILITY = {0: "Calm", 1: "Choppy", 2: "Turbulent"}
VOL_COLORS = {"Calm": "#2E7D6B", "Choppy": "#B8962E", "Turbulent": "#8B2E4A"}


@dataclass(frozen=True)
class ChampionConfig:
    # direction
    kalman_q: float = 1e-5          # slope-noise / observation-noise variance ratio: higher = faster slope
    bull_slope: float = 0.10        # annualised drift above this = Bull
    bear_slope: float = -0.20       # annualised drift below this = Bear
    band: float = 0.025             # hysteresis band around both lines
    burn_in: int = 252              # days before the slope is trusted
    # volatility
    calm_below: float = 0.12        # Calm < 12% <= Choppy <= 22% < Turbulent
    turbulent_above: float = 0.22
    past_days: int = 11             # realised part of the nowcast: days t-10 .. t
    ahead_days: int = 10            # GARCH part: average variance forecast over t+1 .. t+10
    train_years: int = 3            # first GARCH forecast after this many years of data

    def to_dict(self) -> dict:
        return asdict(self)


CHAMPION = ChampionConfig()


# ── Direction: Kalman trend slope + hysteresis ────────────────────────────────

def kalman_slope(px: pd.Series, q: float = CHAMPION.kalman_q,
                 burn_in: int = CHAMPION.burn_in) -> pd.Series:
    """Filtered slope of log price, annualised. Model:

        y_t = level_t + e_t                      (var 1)
        level_t = level_{t-1} + slope_{t-1}
        slope_t = slope_{t-1} + z_t              (var q)
    """
    y = np.log(px.to_numpy(dtype=float))
    slope = np.empty(len(y))
    lvl, b = y[0], 0.0
    P00, P01, P11 = 1.0, 0.0, 1e-4
    for t in range(len(y)):
        # predict
        lvl, b = lvl + b, b
        n00 = P00 + 2 * P01 + P11
        n01 = P01 + P11
        n11 = P11 + q
        # update with today's log price
        S = n00 + 1.0
        k0, k1 = n00 / S, n01 / S
        v = y[t] - lvl
        lvl += k0 * v
        b += k1 * v
        P00 = (1 - k0) * n00
        P01 = (1 - k0) * n01
        P11 = n11 - k1 * n01
        slope[t] = b
    out = pd.Series(slope * 252, index=px.index)
    out.iloc[:burn_in] = np.nan
    return out


def direction(slope: pd.Series, bull: float = CHAMPION.bull_slope, bear: float = CHAMPION.bear_slope,
              band: float = CHAMPION.band) -> pd.Series:
    """0 Bear / 1 Sideways / 2 Bull. Enter a state past its line + band, leave it past its line - band."""
    out = np.full(len(slope), np.nan)
    s = 1
    for i, x in enumerate(slope.to_numpy()):
        if np.isnan(x):
            continue
        if s == 2:
            s = 0 if x < bear - band else 1 if x < bull - band else 2
        elif s == 0:
            s = 2 if x > bull + band else 1 if x > bear + band else 0
        else:
            s = 2 if x > bull + band else 0 if x < bear - band else 1
        out[i] = s
    return pd.Series(out, index=slope.index)


# ── Volatility: GJR-GARCH(1,1)-t nowcast ──────────────────────────────────────

def _gjr_next_variance(x, mu, omega, alpha, gamma, beta, s2_0) -> np.ndarray:
    """Variance of x[t+1] given data up to t, for every t."""
    out = np.empty(len(x))
    prev = s2_0
    for t in range(len(x)):
        e = x[t] - mu
        prev = omega + (alpha + (gamma if e < 0 else 0.0)) * e * e + beta * prev
        out[t] = prev
    return out


def garch_ahead(px: pd.Series, ahead_days: int = CHAMPION.ahead_days,
                train_years: int = CHAMPION.train_years, first_year: int | None = None) -> pd.Series:
    """GARCH forecast of the average daily variance over the next `ahead_days`
    (returns in %), known at close t. Refitted every January from `first_year`
    on, on all returns before that January."""
    from arch import arch_model

    x = (np.log(px).diff() * 100).dropna()
    years = x.index.year
    first_year = first_year or int(years.min()) + train_years
    out = pd.Series(np.nan, index=x.index)
    h = np.arange(1, ahead_days + 1)
    for y in sorted(set(years[years >= first_year])):
        nb = int((years < y).sum())
        pos = np.flatnonzero(years == y)
        tr = x.iloc[:nb]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = arch_model(tr, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off").params
        mu, om, a, g, b = p["mu"], p["omega"], p["alpha[1]"], p["gamma[1]"], p["beta[1]"]
        pers = a + g / 2 + b
        lr = om / (1 - pers) if pers < 0.999 else tr.var()          # long-run variance
        v1 = _gjr_next_variance(x.to_numpy()[:pos[-1] + 1], mu, om, a, g, b, tr.var())[pos]
        # E[var t+h] decays from the 1-day forecast to the long-run variance at rate `pers`
        out.iloc[pos] = (lr + (v1[:, None] - lr) * (pers ** (h - 1))[None, :]).mean(1)
    return out.reindex(px.index)


def vol_nowcast(px: pd.Series, past_days: int = CHAMPION.past_days, ahead_days: int = CHAMPION.ahead_days,
                train_years: int = CHAMPION.train_years, first_year: int | None = None) -> pd.Series:
    """Annualised volatility over t-(past-1) .. t+ahead: realised for the past,
    GARCH forecast for the future."""
    past = ((np.log(px).diff() * 100) ** 2).rolling(past_days).mean()
    ahead = garch_ahead(px, ahead_days, train_years, first_year)
    var = (past_days * past + ahead_days * ahead) / (past_days + ahead_days)
    return np.sqrt(var * 252) / 100


def volatility(vol: pd.Series, cuts: tuple[float, float] = (CHAMPION.calm_below, CHAMPION.turbulent_above)
               ) -> pd.Series:
    """0 Calm / 1 Choppy / 2 Turbulent."""
    return pd.Series(np.select([vol < cuts[0], vol <= cuts[1]], [0.0, 1.0], 2.0),
                     index=vol.index).where(vol.notna())


# ── Both axes ─────────────────────────────────────────────────────────────────

def classify(px: pd.Series, config: ChampionConfig = CHAMPION, first_year: int | None = None) -> pd.DataFrame:
    """Daily regime from a series of closing prices.

    Columns: `direction` / `volatility` codes, their names (`direction_name`,
    `volatility_name`), and the two underlying readings (`slope`, annualised
    drift, and `vol_nowcast`, annualised vol). Rows before the burn-in / first
    GARCH fit are NaN.
    """
    c = config
    px = px.dropna().astype(float)
    slope = kalman_slope(px, c.kalman_q, c.burn_in)
    vol = vol_nowcast(px, c.past_days, c.ahead_days, c.train_years, first_year)
    out = pd.DataFrame({
        "direction": direction(slope, c.bull_slope, c.bear_slope, c.band),
        "volatility": volatility(vol, (c.calm_below, c.turbulent_above)),
        "slope": slope, "vol_nowcast": vol,
    })
    out["direction_name"] = out["direction"].map(DIRECTION)
    out["volatility_name"] = out["volatility"].map(VOLATILITY)
    return out


def _days_in(labels: pd.Series) -> int:
    """Trading days the last label has held, counting today as day 1."""
    v = labels.to_numpy()
    changed = np.flatnonzero(v != v[-1])
    return int(len(v) - (changed[-1] + 1)) if len(changed) else len(v)


def current_state(reg: pd.DataFrame, config: ChampionConfig = CHAMPION,
                  asof: pd.Timestamp | None = None) -> dict:
    """The regime on `asof` (default: the last classified day), with how long
    each axis has held and how far the slope is from flipping direction."""
    r = reg.dropna(subset=["direction", "volatility"])
    if asof is not None:
        r = r.loc[:pd.Timestamp(asof)]
    if r.empty:
        raise ValueError("No classified days on or before the requested date.")
    last = r.iloc[-1]
    d, slope = last["direction_name"], float(last["slope"])
    c = config
    # Slope distance to the line that would change the state, hysteresis included.
    exits = {"Bull": [("Sideways", c.bull_slope - c.band)],
             "Bear": [("Sideways", c.bear_slope + c.band)],
             "Sideways": [("Bull", c.bull_slope + c.band), ("Bear", c.bear_slope - c.band)]}[d]
    to, line = min(exits, key=lambda e: abs(slope - e[1]))
    return {
        "date": r.index[-1],
        "direction": d, "volatility": last["volatility_name"],
        "slope": slope, "vol_nowcast": float(last["vol_nowcast"]),
        "days_direction": _days_in(r["direction_name"]),
        "days_volatility": _days_in(r["volatility_name"]),
        "nearest_flip": to, "flip_line": line, "flip_distance": line - slope,
    }
