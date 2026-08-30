"""
Raw descriptor panels.

Every function here takes whole wide matrices (index = date, columns =
security_id) and returns one, so the cost of the model is independent of the
number of securities in any meaningful sense — 2,000 names cost the same few
matrix operations as 20. Nothing in this module loops over securities.

Descriptors are *raw*: unwinsorised, unstandardised, in their natural units.
Standardisation and aggregation into style factors happen in exposures.py.
"""
import numpy as np
import pandas as pd

from capital.analytics.factors.spec import DESCRIPTORS, ModelSpec

_TRADING_DAYS = 252
_EPS = 1e-12


# ── Small numeric helpers ─────────────────────────────────────────────────────

def _ewm_pair(x: pd.DataFrame, y: pd.Series, halflife: int, min_periods: int):
    """EWMA covariance of every column of `x` with `y`, and the variance of `y`.

    Computed from EWMA first/second moments rather than DataFrame.ewm().cov(),
    which materialises a pairwise panel and is orders of magnitude slower on a
    2,000-column frame.
    """
    kw = dict(halflife=halflife, min_periods=min_periods, adjust=True)
    mx = x.ewm(**kw).mean()
    my = y.ewm(**kw).mean()
    mxy = x.mul(y, axis=0).ewm(**kw).mean()
    myy = (y * y).ewm(**kw).mean()
    cov = mxy.sub(mx.mul(my, axis=0))
    var = myy - my * my
    return cov, var, mx, my


def _effective_n(halflife: int) -> float:
    """Effective sample size of an EWMA with this half-life."""
    lam = 0.5 ** (1.0 / max(halflife, 1))
    return (1.0 + lam) / (1.0 - lam)


def _safe_log(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df.where(df > 0))


# ── Returns and the estimation-universe market ────────────────────────────────

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns, with absurd values (bad ticks, splits) removed."""
    ret = prices.sort_index().pct_change(fill_method=None)
    return ret.where(ret.abs() < 1.0)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(prices.sort_index()).diff()
    return ret.where(ret.abs() < np.log(2.0))


def market_returns(returns: pd.DataFrame, market_cap: pd.DataFrame) -> pd.Series:
    """Cap-weighted return of the estimation universe — the model's market factor.

    Weights are lagged one day so today's return is weighted by yesterday's
    known capitalisation (no look-ahead), and renormalised over the names that
    actually traded, so a missing price does not silently shrink the market.
    """
    caps = market_cap.reindex(returns.index).ffill().shift(1)
    caps = caps.where(returns.notna())
    weights = caps.div(caps.sum(axis=1).replace(0, np.nan), axis=0)
    return (returns * weights).sum(axis=1, min_count=1)


def convert_prices_to_numeraire(prices: pd.DataFrame, currency_by_sid: pd.Series,
                                fx: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Restate a local-currency price panel in the numeraire implied by `fx`.

    Converting prices rather than returns means every price-derived descriptor —
    momentum, volatility, beta, reversal — inherits the conversion for free and
    consistently. Ratio descriptors (turnover, valuation) are left alone: they
    are currency-free as long as numerator and denominator share a currency,
    which is why `close` and `market_cap` stay in local units.

    Returns the input unchanged plus `False` when the FX panel does not cover
    every currency in play, so the caller can record "estimated in local
    currency" in the coverage report instead of half-converting the universe.
    """
    if fx.empty or prices.empty:
        return prices, False
    ccy = currency_by_sid.reindex(prices.columns)
    needed = set(ccy.dropna().unique())
    if not needed or not needed.issubset(set(fx.columns)):
        return prices, False
    rates = fx.reindex(prices.index).ffill().bfill()
    per_sid = rates.reindex(columns=ccy.to_numpy())
    per_sid.columns = prices.columns
    return prices * per_sid, True


# ── Price-based descriptors ───────────────────────────────────────────────────

def beta_and_residuals(returns: pd.DataFrame, market: pd.Series, halflife: int,
                       shrink: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """EWMA market beta, its residual series, and the residual EWMA volatility.

    Beta is Vasicek-shrunk toward the cross-sectional mean: the shrinkage weight
    is the estimate's precision relative to the cross-sectional dispersion of
    betas, so noisily-estimated (short-history, high-residual) names are pulled
    toward the crowd rather than trusted at face value.
    """
    min_periods = max(halflife, 20)
    cov, var, _, my = _ewm_pair(returns, market, halflife, min_periods)
    raw_beta = cov.div(var.replace(0, np.nan), axis=0)

    alpha = returns.ewm(halflife=halflife, min_periods=min_periods).mean() \
        .sub(raw_beta.mul(my, axis=0))
    resid = returns - alpha - raw_beta.shift(1).mul(market, axis=0)
    resid_vol = resid.ewm(halflife=halflife, min_periods=min_periods).std()

    if not shrink:
        return raw_beta, resid, resid_vol

    # Vasicek: w = sigma^2_cross / (sigma^2_cross + se^2), se^2 = var(e)/(N_eff*var(m))
    n_eff = _effective_n(halflife)
    se2 = (resid_vol ** 2).div(var.replace(0, np.nan) * n_eff, axis=0)
    cross_var = raw_beta.var(axis=1)
    prior = raw_beta.mean(axis=1)
    w = cross_var.values[:, None] / (cross_var.values[:, None] + se2.values + _EPS)
    w = np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0)
    shrunk = pd.DataFrame(w * raw_beta.values + (1 - w) * prior.values[:, None],
                          index=raw_beta.index, columns=raw_beta.columns)
    return shrunk.where(raw_beta.notna()), resid, resid_vol


def relative_strength(log_ret: pd.DataFrame, halflife: int, lag: int = 21,
                      window: int = 252) -> pd.DataFrame:
    """Barra RSTR: exponentially weighted mean log return over `window` days,
    skipping the most recent `lag` days so momentum is not polluted by reversal.
    Annualised purely so the raw numbers read as returns."""
    return (log_ret.shift(lag)
            .ewm(halflife=halflife, min_periods=min(window // 2, 120))
            .mean() * _TRADING_DAYS)


def cumulative_range(log_ret: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Barra CMRA: spread between the best and worst cumulative log return over
    the trailing 1..`months` monthly horizons. Loops over months (~120 of them),
    never over securities."""
    monthly = log_ret.resample("ME").sum(min_count=1)
    if len(monthly) < 2:
        return pd.DataFrame(index=log_ret.index, columns=log_ret.columns, dtype=float)
    values = monthly.to_numpy(dtype=float)
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        lo = max(0, i - months + 1)
        window = values[lo:i + 1]
        if len(window) < 3:
            continue
        cum = np.nancumsum(np.nan_to_num(window, nan=0.0), axis=0)
        valid = np.isfinite(window).sum(axis=0) >= 3
        rng = cum.max(axis=0) - cum.min(axis=0)
        out[i] = np.where(valid, rng, np.nan)
    return pd.DataFrame(out, index=monthly.index, columns=monthly.columns) \
        .reindex(log_ret.index, method="ffill")


def turnover(volume: pd.DataFrame, close: pd.DataFrame, market_cap: pd.DataFrame,
             shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily share turnover = traded shares / shares outstanding.

    Uses shares outstanding when the store has it (currency-free and exact);
    otherwise falls back to traded value over market cap, which is the same
    quantity as long as price and market cap are quoted in the same currency.
    """
    mcap = market_cap.reindex_like(volume).ffill()
    by_value = ((volume * close) / mcap.where(mcap > 0))
    if shares is None:
        return by_value.replace([np.inf, -np.inf], np.nan)
    # Per security, not per universe. During a backfill only some names have
    # shares outstanding yet; choosing one method for everyone based on whether
    # *any* security has the column would blank the turnover of all the others.
    denom = shares.reindex_like(volume).ffill()
    by_shares = (volume / denom.where(denom > 0))
    return by_shares.combine_first(by_value).replace([np.inf, -np.inf], np.nan)


def turnover_descriptors(daily_turnover: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """STOM / STOQ / STOA — log turnover over 1, 3 and 12 months."""
    floor = 1e-8
    stom = daily_turnover.rolling(21, min_periods=15).sum()
    stoq = daily_turnover.rolling(63, min_periods=40).sum() / 3.0
    stoa = daily_turnover.rolling(252, min_periods=120).sum() / 12.0
    return {
        "stom": _safe_log(stom.clip(lower=floor)),
        "stoq": _safe_log(stoq.clip(lower=floor)),
        "stoa": _safe_log(stoa.clip(lower=floor)),
    }


# ── Assembly ──────────────────────────────────────────────────────────────────

def build_price_descriptors(prices: pd.DataFrame, close: pd.DataFrame,
                            volume: pd.DataFrame, market_cap: pd.DataFrame,
                            shares: pd.DataFrame | None, spec: ModelSpec,
                            wanted: set[str]) -> tuple[dict[str, pd.DataFrame], dict]:
    """All price/volume descriptors the spec asks for, plus model by-products.

    The by-products (daily returns, the market series, residuals) are reused by
    the regression and risk steps, so they travel back with the descriptors
    instead of being recomputed.
    """
    ret = simple_returns(prices)
    lret = log_returns(prices)
    market = market_returns(ret, market_cap)

    out: dict[str, pd.DataFrame] = {}
    beta = resid = resid_vol = None
    needs_beta = bool({"beta", "hsigma", "dastd"} & wanted)
    if needs_beta:
        beta, resid, resid_vol = beta_and_residuals(ret, market, spec.beta_halflife)

    if "beta" in wanted:
        out["beta"] = beta
    if "hsigma" in wanted:
        out["hsigma"] = resid.ewm(halflife=spec.resvol_halflife,
                                  min_periods=spec.resvol_halflife).std() * np.sqrt(_TRADING_DAYS)
    if "dastd" in wanted:
        out["dastd"] = ret.ewm(halflife=spec.resvol_halflife,
                               min_periods=spec.resvol_halflife).std() * np.sqrt(_TRADING_DAYS)
    if "cmra" in wanted:
        out["cmra"] = cumulative_range(lret)
    if "rstr" in wanted:
        out["rstr"] = relative_strength(lret, spec.momentum_halflife)
    if "mom6" in wanted:
        out["mom6"] = lret.shift(21).rolling(105, min_periods=80).sum()
    if "strev" in wanted:
        out["strev"] = -lret.rolling(21, min_periods=15).sum()
    if {"stom", "stoq", "stoa"} & wanted:
        daily_to = turnover(volume, close, market_cap, shares)
        for key, frame in turnover_descriptors(daily_to).items():
            if key in wanted:
                out[key] = frame

    byproducts = {"returns": ret, "log_returns": lret, "market": market,
                  "beta": beta, "residuals": resid, "residual_vol": resid_vol}
    return out, byproducts


def build_growth_descriptors(level_panels: dict[str, pd.DataFrame],
                             wanted: set[str], lookback: int = 252) -> dict[str, pd.DataFrame]:
    """Year-on-year growth from stored *level* series (revenue, EPS).

    Deriving growth here rather than ingesting a vendor growth field means the
    trailing window is the one we chose and is the same for every security —
    vendor growth fields differ in whether they use fiscal years, TTM or
    estimates, which makes them incomparable across a mixed-market universe.
    Levels can be negative (loss-making EPS), where a growth *rate* is
    meaningless, so those observations are dropped rather than sign-flipped.
    """
    out: dict[str, pd.DataFrame] = {}
    for key in wanted:
        desc = DESCRIPTORS.get(key)
        if desc is None or desc.transform != "yoy_growth":
            continue
        level = level_panels.get(desc.column)
        if level is None or level.empty:
            continue
        base = level.shift(lookback)
        growth = (level / base.where(base > 0) - 1.0)
        out[key] = growth.where(level > 0).replace([np.inf, -np.inf], np.nan)
    return out


def build_fundamental_descriptors(fund_wide: dict[str, pd.DataFrame],
                                  wanted: set[str]) -> dict[str, pd.DataFrame]:
    """Apply each fundamental descriptor's transform to its store column.

    A ratio is inverted rather than used raw so that "more of the style" always
    means "bigger number" — 1/(P/B) is book-to-price, and unlike P/B it stays
    finite and well-ordered when the ratio is tiny.
    """
    out: dict[str, pd.DataFrame] = {}
    for key in wanted:
        desc = DESCRIPTORS.get(key)
        if desc is None or desc.source != "fund" or not desc.column:
            continue
        if desc.transform == "yoy_growth":
            continue          # handled by build_growth_descriptors, from daily levels
        raw = fund_wide.get(desc.column)
        if raw is None or raw.empty:
            continue
        if desc.transform == "reciprocal":
            value = 1.0 / raw.where(raw > 0)
        elif desc.transform == "log":
            value = _safe_log(raw)
        else:
            value = raw.where(np.isfinite(raw))
        if desc.invert:
            value = -value
        out[key] = value.replace([np.inf, -np.inf], np.nan)
    return out
