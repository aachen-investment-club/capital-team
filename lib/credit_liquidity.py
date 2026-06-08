"""
Credit & Liquidity filters — ported from dump/credit_liquid.

CreditFilter   : level / shock / acceleration from HY OAS (FRED BAMLH0A0HYM2)
LiquidityFilter: level / shock / acceleration from price + volume (Amihud + volume + stability)
StressScore    : weighted composite of all six signals + Z-significance test
"""
import numpy as np
import pandas as pd
from scipy import stats


# ── helpers ───────────────────────────────────────────────────────────────────

def _zscore(x: pd.Series, window: int) -> pd.Series:
    return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-9)

def _clean(s) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.squeeze()
    s.index = pd.to_datetime(s.index)
    return s.apply(pd.to_numeric, errors="coerce").dropna().astype(float)


# ── Credit filter ─────────────────────────────────────────────────────────────

class CreditFilter:
    """
    Input: ICE BofA HY OAS in basis points (FRED: BAMLH0A0HYM2).
    Positive z-scores = stress / wider spreads.
    """
    def __init__(self, hy_oas: pd.Series, window: int = 504):
        self.oas    = _clean(hy_oas)
        self.window = window

    def level(self) -> pd.Series:
        """Z-score of spread level vs trailing window."""
        return _zscore(self.oas, self.window).rename("credit_level")

    def shock(self) -> pd.Series:
        """Z-score of smoothed daily change in OAS."""
        s = self.oas.diff().rolling(5).mean()
        return _zscore(s, self.window).rename("credit_shock")

    def acceleration(self) -> pd.Series:
        """Z-score of smoothed change-of-change in OAS."""
        s = self.oas.diff().rolling(5).mean()
        a = s.diff().rolling(5).mean()
        return _zscore(a, self.window).rename("credit_accel")

    def run(self) -> dict[str, pd.Series]:
        return {
            "credit_level": self.level(),
            "credit_shock": self.shock(),
            "credit_accel": self.acceleration(),
        }


# ── Liquidity filter ──────────────────────────────────────────────────────────

class LiquidityFilter:
    """
    Input: price and volume series (e.g. SPY).
    Positive composite = illiquidity stress.
    """
    def __init__(self, price: pd.Series, volume: pd.Series, window: int = 504):
        self.price  = _clean(price)
        self.volume = _clean(volume)
        self.window = window

    def _returns(self) -> pd.Series:
        return np.log(self.price / self.price.shift(1))

    def amihud(self) -> pd.Series:
        """Amihud illiquidity ratio, z-scored. Higher = more illiquid."""
        ret       = self._returns().abs()
        dvol      = self.price * self.volume
        raw       = ret / (dvol + 1e-9)
        smoothed  = raw.rolling(20).mean()
        return _zscore(smoothed, self.window)

    def _volume_z(self) -> pd.Series:
        v = np.log(self.volume.replace(0, np.nan))
        return _zscore(v, self.window)

    def _stability_z(self) -> pd.Series:
        turnover  = np.log(self.price * self.volume)
        stability = 1.0 / (turnover.rolling(20).std() + 1e-9)
        return _zscore(stability, self.window)

    def level(self) -> pd.Series:
        """
        Composite illiquidity: 50% Amihud + 30% (−volume) + 20% instability.
        Positive = tighter/worse liquidity.
        """
        a = self.amihud()
        v = self._volume_z()
        s = self._stability_z()
        liq = 0.5 * a + 0.3 * (-v) + 0.2 * s
        return liq.rename("liquidity_level")

    def shock(self, liq_level: pd.Series) -> pd.Series:
        s = liq_level.diff().rolling(5).mean()
        return _zscore(s, self.window).rename("liquidity_shock")

    def acceleration(self, liq_shock: pd.Series) -> pd.Series:
        a = liq_shock.diff().rolling(5).mean()
        return _zscore(a, self.window).rename("liquidity_accel")

    def run(self) -> dict[str, pd.Series]:
        lev   = self.level()
        shk   = self.shock(lev)
        accel = self.acceleration(shk)
        return {
            "liquidity_level": lev,
            "liquidity_shock": shk,
            "liquidity_accel": accel,
        }


# ── Composite stress score ────────────────────────────────────────────────────

class StressScore:
    """
    Stress index = 0.5 * (credit_shock + liquidity_shock).
    Simple, interpretable, and focused on the fastest-moving signals.
    """
    def __init__(self, signals: dict[str, pd.Series]):
        self.signals = signals

    def composite(self) -> pd.Series:
        cs = self.signals.get("credit_shock",    pd.Series(dtype=float))
        ls = self.signals.get("liquidity_shock", pd.Series(dtype=float))
        if cs.empty and ls.empty:
            return pd.Series(dtype=float)
        if cs.empty:
            return ls.rename("stress_score")
        if ls.empty:
            return cs.rename("stress_score")
        idx = cs.index.intersection(ls.index)
        return (0.5 * cs.loc[idx] + 0.5 * ls.loc[idx]).abs().rename("stress_score")

    def run(self) -> tuple[pd.Series, pd.Series]:
        """Returns (composite, empty Series — significance now handled empirically in the page)."""
        comp = self.composite()
        return comp, pd.Series(dtype=float)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a FRED series directly via the public CSV endpoint (no API key needed)."""
    import requests, io
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    date_col = df.columns[0]
    val_col  = df.columns[1]
    df.index = pd.to_datetime(df[date_col])
    s = df[val_col].replace(".", np.nan).dropna().astype(float).sort_index()
    return s.loc[start:end]


def load_spy_lseg(start: str, end: str) -> tuple[pd.Series, pd.Series]:
    """Fetch SPY close and volume from LSEG."""
    import lseg.data as ld
    ld.open_session()
    try:
        close = ld.get_history(
            universe=["SPY.P"], fields=["TR.PriceClose"],
            parameters={"Adjusted": 1}, start=start, end=end, interval="1D",
        )
        volume = ld.get_history(
            universe=["SPY.P"], fields=["TR.Volume"],
            parameters={"Adjusted": 1}, start=start, end=end, interval="1D",
        )
    finally:
        try:
            ld.close_session()
        except Exception:
            pass

    def _s(df):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        return df.iloc[:, 0].sort_index()

    return _s(close), _s(volume)
