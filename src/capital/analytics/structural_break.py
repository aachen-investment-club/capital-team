"""
Structural break detection — two estimators.

CUSUMDetector   : two-sided Page-Hinkley CUSUM
                  input: z-scored stress level signal
                  output: +1 (upward break) / -1 (downward break) / 0 (stable)

StudentTBOCPD   : Bayesian Online Changepoint Detection with Student-t likelihood
                  input: z-scored stress level signal
                  output: changepoint probability at each step [0, 1]

SignalBus       : builds the aligned feature DataFrame fed to both estimators
                  features: gjr_vol · c_level · liq_level · stress_idx · roll_corr

_expanding_zscore: causal z-score — at time t uses only data up to t (no lookahead)
_level_signal    : composite 10-day smoothed (vol + credit + liquidity), then z-scored
"""

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expanding_zscore(s: pd.Series, min_periods: int = 120) -> pd.Series:
    """Causal expanding z-score — no lookahead."""
    mu  = s.expanding(min_periods=min_periods).mean()
    std = s.expanding(min_periods=min_periods).std()
    return ((s - mu) / (std + 1e-9)).dropna()


def build_level_signal(gjr_vol: pd.Series,
                       c_level: pd.Series,
                       liq_level: pd.Series,
                       smooth: int = 10) -> pd.Series:
    """
    10-day smoothed composite of GJR-vol + credit level + liquidity level,
    then causal expanding z-score.  This is the primary input to CUSUM and BOCPD.
    """
    raw = (
        gjr_vol.rolling(smooth).mean().fillna(0)
        + c_level.rolling(smooth).mean().fillna(0)
        + liq_level.rolling(smooth).mean().fillna(0)
    )
    return _expanding_zscore(raw).rename("level_signal")


# ── CUSUM ─────────────────────────────────────────────────────────────────────

class CUSUMDetector:
    """
    Two-sided Page-Hinkley CUSUM.

    S+_t = max(0, S+_{t-1} + x_t - k)   → detects upward stress shifts
    S-_t = min(0, S-_{t-1} + x_t + k)   → detects downward stress shifts

    Signal on threshold cross, then reset.
    k=0.3 (balanced), h=6.0 (conservative — reduces false positives).
    """

    def __init__(self, k: float = 0.3, h: float = 6.0):
        self.k = k
        self.h = h

    def detect(self, x: pd.Series) -> pd.Series:
        clean  = x.dropna()
        vals   = clean.values
        n      = len(vals)
        s_pos  = np.zeros(n)
        s_neg  = np.zeros(n)
        sigs   = np.zeros(n)

        for t in range(1, n):
            s_pos[t] = max(0.0, s_pos[t-1] + vals[t] - self.k)
            s_neg[t] = min(0.0, s_neg[t-1] + vals[t] + self.k)

            if s_pos[t] > self.h:
                sigs[t]  =  1
                s_pos[t] =  0.0   # reset after signal
            elif s_neg[t] < -self.h:
                sigs[t]  = -1
                s_neg[t] =  0.0

        return pd.Series(sigs, index=clean.index, name="cusum")

    def stat_series(self, x: pd.Series) -> pd.DataFrame:
        """Return the running S+ and S- statistics (useful for plotting)."""
        clean  = x.dropna()
        vals   = clean.values
        n      = len(vals)
        s_pos  = np.zeros(n)
        s_neg  = np.zeros(n)

        for t in range(1, n):
            sp = max(0.0, s_pos[t-1] + vals[t] - self.k)
            sn = min(0.0, s_neg[t-1] + vals[t] + self.k)
            if sp > self.h:
                sp = 0.0
            if sn < -self.h:
                sn = 0.0
            s_pos[t] = sp
            s_neg[t] = sn

        return pd.DataFrame({"S_pos": s_pos, "S_neg": s_neg}, index=clean.index)


# ── BOCPD ─────────────────────────────────────────────────────────────────────

class StudentTBOCPD:
    """
    Bayesian Online Changepoint Detection with Student-t likelihood.

    At each step the model maintains a run-length distribution R over how long
    the current regime has lasted.  The changepoint probability is R[0].

    hazard  = 1/expected_regime_length
              1/50  = short  (~10 weeks)
              1/150 = medium (~30 weeks)  [default]
              1/300 = long   (~60 weeks)

    nu      = Student-t degrees of freedom (fat-tail control, default 5)
    """

    def __init__(self, hazard: float = 1/150, nu: float = 5):
        self.hazard = hazard
        self.nu     = nu
        self._reset()

    def _reset(self):
        self.R     = np.array([1.0])
        self.mu    = 0.0
        self.kappa = 1.0
        self.alpha = 1.0
        self.beta  = 1.0

    def _predictive(self):
        scale = self.beta / (self.alpha * self.kappa)
        var   = scale * (self.nu / (self.nu - 2 + 1e-9))
        return self.mu, var

    def _update(self, x: float):
        self.kappa += 1
        delta       = x - self.mu
        self.mu    += delta / self.kappa
        self.alpha += 0.5
        self.beta  += 0.5 * delta ** 2

    def step(self, x: float) -> float:
        """Process one observation; return P(changepoint at t)."""
        x    = float(x)
        mean, var = self._predictive()

        likelihood = (1.0 + (x - mean) ** 2 / (self.nu * (var + 1e-9))) ** (-(self.nu + 1) / 2)

        growth = self.R * likelihood * (1.0 - self.hazard)
        cp     = float(np.sum(self.R * likelihood * self.hazard))

        new_R    = np.zeros(len(growth) + 1)
        new_R[0] = cp
        new_R[1:] = growth
        total    = new_R.sum()
        self.R   = new_R / (total if total > 0 else 1.0)

        self._update(x)
        return self.R[0]

    def run(self, x: pd.Series) -> pd.Series:
        """
        Run the full series.  Returns normalised surprise [0, 1]:
        raw changepoint probability smoothed over 20 days then min-max scaled.
        Also stores .raw_probs for the unsmoothed series.
        """
        self._reset()
        clean = x.dropna()
        probs = np.array([self.step(v) for v in clean.values])
        self.raw_probs = pd.Series(probs, index=clean.index, name="bocpd_raw")

        smoothed = self.raw_probs.rolling(20).mean()
        rng      = smoothed.max() - smoothed.min() + 1e-9
        normed   = ((smoothed - smoothed.min()) / rng).rename("bocpd")
        return normed
