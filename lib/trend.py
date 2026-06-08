"""
Trend detection — three models + GJR-GARCH volatility input.

SMAFilter      : simple moving-average trend + slope
KalmanFilter2D : pykalman 2-state (level + velocity) linear Kalman
GJRGarch       : GJR-GARCH(1,1) conditional volatility via arch
UKFFilter      : filterpy Unscented Kalman, adaptive to vol + stress

All filters share a common interface:
    .run(prices, vol=None, stress=None) -> pd.DataFrame
    columns: trend, slope, acceleration (+ residual for Kalman models)

forecast_linear(df, horizon) -> pd.DataFrame
    Projects trend forward using last slope, returns dates + levels + slope.
"""
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(s) -> pd.Series:
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.squeeze().copy()
    s.index = pd.to_datetime(s.index)
    return s.apply(pd.to_numeric, errors="coerce").dropna().sort_index().astype(float)


def _align(*series: pd.Series) -> list[pd.Series]:
    idx = series[0].index
    for s in series[1:]:
        if s is not None and not s.empty:
            idx = idx.intersection(s.index)
    return [s.loc[idx] if (s is not None and not s.empty) else None for s in series]


# ── SMA benchmark ────────────────────────────────────────────────────────────

class SMAFilter:
    def __init__(self, window: int = 50):
        self.window = window

    def run(self, prices: pd.Series, vol=None, stress=None) -> pd.DataFrame:
        p = _clean(prices)
        trend = p.rolling(self.window).mean()
        slope = trend.diff()
        out = pd.DataFrame({"trend": trend, "slope": slope}, index=p.index)
        out["acceleration"] = out["slope"].diff()
        return out.dropna()


# ── 2D Kalman (pykalman) ─────────────────────────────────────────────────────

class KalmanFilter2D:
    """Standard linear Kalman: state = [level, velocity]."""

    def __init__(self, obs_cov: float = 1.0, proc_cov: float = 0.01):
        self.obs_cov  = obs_cov
        self.proc_cov = proc_cov

    def run(self, prices: pd.Series, vol=None, stress=None) -> pd.DataFrame:
        from pykalman import KalmanFilter

        p = _clean(prices)

        kf = KalmanFilter(
            transition_matrices=np.array([[1, 1], [0, 1]]),
            observation_matrices=np.array([[1, 0]]),
            initial_state_mean=[float(p.iloc[0]), 0.0],
            observation_covariance=self.obs_cov,
            transition_covariance=self.proc_cov * np.eye(2),
        )
        means, _ = kf.filter(p.values)

        out = pd.DataFrame({
            "trend": means[:, 0],
            "slope": means[:, 1],
        }, index=p.index)
        out["acceleration"] = out["slope"].diff()
        out["residual"]     = p.values - out["trend"].values
        return out


# ── GJR-GARCH volatility ─────────────────────────────────────────────────────

class GJRGarch:
    """
    Fits GJR-GARCH(1,1,1) on daily log-returns (% scale).
    Returns conditional volatility as a daily Series (annualised, fraction).
    """

    def fit(self, prices: pd.Series) -> pd.Series:
        from arch import arch_model

        p   = _clean(prices)
        ret = np.log(p / p.shift(1)).dropna() * 100.0   # pct scale for arch

        am  = arch_model(ret, mean="Constant", vol="GARCH",
                         p=1, o=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off", show_warning=False)

        cond_vol = res.conditional_volatility          # pct daily
        cond_vol.index = pd.to_datetime(cond_vol.index)

        # Annualised vol in decimal (e.g. 0.18 = 18%) — exposed for display
        self.annualised_vol = (cond_vol * np.sqrt(252) / 100.0).rename("gjr_vol_ann")

        # Full z-score (unclipped) — used by structural break signal bus
        mu  = cond_vol.mean()
        sig = cond_vol.std() + 1e-9
        self.zscore_vol = ((cond_vol - mu) / sig).rename("gjr_vol_z")

        # Clipped z-score — used as non-negative UKF scaling input
        return self.zscore_vol.clip(lower=0).rename("gjr_vol")


# ── Unscented Kalman Filter (filterpy) ───────────────────────────────────────

class UKFFilter:
    """
    Adaptive UKF: non-linear slope persistence, covariances driven by
    GJR-GARCH vol and credit/liquidity stress score.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta:  float = 2.0,
        kappa: float = 0.0,
        base_process_var: float   = 0.01,
        base_observation_var: float = 1.0,
    ):
        self.alpha             = alpha
        self.beta              = beta
        self.kappa             = kappa
        self.base_process_var  = base_process_var
        self.base_obs_var      = base_observation_var

    @staticmethod
    def _fx(x, dt):
        level, slope = x
        new_level = level + slope * dt
        new_slope = 0.95 * slope - 0.05 * slope ** 3
        return np.array([new_level, new_slope])

    @staticmethod
    def _hx(x):
        return np.array([x[0]])

    def run(self, prices: pd.Series,
            vol: pd.Series = None,
            stress: pd.Series = None) -> pd.DataFrame:
        from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

        p = _clean(prices)

        # defaults
        if vol is None or (isinstance(vol, pd.Series) and vol.empty):
            vol = pd.Series(np.zeros(len(p)), index=p.index)
        if stress is None or (isinstance(stress, pd.Series) and stress.empty):
            stress = pd.Series(np.zeros(len(p)), index=p.index)

        # align all to price index (forward-fill sparse series like stress)
        vol    = vol.reindex(p.index).ffill().bfill().fillna(0.0)
        stress = stress.reindex(p.index).ffill().bfill().fillna(0.0)

        pts = MerweScaledSigmaPoints(n=2, alpha=self.alpha,
                                     beta=self.beta, kappa=self.kappa)
        ukf = UnscentedKalmanFilter(
            dim_x=2, dim_z=1, dt=1.0,
            fx=self._fx, hx=self._hx, points=pts,
        )
        ukf.x = np.array([float(p.iloc[0]), 0.0])
        ukf.P = np.diag([float(p.iloc[0]) ** 2 * 0.01, 1.0])
        ukf.Q = self.base_process_var * np.eye(2)
        ukf.R = np.array([[self.base_obs_var]])

        levels, slopes, residuals = [], [], []

        for t in range(len(p)):
            vol_t    = float(vol.iloc[t])
            stress_t = float(stress.iloc[t])

            ukf.R = np.array([[self.base_obs_var * (1 + vol_t)]])
            ukf.Q = self.base_process_var * (1 + stress_t) * np.eye(2)

            # keep P positive-definite
            P_sym = (ukf.P + ukf.P.T) / 2
            ev, evec = np.linalg.eigh(P_sym)
            ev = np.maximum(ev, 1e-4)
            ukf.P = evec @ np.diag(ev) @ evec.T

            ukf.predict()
            ukf.update(p.iloc[t])

            levels.append(ukf.x[0])
            slopes.append(ukf.x[1])
            residuals.append(float(p.iloc[t]) - ukf.x[0])

        out = pd.DataFrame({
            "trend":    levels,
            "slope":    slopes,
            "residual": residuals,
        }, index=p.index)
        out["acceleration"] = out["slope"].diff()
        return out


# ── Forward projection ────────────────────────────────────────────────────────

def forecast_linear(filter_df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Linear extrapolation using the last estimated slope.
    Returned DataFrame: index = future dates, columns = trend, slope, lower_1s, upper_1s.
    Uncertainty band: grows as sqrt(t) × |residual std|.
    """
    last_date  = filter_df.index[-1]
    last_level = float(filter_df["trend"].iloc[-1])
    last_slope = float(filter_df["slope"].iloc[-1])

    # residual std as 1-sigma uncertainty (if available)
    if "residual" in filter_df.columns:
        res_std = float(filter_df["residual"].std())
    else:
        res_std = 0.0

    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                  periods=horizon)
    t = np.arange(1, horizon + 1)

    # nonlinear slope decay mirrors UKF fx: s_t = 0.95*s_{t-1} - 0.05*s_{t-1}^3
    slopes_fwd = np.empty(horizon)
    s = last_slope
    for i in range(horizon):
        s = 0.95 * s - 0.05 * s ** 3
        slopes_fwd[i] = s

    levels_fwd = last_level + np.cumsum(slopes_fwd)
    sigma = res_std * np.sqrt(t)

    return pd.DataFrame({
        "trend":    levels_fwd,
        "slope":    slopes_fwd,
        "lower_1s": levels_fwd - sigma,
        "upper_1s": levels_fwd + sigma,
    }, index=future_dates)
