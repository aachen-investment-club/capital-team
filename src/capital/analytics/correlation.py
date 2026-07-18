"""
Correlation analysis — three models:
  RollingCorr    : Fisher-z smoothed rolling window correlation
  PartialCorr    : controls for a third variable via OLS residuals
  DCC            : Dynamic Conditional Correlation with GARCH(1,1) standardisation
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize


# ── Rolling correlation ───────────────────────────────────────────────────────

class RollingCorr:
    def __init__(self, price1: pd.Series, price2: pd.Series,
                 window_corr: int = 30, window_smooth: int = 10):
        self.price1 = price1
        self.price2 = price2
        self.window_corr   = window_corr
        self.window_smooth = window_smooth

    @staticmethod
    def _to_float(s) -> pd.Series:
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.squeeze()
        return s.apply(pd.to_numeric, errors="coerce").dropna().astype(float)

    def _log_ret(self, price) -> pd.Series:
        p = self._to_float(price)
        return np.log(p / p.shift(1))

    def rolling_corr(self) -> pd.Series:
        r1 = self._log_ret(self.price1)
        r2 = self._log_ret(self.price2)
        return r1.rolling(self.window_corr).corr(r2)

    @staticmethod
    def fisher_transform(corr: pd.Series) -> pd.Series:
        corr = corr.clip(-0.9999, 0.9999)
        return 0.5 * np.log((1 + corr) / (1 - corr))

    def run(self) -> pd.Series:
        """Returns smoothed Fisher-z rolling correlation."""
        corr   = self.rolling_corr()
        fisher = self.fisher_transform(corr)
        return fisher.rolling(self.window_smooth).mean()

    def run_rho(self) -> pd.Series:
        """Returns smoothed correlation in [-1, 1] (tanh of Fisher-z)."""
        return np.tanh(self.run())


# ── Partial correlation ───────────────────────────────────────────────────────

class PartialCorr:
    """
    Measures the correlation between x and y after removing the linear
    influence of `control` from both series via OLS.

    Returns a scalar partial correlation and optionally the full residual
    time-series for plotting a rolling version.
    """
    def __init__(self, x: pd.Series, y: pd.Series, control: pd.Series):
        self.x       = x
        self.y       = y
        self.control = control

    @staticmethod
    def _clean(s: pd.Series) -> pd.Series:
        """Ensure 1-D float64 Series with a clean DatetimeIndex."""
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = s.squeeze()
        s.index = pd.to_datetime(s.index)
        return s.apply(pd.to_numeric, errors="coerce").dropna().astype(float)

    def _log_rets(self):
        x   = self._clean(self.x)
        y   = self._clean(self.y)
        c   = self._clean(self.control)
        x_r = np.log(x / x.shift(1)).dropna()
        y_r = np.log(y / y.shift(1)).dropna()
        c_r = np.log(c / c.shift(1)).dropna()
        idx = x_r.index.intersection(y_r.index).intersection(c_r.index)
        return x_r.loc[idx], y_r.loc[idx], c_r.loc[idx]

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        return ((s - s.mean()) / (s.std() + 1e-10)).astype(float)

    def _residuals(self):
        x_r, y_r, c_r = self._log_rets()
        xz = self._zscore(x_r)
        yz = self._zscore(y_r)
        cz = self._zscore(c_r)
        C  = sm.add_constant(cz.values)
        x_resid = pd.Series(sm.OLS(xz.values, C).fit().resid, index=xz.index)
        y_resid = pd.Series(sm.OLS(yz.values, C).fit().resid, index=yz.index)
        return x_resid, y_resid

    def compute(self) -> float:
        """Scalar partial correlation (EWMA-smoothed, span=60)."""
        x_r, y_r = self._residuals()
        return float(x_r.ewm(span=60).mean().corr(y_r.ewm(span=60).mean()))

    def rolling(self, window: int = 60) -> pd.Series:
        """Rolling partial correlation in [-1, 1]."""
        x_r, y_r = self._residuals()
        return x_r.rolling(window).corr(y_r).rename(
            f"partial({self.x.name},{self.y.name}|{self.control.name})"
        )


# ── GARCH(1,1) ────────────────────────────────────────────────────────────────

class GARCH11:
    def __init__(self):
        self.mu = self.omega = self.alpha = self.beta = None
        self.sigma2 = None

    @staticmethod
    def _recursion(eps, omega, alpha, beta, s2_init):
        T      = len(eps)
        sigma2 = np.empty(T)
        sigma2[0] = s2_init
        for t in range(1, T):
            sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        return sigma2

    def _neg_loglik(self, params, ret):
        mu, omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1 - 1e-6:
            return 1e12
        eps    = ret - mu
        sigma2 = self._recursion(eps, omega, alpha, beta, eps.var())
        if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)):
            return 1e12
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2)

    def fit(self, ret: np.ndarray) -> "GARCH11":
        ret    = np.asarray(ret, dtype=float)
        v      = ret.var()
        x0     = [ret.mean(), v * 0.05, 0.05, 0.90]
        bounds = [(None, None), (1e-10, None), (0.0, 0.5), (0.0, 0.999)]
        cons   = [{"type": "ineq", "fun": lambda p: 1 - p[2] - p[3] - 1e-6}]
        res    = minimize(self._neg_loglik, x0, args=(ret,),
                          method="SLSQP", bounds=bounds, constraints=cons,
                          options={"ftol": 1e-9, "maxiter": 300})
        self.mu, self.omega, self.alpha, self.beta = res.x
        eps         = ret - self.mu
        self.sigma2 = self._recursion(eps, self.omega, self.alpha, self.beta, eps.var())
        return self

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(self.sigma2)


def garch_standardise(log_ret_df: pd.DataFrame) -> pd.DataFrame:
    """Fit GARCH(1,1) per column; return z_t = r_t / sigma_t."""
    out = {}
    for col in log_ret_df.columns:
        r     = log_ret_df[col].dropna().values
        m     = GARCH11().fit(r * 100.0)
        sigma = pd.Series(m.sigma / 100.0, index=log_ret_df[col].dropna().index)
        out[col] = log_ret_df[col] / sigma
    z = pd.DataFrame(out).dropna()
    return z


# ── DCC ───────────────────────────────────────────────────────────────────────

class DCC:
    """
    Dynamic Conditional Correlation (Engle 2002).
    Q_t = (1-a-b)*S + a*(z_{t-1} z_{t-1}^T) + b*Q_{t-1}
    R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    """
    def __init__(self, a: float = 0.02, b: float = 0.97):
        self.a = a
        self.b = b

    def fit(self, z: pd.DataFrame) -> pd.DataFrame:
        """
        z  : DataFrame of GARCH-standardised returns (T × N)
        Returns DataFrame of pairwise DCC correlations over time,
        columns named 'A_B'.
        """
        arr  = z.values
        T, N = arr.shape
        cols = list(z.columns)
        S    = (arr.T @ arr) / T
        Q    = S.copy()
        R_history = []

        for t in range(1, T):
            z_prev = arr[t-1].reshape(-1, 1)
            Q      = (1 - self.a - self.b) * S + self.a * (z_prev @ z_prev.T) + self.b * Q
            d      = np.sqrt(np.diag(Q))
            R_t    = Q / np.outer(d, d)
            np.fill_diagonal(R_t, 1.0)
            R_history.append(R_t)

        R_arr = np.array(R_history)   # (T-1, N, N)
        dates = z.index[1:]
        out   = {}
        for i in range(N):
            for j in range(i + 1, N):
                out[f"{cols[i]}_{cols[j]}"] = pd.Series(R_arr[:, i, j], index=dates)
        return pd.DataFrame(out)


# ── Convenience: fetch prices from EOD cache ─────────────────────────────────

def prices_from_eod(tickers: list[str], eod_fn, security_id_map: dict) -> pd.DataFrame:
    """
    Build a price DataFrame from the app's EOD cache.
    eod_fn     : callable(security_id) -> DataFrame with date + adj_close
    security_id_map : {ticker: security_id}
    """
    cols = {}
    for ticker in tickers:
        sid = security_id_map.get(ticker)
        if sid is None:
            continue
        eod = eod_fn(sid)
        if isinstance(eod, pd.DataFrame) and not eod.empty:
            eod = eod.copy()
            eod["date"] = pd.to_datetime(eod["date"])
            p_col = "adj_close" if "adj_close" in eod.columns else "close"
            cols[ticker] = eod.set_index("date")[p_col]
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).dropna()
