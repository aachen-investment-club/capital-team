'''HAR_RV could be used for benchmark to GARCH, however needs intraday data to compute, which might not be what I am looking for.
HAR-RV is very good short-term predictor of volatility, but not so much for long-term?
The HAR-RV is a simple regression model of daily, weekly and monthly realised volatitility to predict the next day's vol

I have also added HAR-CJ, which decomposes continuous and jump components of volatility 
This is done using bipower variation
Idea: RV = BV + J
BV = (pi/2) * sum(|r_i| * |r_{i-1}|)
J = max(RV - BV, 0)
-------------------------------------------------
The prices should be a series of prices every five minutes 

'''

'''HAR-RV and HAR-CJ.

HAR-RV: simple regression of next-day log RV on daily/weekly/monthly log RV.
HAR-CJ: adds a jump component using bipower variation:
    BPV   = (pi/2) * sum_i |r_i| * |r_{i-1}|     (jump-robust variance)
    J     = max(RV - BPV, 0)                     (jump component)
    C     = RV - J                               (continuous component)
The HAR-CJ regression is:
    log(RV_{t+1}) ~ const + log(C_d) + log(C_w) + log(C_m) + log1p(J_d)

Input is a series of intraday prices (e.g. 5-min closes).
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ------------------------------------------------------------
# realised variance (sum of squared 5-min log returns, in %^2)
# ------------------------------------------------------------
def realised_variance(intraday: pd.Series):
    r = np.log(intraday / intraday.shift(1)) * 100
    rv = (r ** 2).groupby(r.index.date).sum()
    rv.index = pd.to_datetime(rv.index)
    return rv.replace(0, np.nan).dropna()


# ------------------------------------------------------------
# bipower variation (jump-robust, SAME SCALE as realised_variance)
# ------------------------------------------------------------
def bipower_variation(intraday: pd.Series):
    # IMPORTANT: multiply by 100 to match realised_variance's % scale.
    r = (np.log(intraday / intraday.shift(1)) * 100).dropna()
    abs_r = np.abs(r)

    def _bpv(vals):
        v = vals.values
        if len(v) < 2:
            return np.nan
        return (np.pi / 2.0) * np.sum(v[1:] * v[:-1])   # SUM, not mean

    bv = abs_r.groupby(abs_r.index.date).apply(_bpv)
    bv.index = pd.to_datetime(bv.index)
    return bv.replace([np.inf, -np.inf, 0], np.nan).dropna()


# ------------------------------------------------------------
# HAR-RV
# ------------------------------------------------------------
class HAR_RVModel:

    def __init__(self, intraday):
        self.intraday = intraday
        self.rv = realised_variance(intraday)

    @staticmethod
    def _features(rv):
        return pd.DataFrame({
            "d": np.log(rv),
            "w": np.log(rv.rolling(5).mean()),
            "m": np.log(rv.rolling(22).mean()),
        }).replace([np.inf, -np.inf], np.nan)

    def _design(self):
        feats = self._features(self.rv)
        y     = np.log(self.rv.shift(-1)).replace([np.inf, -np.inf], np.nan)
        df    = feats.join(y.rename("y")).dropna()
        Y = df["y"].values
        X = sm.add_constant(df[["d", "w", "m"]].values)
        self.index_ = df.index
        return Y, X

    def fit_HAR_RV(self):
        Y, X = self._design()
        self.model   = sm.OLS(Y, X).fit()
        self.sigma2_ = float(self.model.resid.var(ddof=1))
        return self.model

    def predict(self, X):
        """Jensen-corrected: E[RV] ≈ exp(E[log RV] + σ²/2)."""
        return np.exp(self.model.predict(X) + 0.5 * self.sigma2_)

    def build_features_for(self, intraday):
        """Build (feature DataFrame, design matrix X) from any intraday series.
        Use this to forecast over a window different from the training one."""
        rv    = realised_variance(intraday)
        feats = self._features(rv).dropna()
        X     = sm.add_constant(feats.values, has_constant="add")
        return feats, X


# ------------------------------------------------------------
# HAR-CJ
# ------------------------------------------------------------
class HAR_RV_CJ:

    def __init__(self, intraday):
        self.intraday = intraday
        self.rv = realised_variance(intraday)
        self.bv = bipower_variation(intraday)
        self._df = self._make_table(self.rv, self.bv)

    @staticmethod
    def _make_table(rv, bv):
        bv_aligned = bv.reindex(rv.index)
        j = (rv - bv_aligned).clip(lower=0).fillna(0.0)
        c = (rv - j).clip(lower=1e-12)
        return pd.DataFrame({"RV": rv, "BV": bv_aligned, "C": c, "J": j})

    @staticmethod
    def _features(df_cj):
        c = df_cj["C"]
        j = df_cj["J"]
        return pd.DataFrame({
            "d":  np.log(c),
            "w":  np.log(c.rolling(5).mean()),
            "m":  np.log(c.rolling(22).mean()),
            "j":  np.log1p(j),                # log(1 + J) so zero-jump days survive
        }).replace([np.inf, -np.inf], np.nan)

    def _design(self):
        feats = self._features(self._df)
        y     = np.log(self._df["RV"].shift(-1)).replace([np.inf, -np.inf], np.nan)
        df    = feats.join(y.rename("y")).dropna()
        if len(df) < 50:
            raise ValueError(
                f"HAR-CJ design too small after cleaning (got {len(df)} rows). "
                f"Check that bipower_variation is on the right scale."
            )
        Y = df["y"].values
        X = sm.add_constant(df[["d", "w", "m", "j"]].values)
        self.index_ = df.index
        return Y, X

    def fit_HAR_RV_CJ(self):
        Y, X = self._design()
        self.model   = sm.OLS(Y, X).fit()
        self.sigma2_ = float(self.model.resid.var(ddof=1))
        return self.model

    def predict(self, X):
        return np.exp(self.model.predict(X) + 0.5 * self.sigma2_)

    def build_features_for(self, intraday):
        """Build (feature DataFrame, X) from any intraday series.
        Used to forecast across train+test without re-fitting."""
        rv    = realised_variance(intraday)
        bv    = bipower_variation(intraday)
        df_cj = self._make_table(rv, bv)
        feats = self._features(df_cj).dropna()
        X     = sm.add_constant(feats.values, has_constant="add")
        return feats, X