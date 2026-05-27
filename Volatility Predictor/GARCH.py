'''This is the GARCH family volatility model

For fitting my GARCH, I consider 4 factors:

1. Mean equation: I will start with mean as "constant" and then move to AR(1) if necessary, but seeing as I am taking daily data, mean is mostly not autocorrelated. 
Daily returns are mostly white noise, if i decided to use weekly data, I could take into account momentum or mean reversion effects through AR(1). Start simple first!

2. Variance equation: I will start with GARCH as baseline, I can add GJR-GARCH to compare leverage effects later on (idea that negative days are more severe than positive).

3. Orders (p,q): I will start with (1,1), keep it simple.

4. Distribution: I will start with  student-t and move to skewed student-t if necessary. Gaussian often breaks down and does not capture fat tails.
Student-t has slower probability decay in tails than Gaussian (polynomial vs exponential), therefore better capturing extreme events. 

--------------------------------
EVALUATION WITHIN GARCH FAMILY:
I will use 
1. AIC to compare models, lower is better: -2*log-likelihood + 2*k, where k is the number of parameters in the model. Penalisation parameter of 2, tends to favour more complex than BIC
2. BIC to compare models, lower is better: -2*log-likelihood + k*log(n), where k is the number of parameters and n is the number of observations. Penalisation parameter of log(n)
3. Log-likelihood to compare models, higher is better: it does not penalse model complexity though
4. Persistence of volatility: the sum must always be less than 1, else there is no stationarity in the variance and shocks will not die out. The closer to 1, the more persistent the volatility is, meaning that shocks will take a long time to die out.
5. Residual diagnostics: I will check if the residuals of the model are white noise, as suggested in testing

GLOBAL EVALUATION:
QLIKE? MSE? Diebold-Mariano test? Rolling forecasts?
Benchmark to HAR RV, issue with HAR-RV needs intraday data e.g. every five minutes? 
Benchmark alternatives: EWMA, rolling historical volatility, HAR-RV (if I add source five minute data from LSEG)

We want to see who gets closest to daily realised volatility
----------------------------------
The class needs to be instantiated with data which is a series of prices
'''

'''GARCH-family volatility models.

- apply_garch          : GARCH(1,1), constant mean, t-distributed innovations
- apply_gjr_garch      : GJR-GARCH(1,1,1), leverage effects, t innovations
- calc_realised_garch  : Realised-GARCH(1,1) with Hansen skew-t innovations,
                         analytic log-pdf, JIT-compilable inner loop.
- apply_realised_garch : multi-step forecast from the estimated params

Class is instantiated with both daily-close and intraday price series.
'''
import math
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

# Optional numba acceleration.
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(f):
            return f
        return deco


def _log_returns_pct(prices):
    """Daily/intra log returns in % with NaN/inf rows removed."""
    p = prices.replace(0, np.nan).dropna()
    r = np.log(p / p.shift(1)) * 100
    return r.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# Hansen (1994) skew-t — closed form, no t.cdf / t.pdf
# ---------------------------------------------------------------------------
# Density:
#   f(z; nu, lam) = b * c * (1 + (1/(nu-2)) * ((bz+a)/(1 - lam * sign(bz+a)))^2) ^ (-(nu+1)/2)
# where
#   c   = Γ((nu+1)/2) / (sqrt(pi*(nu-2)) * Γ(nu/2))
#   a   = 4 * lam * c * (nu-2)/(nu-1)
#   b^2 = 1 + 3 lam^2 - a^2
# Constraints: nu > 2, |lam| < 1.
# Implemented as a single JIT'd negative-log-likelihood for Realised-GARCH so
# the whole inner loop compiles to native code.


@njit(cache=True)
def _rgarch_skewt_negloglik(params, r, RV, T):
    mu, omega, beta, gamma, xi, nu, lam = params

    # parameter validity guards (cheap and important — keeps the optimiser
    # from wandering into garbage regions where the density is undefined)
    if nu <= 2.05 or nu > 100.0:
        return 1e10
    if lam <= -0.99 or lam >= 0.99:
        return 1e10
    if omega <= 0.0:
        return 1e10
    if beta + gamma + xi >= 0.999:
        return 1e10

    # Hansen skew-t constants (computed once per evaluation)
    log_c = (math.lgamma((nu + 1.0) / 2.0)
             - 0.5 * math.log(math.pi * (nu - 2.0))
             - math.lgamma(nu / 2.0))
    c     = math.exp(log_c)
    a     = 4.0 * lam * c * (nu - 2.0) / (nu - 1.0)
    b2    = 1.0 + 3.0 * lam * lam - a * a
    if b2 <= 0.0:
        return 1e10
    b     = math.sqrt(b2)
    log_b = math.log(b)
    cutoff = -a / b
    half_nup1 = 0.5 * (nu + 1.0)
    inv_num2  = 1.0 / (nu - 2.0)

    # initial h = sample variance of r (no pandas)
    s = 0.0
    for i in range(T):
        s += r[i]
    rmean = s / T
    s2 = 0.0
    for i in range(T):
        d = r[i] - rmean
        s2 += d * d
    h_prev = s2 / T
    if h_prev < 1e-8:
        h_prev = 1e-8

    ll = 0.0
    for t in range(1, T):
        eps = r[t - 1] - mu
        h = omega + beta * h_prev + gamma * RV[t - 1] + xi * eps * eps
        if h < 1e-8:
            h = 1e-8

        # standardised residual under skew-t
        z = (r[t] - mu) / math.sqrt(h)

        # piecewise scaling factor
        if z < cutoff:
            lam_eff = 1.0 - lam
        else:
            lam_eff = 1.0 + lam
        arg = (b * z + a) / lam_eff

        # log f(z; nu, lam)
        log_fz = log_c + log_b - half_nup1 * math.log(1.0 + inv_num2 * arg * arg)
        # log p(r_t | h_t) = log f(z) - 0.5 log h  (Jacobian)
        ll += log_fz - 0.5 * math.log(h)

        h_prev = h

    return -ll


class GARCHModel:
    def __init__(self, prices_close, prices_intraday):
        self.prices_close = prices_close
        self.prices_intraday = prices_intraday
        self.returns_close    = _log_returns_pct(prices_close)
        self.returns_intraday = _log_returns_pct(prices_intraday)

    # --- standard GARCH (via arch package) ---------------------------------
    def apply_garch(self):
        model = arch_model(
            self.returns_close,
            mean="constant", vol="GARCH",
            p=1, o=0, q=1, dist="t", rescale=False,
        )
        self.fit_baseline = model.fit(disp="off", show_warning=True)
        return self.fit_baseline

    # --- GJR-GARCH (via arch package) --------------------------------------
    def apply_gjr_garch(self):
        model = arch_model(
            self.returns_close,
            mean="constant", vol="GARCH",
            p=1, o=1, q=1, dist="t", rescale=False,
        )
        self.fit_gjr = model.fit(disp="off", show_warning=True)
        return self.fit_gjr

    # --- Realised GARCH with Hansen skew-t innovations ---------------------
    def calc_realised_garch(self, maxiter=200, verbose=True):
        # realised variance from intraday returns
        RS_intra = self.returns_intraday ** 2
        RV = RS_intra.groupby(self.returns_intraday.index.date).sum()

        # align dates
        r_series = self.returns_close.copy()
        r_series.index = pd.to_datetime(r_series.index.date)
        RV.index = pd.to_datetime(RV.index)
        df = pd.concat([r_series, RV], axis=1).dropna()
        df.columns = ["r", "RV"]

        r_arr  = df["r"].values.astype(np.float64)
        RV_arr = df["RV"].values.astype(np.float64)
        T      = len(r_arr)
        if verbose:
            print(f"  [RGARCH skew-t] T={T} days, "
                  f"numba={'on' if _HAS_NUMBA else 'OFF (pip install numba for ~50x speedup)'}")

        # initial values:           mu  omega  beta  gamma   xi    nu   lam
        init_params = np.array([    0.0, 0.05, 0.85, 0.05, 0.05,  8.0,  0.0])
        bounds = [
            (-1.0, 1.0),     # mu
            (1e-6, None),    # omega
            (0.0, 0.999),    # beta
            (0.0, 0.999),    # gamma
            (0.0, 0.999),    # xi
            (2.1, 60.0),     # nu  (df, must be > 2)
            (-0.95, 0.95),   # lam (skew, |lam| < 1)
        ]

        # warm JIT
        if _HAS_NUMBA:
            _rgarch_skewt_negloglik(init_params, r_arr, RV_arr, T)

        n_calls = [0]
        def obj(p):
            n_calls[0] += 1
            return _rgarch_skewt_negloglik(p, r_arr, RV_arr, T)

        res = minimize(obj, init_params, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": maxiter})
        if verbose:
            print(f"  [RGARCH skew-t] done: {n_calls[0]} fn evals, "
                  f"converged={res.success}, -loglik={res.fun:.2f}")
            print(f"  [RGARCH skew-t] params: "
                  f"mu={res.x[0]:.3f}, omega={res.x[1]:.3f}, "
                  f"beta={res.x[2]:.3f}, gamma={res.x[3]:.3f}, "
                  f"xi={res.x[4]:.3f}, nu={res.x[5]:.2f}, lam={res.x[6]:.3f}")

        self.params_ = res.x
        self.realised_garch_data_ = df
        return res

    # --- multi-step variance forecast from estimated params ----------------
    def apply_realised_garch(self, steps=10, refit=True):
        if refit or not hasattr(self, "params_"):
            self.calc_realised_garch()

        mu, omega, beta, gamma, xi, nu, lam = self.params_
        df = self.realised_garch_data_
        r  = df["r"].values
        RV = df["RV"].values
        T  = len(r)

        h = np.zeros(T + steps)
        h[0] = np.var(r)
        for t in range(1, T):
            eps = r[t - 1] - mu
            h[t] = max(omega + beta * h[t - 1] + gamma * RV[t - 1] + xi * eps ** 2,
                       1e-8)
        # out-of-sample: assume future RV ≈ last observed RV, zero shock
        last_rv = RV[-1]
        for t in range(T, T + steps):
            h[t] = max(omega + beta * h[t - 1] + gamma * last_rv, 1e-8)
        return h[T:T + steps]