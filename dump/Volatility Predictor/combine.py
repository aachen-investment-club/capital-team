'''

for this I will use a logscore (Bayesian) DMS to weight dynamically- dynamic model averaging system

Idea: 
1. Bayesian dynamic model averaging
2. logscored weighting
3. student-t distribution to capture fat tails
4. adaptive variance (EWMA) to capture volatility clustering

'''

import numpy as np
from scipy.special import gammaln


class StudentT_DMA_HAR_RGARCH_VIX:

    def __init__(self, har_cj, rgarch, vix, rv):

        self.har = np.asarray(har_cj, dtype=float)
        self.rgarch = np.asarray(rgarch, dtype=float)
        self.vix = np.asarray(vix, dtype=float)
        self.rv = np.asarray(rv, dtype=float)

    # --------------------------------------------------
    # stable student-t log pdf
    # --------------------------------------------------
    def _logpdf(self, y, mu, sigma2, nu):

        eps = 1e-12
        sigma2 = max(sigma2, eps)

        z = (y - mu) / np.sqrt(sigma2)

        return (
            gammaln((nu + 1) / 2)
            - gammaln(nu / 2)
            - 0.5 * np.log((nu - 2) * np.pi * sigma2 + eps)
            - ((nu + 1) / 2) * np.log1p((z ** 2) / (nu - 2 + eps))
        )

    # --------------------------------------------------
    # fit DMA
    # --------------------------------------------------
    def fit(self, decay=0.98, lam=0.97, sigma_init=0.1, nu=8):

        T = len(self.rv)

        w = np.zeros((T, 3))
        w[0] = np.array([1/3, 1/3, 1/3], dtype=float)

        sigma2 = np.zeros(T)
        sigma2[0] = sigma_init

        y = np.log(self.rv + 1e-12)

        for t in range(1, T):

            mu_h = np.log(self.har[t] + 1e-12)
            mu_g = np.log(self.rgarch[t] + 1e-12)

            # VIX → variance proxy (IMPORTANT: already variance space)
            mu_v = np.log(self.vix[t] + 1e-12)

            s2 = sigma2[t - 1] + 1e-12

            f_h = np.exp(self._logpdf(y[t], mu_h, s2, nu))
            f_g = np.exp(self._logpdf(y[t], mu_g, s2, nu))
            f_v = np.exp(self._logpdf(y[t], mu_v, s2, nu))

            prev = w[t - 1]

            w_h = (prev[0] ** decay) * f_h
            w_g = (prev[1] ** decay) * f_g
            w_v = (prev[2] ** decay) * f_v

            denom = w_h + w_g + w_v + 1e-12

            w[t, 0] = w_h / denom
            w[t, 1] = w_g / denom
            w[t, 2] = w_v / denom

            mu_ens = (
                w[t, 0] * mu_h +
                w[t, 1] * mu_g +
                w[t, 2] * mu_v
            )

            err = y[t] - mu_ens
            sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * err ** 2

        self.w_ = w
        self.sigma2_ = sigma2

        return w

    # --------------------------------------------------
    # forecast
    # --------------------------------------------------
    def forecast(self):

        eps = 1e-12

        mu_h = np.log(self.har + eps)
        mu_g = np.log(self.rgarch + eps)
        mu_v = np.log(self.vix + eps)

        w = self.w_

        log_pred = (
            w[:, 0] * mu_h +
            w[:, 1] * mu_g +
            w[:, 2] * mu_v
        )

        return np.exp(log_pred)