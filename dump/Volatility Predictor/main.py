'''critical dates where none work: 2008, 2020
next step: add crisis detection layer'''



import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loader import DataLoader
from GARCH import GARCHModel
from HAR import HAR_RVModel, HAR_RV_CJ
from eval import Evaluation
from combine import StudentT_DMA_HAR_RGARCH_VIX


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
TICKER = "SPY.P"
TEST  = ("2024-01-01", "2025-01-01")
TRAIN   = ("2020-01-01", "2024-01-01")

CSVS = {
    "train_close":    "data/SPY_daily_close.csv",
    "train_intraday": "data/SPY_intraday.csv",
    "test_close":     "data/SPY_daily_close_test.csv",
    "test_intraday":  "data/SPY_intraday_test.csv",
    "train_vix":      "data/VIX_daily.csv",
    "test_vix":       "data/VIX_daily_test.csv",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def realised_variance(intraday):
    r = np.log(intraday / intraday.shift(1)) * 100
    rv = (r ** 2).groupby(r.index.date).sum()
    rv.index = pd.to_datetime(rv.index)
    return rv.dropna()


def cache(loader_fn, path, *args):
    if not os.path.exists(path):
        loader_fn(*args).to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")


def recurse_garch(omega, alpha, beta, mu, returns, gamma=0.0):
    r = returns.values
    h = np.empty(len(r))
    h[0] = returns.var()

    for t in range(1, len(r)):
        eps = r[t - 1] - mu
        ind = 1.0 if eps < 0 else 0.0
        h[t] = omega + (alpha + gamma * ind) * eps**2 + beta * h[t - 1]

    return pd.Series(h, index=returns.index)


def recurse_realised_garch(omega, beta, gamma, xi, mu, returns, rv_lag, h_init,
                           floor=1e-6):

    r = returns.values
    r_prev = np.concatenate([[np.nan], r[:-1]])
    rv_prev = rv_lag.values

    h = np.empty(len(r))
    last_h = h_init

    for t in range(len(r)):

        if np.isnan(r_prev[t]) or np.isnan(rv_prev[t]):
            h[t] = np.nan
            continue

        last_h = max(
            omega
            + beta * last_h
            + gamma * rv_prev[t]
            + xi * (r_prev[t] - mu) ** 2,
            floor
        )

        h[t] = last_h

    return pd.Series(h, index=returns.index)


def log_rv(series):
    return np.log(series + 1e-12)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
need_lseg = any(not os.path.exists(p) for p in CSVS.values())
data = DataLoader(open_session=need_lseg)

train_close = cache(data.load_daily_close, CSVS["train_close"], TICKER, *TRAIN)
train_intra = cache(data.load_intraday,    CSVS["train_intraday"], TICKER, *TRAIN)

test_close  = cache(data.load_daily_close, CSVS["test_close"], TICKER, *TEST)
test_intra  = cache(data.load_intraday,    CSVS["test_intraday"], TICKER, *TEST)

train_vix   = cache(data.load_vix, CSVS["train_vix"], *TRAIN)
test_vix    = cache(data.load_vix, CSVS["test_vix"], *TEST)

RV_train = realised_variance(train_intra)
RV_test  = realised_variance(test_intra)

all_RV = pd.concat([RV_train, RV_test]).sort_index().replace(0, np.nan)

full_close = pd.concat([train_close, test_close]).sort_index()
full_ret = np.log(full_close / full_close.shift(1)) * 100
full_ret = full_ret.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# fit models
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
har_model = HAR_RVModel(train_intra).fit_HAR_RV()
print(f"[fit] HAR-RV         {time.perf_counter()-t0:5.2f}s")

t0 = time.perf_counter()
har_cj_obj   = HAR_RV_CJ(train_intra)        # the instance — has build_features_for, predict, _df
har_cj_model = har_cj_obj.fit_HAR_RV_CJ() 
print(f"[fit] HAR-CJ         {time.perf_counter()-t0:5.2f}s")

t0 = time.perf_counter()
g_obj = GARCHModel(train_close, train_intra)
garch_fit = g_obj.apply_garch()
gjr_fit = g_obj.apply_gjr_garch()
g_obj.calc_realised_garch()
print(f"[fit] GARCH family   {time.perf_counter()-t0:5.2f}s")


# ---------------------------------------------------------------------------
# forecasts
# ---------------------------------------------------------------------------
test_dates = RV_test.index

# ---------------- GARCH ----------------
gp = garch_fit.params
garch_h = recurse_garch(gp["omega"], gp["alpha[1]"], gp["beta[1]"], gp["mu"], full_ret)

# ---------------- GJR ----------------
jp = gjr_fit.params
gjr_h = recurse_garch(jp["omega"], jp["alpha[1]"], jp["beta[1]"], jp["mu"],
                      full_ret, gamma=jp["gamma[1]"])

# ---------------- HAR-RV ----------------
X = pd.DataFrame({
    "d": log_rv(all_RV),
    "w": log_rv(all_RV.rolling(5).mean()),
    "m": log_rv(all_RV.rolling(22).mean()),
}).dropna()

X.insert(0, "const", 1.0)
sigma2 = har_model.resid.var(ddof=1)

har_pred = pd.Series(
    np.exp(har_model.predict(X.values) + 0.5 * sigma2),
    index=X.index + pd.offsets.BDay(1)
)


# ---------------- HAR-CJ ----------------

full_intra = pd.concat([train_intra, test_intra]).sort_index()
feats_cj, X_cj = har_cj_obj.build_features_for(full_intra)
har_cj_pred = pd.Series(
    har_cj_obj.predict(X_cj),
    index=feats_cj.index + pd.offsets.BDay(1),
)
# ---------------- VIX (for DMA only) ----------------
vix = pd.concat([train_vix, test_vix]).reindex(all_RV.index).ffill()
vix_var = (vix / np.sqrt(252))**2

# ---------------- Realised GARCH ----------------
p = g_obj.params_
mu, omega, beta, gamma, xi = p[0], p[1], p[2], p[3], p[4]
rv_lag = all_RV.reindex(full_ret.index).ffill().shift(1)
rgarch_h = recurse_realised_garch(
    omega=omega, beta=beta, gamma=gamma, xi=xi, mu=mu,
    returns=full_ret, rv_lag=rv_lag,
    h_init=full_ret.loc[:TRAIN[1]].var(),
)


# ---------------------------------------------------------------------------
# DMA (HAR-CJ + RGARCH + VIX)
# ---------------------------------------------------------------------------
dma_index = har_cj_pred.index.intersection(rgarch_h.index).intersection(vix_var.index).intersection(all_RV.index)

dma = StudentT_DMA_HAR_RGARCH_VIX(
    har_cj=har_cj_pred.reindex(dma_index).ffill().values,
    rgarch=rgarch_h.reindex(dma_index).ffill().values,
    vix=vix_var.reindex(dma_index).ffill().values,
    rv=all_RV.reindex(dma_index).ffill().values
)

dma.fit(decay=0.99, lam=0.97, nu=8)
dma_forecast = pd.Series(dma.forecast(), index=dma_index)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
out = pd.DataFrame({
    "Realised":       RV_test,
    "GARCH":          garch_h.reindex(test_dates, method="ffill"),
    "GJR-GARCH":      gjr_h.reindex(test_dates, method="ffill"),
    "HAR-RV":         har_pred.reindex(test_dates, method="ffill"),
    "HAR-CJ":         har_cj_pred.reindex(test_dates, method="ffill"),
    "Realised GARCH": rgarch_h.reindex(test_dates, method="ffill"),
    "DMA":            dma_forecast.reindex(test_dates, method="ffill"),
}).dropna()

print(f"\nAligned {len(out)} test dates")

print("\n=== Forecast distribution ===")
print(out.describe().loc[["min", "50%", "max"]].round(3))

ev = Evaluation()

print("\n=== Out-of-sample accuracy ===")
for col in ["GARCH","GJR-GARCH","HAR-RV","HAR-CJ","Realised GARCH","DMA"]:
    q = np.mean(ev.QLIKE(out[col], out["Realised"]))
    m = ev.MSE(out[col], out["Realised"])
    print(f"{col:15s} QLIKE={q:.5f} MSE={m:.5f}")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(12,6))
plt.plot(out.index, out["Realised"], color="black", lw=2, label="Realised")

for col in ["HAR-RV","HAR-CJ","Realised GARCH","DMA", "GARCH","GJR-GARCH"]:
    plt.plot(out.index, out[col], alpha=0.8, label=col)

plt.title(f"{TICKER} Volatility Forecast Comparison")
plt.legend()
plt.tight_layout()
plt.show()