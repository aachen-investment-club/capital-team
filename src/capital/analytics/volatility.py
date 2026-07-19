"""
GARCH-family volatility modelling — fitting, backtesting, evaluation, forecasting.
Only models that use daily close data (no intraday): GARCH(1,1) and GJR-GARCH(1,1,1).
"""
import math
import warnings
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_MODELS = ("GARCH", "GJR-GARCH")
_ANN = 252  # trading days per year


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_returns_pct(prices: pd.Series) -> pd.Series:
    """Log returns in % from a price series, NaN/inf cleaned."""
    p = prices.replace(0, np.nan).dropna()
    r = np.log(p / p.shift(1)) * 100
    return r.replace([np.inf, -np.inf], np.nan).dropna()


def proxy_rv(returns_pct: pd.Series) -> pd.Series:
    """Squared daily log-return (%²) as a noisy proxy for daily realised variance."""
    return returns_pct ** 2


def ann_vol_from_var(var_series: pd.Series) -> pd.Series:
    """Convert conditional variance (%²/day) → annualised volatility (%)."""
    return np.sqrt(var_series.clip(lower=0) * _ANN)


# ── GARCH recursion (avoids re-fitting on test data) ─────────────────────────

def _recurse(omega: float, alpha: float, beta: float, mu: float,
             returns: pd.Series, gamma: float = 0.0) -> pd.Series:
    """
    Roll GARCH / GJR-GARCH variance equation over `returns`.
    Initial h set to the sample variance.
    """
    r = returns.values
    h = np.empty(len(r))
    h[0] = max(float(np.var(r)), 1e-8)
    for t in range(1, len(r)):
        eps = r[t - 1] - mu
        ind = 1.0 if eps < 0.0 else 0.0
        h[t] = max(omega + (alpha + gamma * ind) * eps ** 2 + beta * h[t - 1], 1e-8)
    return pd.Series(h, index=returns.index)


# ── Model fitting ─────────────────────────────────────────────────────────────

def _fit_arch(returns_pct: pd.Series, gjr: bool):
    """Fit GARCH(1,1) or GJR-GARCH(1,1,1) with Student-t innovations via arch."""
    o = 1 if gjr else 0
    mdl = arch_model(returns_pct, mean="constant", vol="GARCH",
                     p=1, o=o, q=1, dist="t", rescale=False)
    return mdl.fit(disp="off", show_warning=False)


def fit_models(returns_pct: pd.Series, model_names: tuple | list = ALL_MODELS) -> dict:
    """
    Fit selected models on the entire return series.

    Returns
    -------
    dict: {model_name: {"fit": ARCHModelResult, "params": dict}}
    """
    results = {}
    for name in model_names:
        if name not in ALL_MODELS:
            continue
        gjr = name == "GJR-GARCH"
        fit = _fit_arch(returns_pct, gjr=gjr)
        p = fit.params
        results[name] = {
            "fit": fit,
            "params": {
                "mu":    float(p.get("mu", p.get("Const", 0.0))),
                "omega": float(p["omega"]),
                "alpha": float(p["alpha[1]"]),
                "beta":  float(p["beta[1]"]),
                "gamma": float(p.get("gamma[1]", 0.0)),
            },
        }
    return results


# ── Backtest: fit on train, recurse over full series, slice test window ───────

def run_backtest(
    returns_pct: pd.Series,
    backtest_from,
    model_names: tuple | list = ALL_MODELS,
) -> dict:
    """
    Fit each model on returns before `backtest_from`, then recurse the
    variance equation over the full series and return the test-window slice.

    Returns
    -------
    dict: {
        "test_returns":   pd.Series   (returns in test window),
        "proxy_rv":       pd.Series   (squared returns as proxy RV),
        "variances":      {model: pd.Series of conditional variance in test window},
    }
    """
    bt = pd.Timestamp(backtest_from)
    train = returns_pct[returns_pct.index < bt]
    test  = returns_pct[returns_pct.index >= bt]

    if len(train) < 10:
        raise ValueError(
            f"Only {len(train)} training observations before {backtest_from}. "
            "Need at least 10 — try a later backtest date."
        )
    if len(test) < 2:
        raise ValueError("Fewer than 2 test observations. Try an earlier backtest date.")

    variances = {}
    for name in model_names:
        if name not in ALL_MODELS:
            continue
        gjr = name == "GJR-GARCH"
        fit  = _fit_arch(train, gjr=gjr)
        p    = fit.params
        mu    = float(p.get("mu", p.get("Const", 0.0)))
        omega = float(p["omega"])
        alpha = float(p["alpha[1]"])
        beta  = float(p["beta[1]"])
        gamma = float(p.get("gamma[1]", 0.0))

        full_h = _recurse(omega, alpha, beta, mu, returns_pct, gamma=gamma)
        variances[name] = full_h.reindex(test.index)

    return {
        "test_returns": test,
        "proxy_rv":     proxy_rv(test),
        "variances":    variances,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def _qlike(forecast: np.ndarray, realised: np.ndarray) -> np.ndarray:
    """Element-wise QLIKE loss: log(h) + rv/h."""
    h = np.clip(forecast, 1e-12, None)
    return np.log(h) + realised / h


def _mse(forecast: np.ndarray, realised: np.ndarray) -> float:
    return float(np.mean((forecast - realised) ** 2))


def _diebold_mariano(f1: np.ndarray, f2: np.ndarray, rv: np.ndarray) -> tuple[float, float]:
    """DM test comparing f1 vs f2 using QLIKE loss. H0: equal predictive accuracy."""
    d = _qlike(f1, rv) - _qlike(f2, rv)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    stat = np.mean(d) / (np.std(d, ddof=1) / math.sqrt(n))
    p    = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return float(stat), float(p)


def evaluate_models(
    variances: dict,
    proxy_rv_series: pd.Series,
) -> pd.DataFrame:
    """
    Compute MSE and mean QLIKE for each model forecast vs proxy RV.
    Returns a ranked DataFrame (best first).

    Columns: Model, MSE, QLIKE (mean), Rank
    If exactly two models, adds DM_stat and DM_p columns.
    """
    common = proxy_rv_series.dropna().index
    rows = []
    arrays = {}
    for name, h_series in variances.items():
        h   = h_series.reindex(common).ffill().dropna()
        rv  = proxy_rv_series.reindex(h.index)
        mask = h.notna() & rv.notna() & (h > 0)
        h_   = h[mask].values
        rv_  = rv[mask].values
        mse  = _mse(h_, rv_)
        qlike_mean = float(np.mean(_qlike(h_, rv_)))
        rows.append({"Model": name, "MSE": mse, "QLIKE": qlike_mean})
        arrays[name] = (h_, rv_)

    df = pd.DataFrame(rows).sort_values("QLIKE").reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))

    # Diebold-Mariano between the two ranked models (best vs second-best)
    if len(arrays) == 2:
        names = list(arrays.keys())
        h1, rv1 = arrays[names[0]]
        h2, _   = arrays[names[1]]
        n = min(len(h1), len(h2))
        stat, p = _diebold_mariano(h1[:n], h2[:n], rv1[:n])
        df["DM stat"] = [stat if i == 0 else float("nan") for i in range(len(df))]
        df["DM p-val"] = [p    if i == 0 else float("nan") for i in range(len(df))]

    return df


# ── Forward forecast ──────────────────────────────────────────────────────────

def forecast_ahead(
    returns_pct: pd.Series,
    model_names: tuple | list = ALL_MODELS,
    steps: int = 30,
) -> dict:
    """
    Fit each model on the full return series and produce `steps` days of
    out-of-sample conditional variance forecasts.

    Returns
    -------
    dict: {
        "history":   {model: pd.Series of in-sample conditional var (last 90 days)},
        "forecast":  {model: pd.Series indexed by future business dates},
    }
    """
    fitted = fit_models(returns_pct, model_names)

    history = {}
    forecast = {}
    last_date = returns_pct.index[-1]
    future_idx = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    for name, res in fitted.items():
        fit = res["fit"]
        # In-sample conditional variance (last 90 obs for chart clarity)
        cond_var = pd.Series(
            fit.conditional_volatility ** 2,
            index=returns_pct.index[-len(fit.conditional_volatility):]
        )
        history[name] = cond_var.tail(90)

        # Multi-step forecast using arch
        fc = fit.forecast(horizon=steps, reindex=False)
        # fc.variance is DataFrame shape (1, steps); first row = forecasts h+1..h+steps
        fvar = fc.variance.iloc[-1].values
        forecast[name] = pd.Series(fvar, index=future_idx)

    return {"history": history, "forecast": forecast}
