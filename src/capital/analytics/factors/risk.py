"""
Factor covariance, specific risk, and portfolio risk decomposition.

The exposure matrix says what the portfolio is *tilted* towards; this module
turns those tilts into risk, which is what actually decides position sizing:

    sigma_p^2 = x' F x  +  sum_i w_i^2 s_i^2
                 |            |
                 |            +-- specific (idiosyncratic) variance
                 +-- factor (systematic) variance,  x = X'w

Three refinements separate a usable risk model from a naive sample covariance:

- **Two half-lives.** Volatility moves faster than correlation, so variances are
  estimated with a short half-life (~90d) and the correlation matrix with a long
  one (~252d), then recombined. A single half-life either lags a volatility
  spike or produces a jumpy, unusable correlation matrix.
- **Newey-West.** Factor returns are serially correlated (non-synchronous
  trading, slow information diffusion). Ignoring that understates long-horizon
  risk; a Bartlett-kernel lag adjustment corrects it.
- **Bayesian shrinkage of specific risk.** A security's own residual history is
  a noisy estimate of its idiosyncratic volatility, so it is pulled toward the
  average of its size decile in proportion to how far out and how unreliable it
  looks.
"""
import numpy as np
import pandas as pd

_EPS = 1e-12

PERIODS_PER_YEAR = {"B": 252, "D": 252, "W-FRI": 52, "W": 52, "ME": 12, "M": 12}


def periods_per_year(frequency: str) -> int:
    return PERIODS_PER_YEAR.get(frequency, 252)


# ── Factor covariance ─────────────────────────────────────────────────────────

def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    lam = 0.5 ** (1.0 / max(halflife, 1.0))
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum()


def _weighted_cov(X: np.ndarray, w: np.ndarray, lag: int = 0) -> np.ndarray:
    """Exponentially weighted (auto)covariance at `lag`, of a T x K panel."""
    mean = w @ X
    Z = X - mean
    if lag == 0:
        return (Z * w[:, None]).T @ Z
    wl = w[lag:] / w[lag:].sum()
    return (Z[lag:] * wl[:, None]).T @ Z[:-lag]


def factor_covariance(factor_returns: pd.DataFrame, var_halflife: float = 90,
                      corr_halflife: float = 252, nw_lags: int = 5,
                      frequency: str = "W-FRI") -> pd.DataFrame:
    """Annualised factor covariance matrix (two half-lives + Newey-West)."""
    F = factor_returns.dropna(how="all", axis=1).fillna(0.0)
    if F.shape[0] < 8 or F.shape[1] == 0:
        return pd.DataFrame(index=F.columns, columns=F.columns, dtype=float)
    X = F.to_numpy(dtype=float)
    T = len(X)
    scale = periods_per_year(frequency)
    lags = int(min(nw_lags, max(T // 4, 0)))

    def nw(halflife: float) -> np.ndarray:
        w = _ewma_weights(T, halflife)
        S = _weighted_cov(X, w, 0)
        for lag in range(1, lags + 1):
            G = _weighted_cov(X, w, lag)
            S = S + (1 - lag / (lags + 1)) * (G + G.T)
        return S

    S_var, S_corr = nw(var_halflife), nw(corr_halflife)
    d_var = np.sqrt(np.clip(np.diag(S_var), _EPS, None))
    d_corr = np.sqrt(np.clip(np.diag(S_corr), _EPS, None))
    corr = S_corr / np.outer(d_corr, d_corr)
    corr = np.clip((corr + corr.T) / 2, -1, 1)
    np.fill_diagonal(corr, 1.0)

    cov = corr * np.outer(d_var, d_var) * scale
    return pd.DataFrame(_nearest_psd(cov), index=F.columns, columns=F.columns)


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues — the two-half-life recombination and the
    Newey-West sum can each push the matrix slightly indefinite."""
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    if (vals > 0).all():
        return cov
    floor = max(vals.max(), _EPS) * 1e-8
    return vecs @ np.diag(np.clip(vals, floor, None)) @ vecs.T


def factor_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), _EPS, None))
    return pd.DataFrame(cov.to_numpy() / np.outer(d, d), index=cov.index, columns=cov.columns)


# ── Specific risk ─────────────────────────────────────────────────────────────

def specific_risk(specific_returns: pd.DataFrame, halflife: float = 90,
                  size_exposure: pd.Series | None = None, frequency: str = "W-FRI",
                  n_buckets: int = 10, q: float = 0.1) -> pd.DataFrame:
    """Annualised idiosyncratic volatility per security, shrunk by size bucket.

    Returns raw, shrunk and final volatilities plus the observation count, so the
    UI can show how much a thin-history name was pulled toward its peers.
    """
    U = specific_returns
    if U.empty:
        return pd.DataFrame(columns=["sigma_raw", "sigma", "shrink_weight", "n_obs"])
    scale = np.sqrt(periods_per_year(frequency))
    w = _ewma_weights(len(U), halflife)
    n_obs = U.notna().sum()

    Uf = U.to_numpy(dtype=float)
    mask = np.isfinite(Uf)
    Wm = np.where(mask, w[:, None], 0.0)
    Wm = Wm / np.where(Wm.sum(0) > 0, Wm.sum(0), np.nan)
    mean = np.nansum(np.where(mask, Uf, 0.0) * Wm, axis=0)
    var = np.nansum(np.where(mask, (Uf - mean) ** 2, 0.0) * Wm, axis=0)
    sigma_raw = pd.Series(np.sqrt(np.clip(var, 0, None)) * scale, index=U.columns)
    sigma_raw = sigma_raw.where(n_obs >= 8)

    out = pd.DataFrame({"sigma_raw": sigma_raw, "n_obs": n_obs})
    if size_exposure is None or size_exposure.dropna().empty:
        out["sigma"] = sigma_raw
        out["shrink_weight"] = 0.0
        return out

    size = size_exposure.reindex(out.index)
    buckets = pd.qcut(size.rank(method="first"), min(n_buckets, max(size.notna().sum() // 5, 1)),
                      labels=False, duplicates="drop")
    prior = buckets.map(sigma_raw.groupby(buckets).mean())
    delta = buckets.map(sigma_raw.groupby(buckets).std())
    gap = (sigma_raw - prior).abs()
    v = (q * gap) / (delta.replace(0, np.nan) + q * gap)
    v = v.clip(0, 1).fillna(0.0)

    out["shrink_weight"] = v
    out["sigma"] = (v * prior + (1 - v) * sigma_raw).fillna(sigma_raw).fillna(sigma_raw.median())
    return out


# ── Portfolio risk decomposition ──────────────────────────────────────────────

def portfolio_exposure(weights: pd.Series, exposures: pd.DataFrame) -> pd.Series:
    """x = X'w over the securities the model actually covers, renormalised.

    Renormalising matters: if 15% of the book is cash or an uncovered security,
    the honest statement is "the covered 85% has this tilt", not a tilt diluted
    by a silent zero.
    """
    common = exposures.index.intersection(weights.dropna().index)
    if common.empty:
        return pd.Series(dtype=float)
    w = weights.loc[common]
    total = w.sum()
    if abs(total) < _EPS:
        return pd.Series(dtype=float)
    return exposures.loc[common].fillna(0.0).T.dot(w / total)


def decompose_risk(weights: pd.Series, exposures: pd.DataFrame, cov: pd.DataFrame,
                   spec_sigma: pd.Series, groups: dict[str, str] | None = None) -> dict:
    """Split portfolio volatility into factor and specific parts.

    `contribution` is the standard x_k (F x)_k / sigma decomposition: it sums
    exactly to the factor volatility, so the numbers can be read as "this factor
    is worth N volatility points to us" rather than as an unnormalised score.
    """
    common = exposures.index.intersection(weights.dropna().index)
    if common.empty:
        return {}
    w = weights.loc[common]
    w = w / w.sum() if abs(w.sum()) > _EPS else w

    factors = [f for f in cov.columns if f in exposures.columns]
    X = exposures.loc[common, factors].fillna(0.0)
    x = X.T.dot(w)
    F = cov.loc[factors, factors].to_numpy(dtype=float)

    Fx = F @ x.to_numpy(dtype=float)
    factor_var = float(x.to_numpy(dtype=float) @ Fx)
    sig = spec_sigma.reindex(common)
    sig = sig.fillna(sig.median())
    specific_var = float(((w ** 2) * (sig ** 2)).sum())
    total_var = max(factor_var + specific_var, _EPS)
    total_vol = float(np.sqrt(total_var))

    contribution = pd.Series(x.to_numpy(dtype=float) * Fx / total_vol, index=factors)
    marginal = pd.Series(Fx / total_vol, index=factors)

    by_group = pd.Series(dtype=float)
    if groups:
        grp = pd.Series({f: groups.get(f, "Other") for f in factors})
        by_group = contribution.groupby(grp).sum()

    return {
        "exposure": x,
        "total_vol": total_vol,
        "factor_vol": float(np.sqrt(max(factor_var, 0.0))),
        "specific_vol": float(np.sqrt(max(specific_var, 0.0))),
        "factor_share": factor_var / total_var,
        "specific_share": specific_var / total_var,
        "contribution": contribution,
        "marginal": marginal,
        "pct_of_variance": pd.Series(x.to_numpy(dtype=float) * Fx / total_var, index=factors),
        "by_group": by_group,
        "n_covered": int(len(common)),
        "weight_covered": float(weights.loc[common].sum()),
    }


def security_contributions(weights: pd.Series, exposures: pd.DataFrame, cov: pd.DataFrame,
                           spec_sigma: pd.Series) -> pd.DataFrame:
    """Each position's contribution to portfolio volatility.

    A position's contribution is its weight times its covariance with the
    portfolio, over portfolio volatility — so the column sums to sigma_p exactly.
    That is the number to size on: a 3% position in a name that co-moves with
    everything else can carry more risk than an 8% position that does not.
    """
    common = exposures.index.intersection(weights.dropna().index)
    if common.empty:
        return pd.DataFrame(columns=["weight", "factor_ctr", "specific_ctr", "total_ctr", "pct_of_risk"])
    w = weights.loc[common]
    w = w / w.sum() if abs(w.sum()) > _EPS else w
    factors = [f for f in cov.columns if f in exposures.columns]
    X = exposures.loc[common, factors].fillna(0.0)
    F = cov.loc[factors, factors].to_numpy(dtype=float)
    x = X.T.dot(w).to_numpy(dtype=float)

    sig = spec_sigma.reindex(common)
    sig = sig.fillna(sig.median())
    cov_with_port = X.to_numpy(dtype=float) @ (F @ x)          # factor part
    spec_part = (w * sig ** 2).to_numpy(dtype=float)           # specific part
    total_var = float(x @ (F @ x) + ((w ** 2) * (sig ** 2)).sum())
    sigma_p = float(np.sqrt(max(total_var, _EPS)))

    out = pd.DataFrame({
        "weight": w,
        "factor_ctr": w.to_numpy(dtype=float) * cov_with_port / sigma_p,
        "specific_ctr": w.to_numpy(dtype=float) * spec_part / sigma_p,
        "specific_vol": sig,
    }, index=common)
    out["total_ctr"] = out["factor_ctr"] + out["specific_ctr"]
    out["pct_of_risk"] = out["total_ctr"] / sigma_p
    return out.sort_values("total_ctr", ascending=False)


def active_risk(weights: pd.Series, benchmark: pd.Series, exposures: pd.DataFrame,
                cov: pd.DataFrame, spec_sigma: pd.Series,
                groups: dict[str, str] | None = None) -> dict:
    """Tracking error decomposition — the same maths on active weights w - b."""
    idx = weights.index.union(benchmark.index)
    active = weights.reindex(idx).fillna(0.0) - benchmark.reindex(idx).fillna(0.0)
    common = exposures.index.intersection(idx)
    if common.empty:
        return {}
    a = active.loc[common]
    factors = [f for f in cov.columns if f in exposures.columns]
    X = exposures.loc[common, factors].fillna(0.0)
    x = X.T.dot(a)
    F = cov.loc[factors, factors].to_numpy(dtype=float)
    Fx = F @ x.to_numpy(dtype=float)
    factor_var = float(x.to_numpy(dtype=float) @ Fx)
    sig = spec_sigma.reindex(common)
    sig = sig.fillna(sig.median())
    specific_var = float(((a ** 2) * (sig ** 2)).sum())
    te = float(np.sqrt(max(factor_var + specific_var, 0.0)))
    contribution = pd.Series(x.to_numpy(dtype=float) * Fx / max(te, _EPS), index=factors)
    by_group = pd.Series(dtype=float)
    if groups:
        grp = pd.Series({f: groups.get(f, "Other") for f in factors})
        by_group = contribution.groupby(grp).sum()
    return {"active_exposure": x, "tracking_error": te, "contribution": contribution,
            "by_group": by_group,
            "factor_share": factor_var / max(factor_var + specific_var, _EPS)}


# ── What-if ───────────────────────────────────────────────────────────────────

def apply_trades(weights: pd.Series, trades: dict[str, float],
                 funding: str = "pro_rata") -> pd.Series:
    """Weights after a set of weight deltas, keeping the book fully invested.

    `funding` decides where the money comes from: "pro_rata" scales every other
    position (the honest default — you cannot add 5% of a new name without
    selling something), "cash" lets the total drift (models a cash draw-down).
    """
    new = weights.copy().astype(float)
    for sid, delta in trades.items():
        new[sid] = float(new.get(sid, 0.0)) + float(delta)
    new = new[new.abs() > 1e-9]
    if funding == "cash":
        return new
    net = sum(float(d) for d in trades.values())
    if abs(net) < _EPS:
        return new
    others = new.index.difference(pd.Index(list(trades)))
    base = new.loc[others].sum()
    if base > _EPS:
        new.loc[others] = new.loc[others] * (1 - net / base)
    return new


def compare_portfolios(before: pd.Series, after: pd.Series, exposures: pd.DataFrame,
                       cov: pd.DataFrame, spec_sigma: pd.Series,
                       groups: dict[str, str] | None = None) -> dict:
    """Before/after risk and exposure, plus the deltas the trade actually caused."""
    b = decompose_risk(before, exposures, cov, spec_sigma, groups)
    a = decompose_risk(after, exposures, cov, spec_sigma, groups)
    if not b or not a:
        return {"before": b, "after": a}
    idx = b["exposure"].index.union(a["exposure"].index)
    return {
        "before": b, "after": a,
        "exposure_delta": a["exposure"].reindex(idx).fillna(0) - b["exposure"].reindex(idx).fillna(0),
        "contribution_delta": a["contribution"].reindex(idx).fillna(0)
                              - b["contribution"].reindex(idx).fillna(0),
        "vol_delta": a["total_vol"] - b["total_vol"],
        "turnover": float((after.reindex(idx.union(before.index)).fillna(0)
                           - before.reindex(idx.union(before.index)).fillna(0)).abs().sum() / 2),
    }
