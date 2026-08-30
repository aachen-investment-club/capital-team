"""
Factor robustness diagnostics.

A factor that fits the past is easy; a factor you can size a position on is not.
These are the standard tests for telling them apart:

- **Information coefficient** — the cross-sectional rank correlation between an
  exposure today and the return that follows. Its *consistency* (IC / std(IC),
  the "IC IR") matters far more than its average: a factor with IC 0.03 every
  month beats one that averages 0.08 by being right twice and catastrophic once.
- **Decay** — IC measured at 1, 5, 21, 63, 126 and 252 days. A signal that only
  works at one day is a trading-cost problem, not an investment factor.
- **Quantile spread** — sort the universe into buckets by exposure and hold each.
  A real factor is roughly monotone across buckets; a factor that only works in
  the extreme bucket is usually one crowded trade or a handful of small caps.
- **Sub-period stability** — the same statistics year by year. Almost every
  factor "works" over a full sample; few work in most of its calendar years.
- **Exposure persistence** — how much a security's exposure moves period to
  period, which sets the turnover cost of trading the factor at all.
"""
import numpy as np
import pandas as pd

from capital.analytics.factors.risk import periods_per_year

_EPS = 1e-12


# ── Information coefficient ───────────────────────────────────────────────────

def _rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Row-wise Spearman correlation of two aligned panels (one value per date)."""
    b = b.reindex(index=a.index, columns=a.columns)
    valid = a.notna() & b.notna()
    ra = a.where(valid).rank(axis=1)
    rb = b.where(valid).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    out = num / den.replace(0, np.nan)
    return out.where(valid.sum(axis=1) >= 10)


def information_coefficients(style_panels: dict[str, pd.DataFrame],
                             forward: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """IC summary per (factor, horizon) plus the base-horizon IC time series."""
    base_h = min(forward) if forward else None
    rows, series = [], {}
    for key, panel in style_panels.items():
        for horizon, fwd in sorted(forward.items()):
            ic = _rank_corr(panel, fwd).dropna()
            if ic.empty:
                continue
            mean, sd = float(ic.mean()), float(ic.std())
            n = len(ic)
            rows.append({
                "factor": key, "horizon": horizon,
                "ic_mean": mean, "ic_std": sd,
                "ic_ir": mean / sd if sd > _EPS else np.nan,
                "t_stat": mean / (sd / np.sqrt(n)) if sd > _EPS else np.nan,
                "hit_rate": float((np.sign(ic) == np.sign(mean)).mean()),
                "n_periods": n,
            })
            if horizon == base_h:
                series[key] = ic
    summary = pd.DataFrame(rows)
    return summary, (pd.DataFrame(series) if series else pd.DataFrame())


# ── Quantile portfolios ───────────────────────────────────────────────────────

def quantile_spreads(style_panels: dict[str, pd.DataFrame], fwd: pd.DataFrame,
                     caps: pd.DataFrame, n_quantiles: int = 5,
                     frequency: str = "W-FRI") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap-weighted mean forward return per exposure quantile, per factor.

    Cap-weighting inside each bucket stops the result from being a small-cap
    story: an equal-weighted top bucket is dominated by names we could never
    take a position in at our size.
    """
    scale = periods_per_year(frequency)
    means, spread_series = [], {}
    for key, panel in style_panels.items():
        aligned_fwd = fwd.reindex(index=panel.index, columns=panel.columns)
        aligned_cap = caps.reindex(index=panel.index, columns=panel.columns)
        valid = panel.notna() & aligned_fwd.notna() & aligned_cap.notna()
        ranks = panel.where(valid).rank(axis=1, pct=True)
        bucket_returns: dict[int, pd.Series] = {}
        for q in range(n_quantiles):
            lo, hi = q / n_quantiles, (q + 1) / n_quantiles
            sel = (ranks > lo) & (ranks <= hi) if q else (ranks >= 0) & (ranks <= hi)
            w = aligned_cap.where(sel & valid)
            w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
            bucket_returns[q] = (aligned_fwd.where(sel & valid) * w).sum(axis=1, min_count=1)
        bucket = pd.DataFrame(bucket_returns).dropna(how="all")
        if bucket.empty:
            continue
        spread = bucket[n_quantiles - 1] - bucket[0]
        spread_series[key] = spread
        for q in range(n_quantiles):
            means.append({"factor": key, "quantile": f"Q{q + 1}", "q_order": q + 1,
                          "mean_return": float(bucket[q].mean()) * scale,
                          "vol": float(bucket[q].std()) * np.sqrt(scale)})
        means.append({"factor": key, "quantile": "Top-Bottom", "q_order": n_quantiles + 1,
                      "mean_return": float(spread.mean()) * scale,
                      "vol": float(spread.std()) * np.sqrt(scale)})
    return pd.DataFrame(means), (pd.DataFrame(spread_series) if spread_series else pd.DataFrame())


# ── Factor return statistics ──────────────────────────────────────────────────

def factor_performance(factor_returns: pd.DataFrame, frequency: str = "W-FRI") -> pd.DataFrame:
    """Annualised return / volatility / t-statistic / drawdown per factor."""
    scale = periods_per_year(frequency)
    rows = []
    for col in factor_returns.columns:
        s = factor_returns[col].dropna()
        if len(s) < 4:
            continue
        mean, sd, n = float(s.mean()), float(s.std()), len(s)
        cum = (1 + s).cumprod()
        rows.append({
            "factor": col,
            "ann_return": mean * scale,
            "ann_vol": sd * np.sqrt(scale),
            "sharpe": (mean / sd) * np.sqrt(scale) if sd > _EPS else np.nan,
            "t_stat": mean / (sd / np.sqrt(n)) if sd > _EPS else np.nan,
            "hit_rate": float((s > 0).mean()),
            "max_drawdown": float((cum / cum.cummax() - 1).min()),
            "n_periods": n,
        })
    return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()


def subperiod_returns(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Compounded factor return per calendar year — the stability check that
    matters most, because it is the one a full-sample average hides."""
    if factor_returns.empty:
        return pd.DataFrame()
    return (1 + factor_returns.fillna(0.0)).groupby(factor_returns.index.year).prod() - 1


def cumulative_factor_returns(factor_returns: pd.DataFrame) -> pd.DataFrame:
    return (1 + factor_returns.fillna(0.0)).cumprod() - 1


# ── Exposure persistence and collinearity ─────────────────────────────────────

def exposure_persistence(style_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Period-over-period rank correlation and mean absolute change per factor.

    High persistence = a cheap factor to hold. Low persistence = the exposure is
    mostly noise, or the factor is a trading signal that will be eaten by costs.
    """
    rows = []
    for key, panel in style_panels.items():
        if len(panel) < 3:
            continue
        auto = _rank_corr(panel.iloc[1:], panel.shift(1).iloc[1:]).dropna()
        delta = panel.diff().abs().mean(axis=1).dropna()
        rows.append({
            "factor": key,
            "rank_autocorr": float(auto.mean()) if len(auto) else np.nan,
            "mean_abs_change": float(delta.mean()) if len(delta) else np.nan,
            "half_life_periods": (np.log(0.5) / np.log(max(float(auto.mean()), 1e-6))
                                  if len(auto) and 0 < float(auto.mean()) < 1 else np.nan),
        })
    return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()


def variance_inflation(exposures: pd.DataFrame, factors: list[str]) -> pd.Series:
    """VIF per style factor on the latest cross-section.

    Above ~5 the factor's return is not separately identified: whatever the
    regression assigns to it could as easily belong to the factors it overlaps.
    """
    cols = [f for f in factors if f in exposures.columns]
    X = exposures[cols].dropna()
    if len(X) < len(cols) + 5:
        return pd.Series(dtype=float)
    out = {}
    values = X.to_numpy(dtype=float)
    for i, col in enumerate(cols):
        y = values[:, i]
        others = np.column_stack([np.ones(len(values)), np.delete(values, i, axis=1)])
        try:
            coef, *_ = np.linalg.lstsq(others, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        ss_res = float(((y - others @ coef) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > _EPS else 0.0
        out[col] = 1.0 / max(1 - r2, 1e-6)
    return pd.Series(out)


# ── Per-security diagnostics ──────────────────────────────────────────────────

def security_exposure_stability(style_panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Mean, volatility and autocorrelation of every security's own exposures.

    This is the per-name robustness view: a security whose Momentum exposure
    swings +-2 z-scores a quarter is not "a momentum name", it is a name that
    happened to screen that way on the day you looked.
    """
    frames = []
    for key, panel in style_panels.items():
        if panel.empty:
            continue
        lag = panel.shift(1)
        centred, lagged = panel - panel.mean(), lag - lag.mean()
        num = (centred * lagged).sum()
        den = np.sqrt((centred ** 2).sum() * (lagged ** 2).sum())
        frames.append(pd.DataFrame({
            "factor": key,
            "mean": panel.mean(),
            "std": panel.std(),
            "min": panel.min(),
            "max": panel.max(),
            "last": panel.iloc[-1],
            "autocorr": (num / den.replace(0, np.nan)),
            "n_obs": panel.notna().sum(),
        }))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    out.index.name = "security_id"
    return out.reset_index()


def security_fit(returns: pd.DataFrame, specific_returns: pd.DataFrame,
                 frequency: str = "W-FRI") -> pd.DataFrame:
    """Share of each security's return variance the model explains, with
    annualised total and specific volatility.

    R2 near zero does not mean the model is broken for that name — it means the
    name moves for reasons the factors do not carry, so its risk is specific and
    diversifiable rather than something a factor hedge can offset.
    """
    common = returns.columns.intersection(specific_returns.columns)
    if common.empty:
        return pd.DataFrame(columns=["security_id", "r2", "total_vol", "specific_vol"])
    scale = np.sqrt(periods_per_year(frequency))
    r = returns[common].reindex(specific_returns.index)
    u = specific_returns[common]
    var_r, var_u = r.var(), u.var()
    return pd.DataFrame({
        "security_id": common,
        "r2": (1 - (var_u / var_r.replace(0, np.nan))).clip(-1, 1).values,
        "total_vol": var_r.pow(0.5).values * scale,
        "specific_vol": var_u.pow(0.5).values * scale,
        "factor_vol": np.sqrt(np.clip((var_r - var_u).values, 0, None)) * scale,
    })
