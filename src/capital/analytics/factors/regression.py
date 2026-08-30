"""
Cross-sectional factor-return estimation.

Each period we solve one weighted least-squares problem across the universe

    r_t = X_{t-1} f_t + u_t

where the design matrix X carries a market intercept, industry dummies,
optional country dummies and the style exposures. Two details make this a real
Barra-style estimator rather than a plain regression:

**Identification.** Industry dummies sum to the intercept, so [1 | industries]
is rank-deficient and the market/industry split is arbitrary. The standard fix
is the constraint that industry factor returns are cap-weighted zero-sum (and
likewise for countries): the market factor then *is* the cap-weighted universe
return, and each industry return is a deviation from it. We impose it by
reparameterising onto the null space of the constraint matrix — solve for the
free coefficients, then map back — which is exact rather than a ridge fudge.

**Robustness.** A single blown-up small cap can otherwise drag a factor return
by a large fraction of its value. Two Huber IRLS passes cap the influence of any
one residual without discarding the observation.

Weights are sqrt(market cap) by default: full cap weighting lets the largest
handful of names dictate every factor return, equal weighting hands the estimate
to micro caps whose prices are noise. Square root is the usual compromise.
"""
import numpy as np
import pandas as pd
from scipy.linalg import null_space

_EPS = 1e-10
_HUBER_C = 1.345


# ── One cross-section ─────────────────────────────────────────────────────────

def _weights(caps: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "cap":
        w = caps
    elif scheme == "equal":
        w = np.ones_like(caps)
    else:
        w = np.sqrt(caps)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = w.sum()
    return w / total if total > 0 else w


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    xtx = Xw.T @ Xw
    coef = np.linalg.solve(xtx, Xw.T @ yw) if np.linalg.cond(xtx) < 1e12 \
        else np.linalg.lstsq(Xw, yw, rcond=None)[0]
    return coef, xtx


def _huber_reweight(resid: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Downweight residuals beyond ~1.3 robust sigmas. Scale from the MAD so the
    cut-off adapts to the period's own dispersion instead of a fixed threshold."""
    scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    if not np.isfinite(scale) or scale < _EPS:
        return w
    z = np.abs(resid) / scale
    return w * np.where(z <= _HUBER_C, 1.0, _HUBER_C / z)


def solve_cross_section(X: pd.DataFrame, y: pd.Series, caps: pd.Series,
                        constraints: np.ndarray | None, scheme: str = "sqrt_cap",
                        robust: bool = True) -> dict:
    """Constrained WLS for one date. Returns factor returns, residuals and fit stats."""
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    w0 = _weights(caps.to_numpy(dtype=float), scheme)

    # Reparameterise onto the null space of the constraints: f = R g.
    R = null_space(constraints) if constraints is not None and len(constraints) else np.eye(Xv.shape[1])
    if R.shape[1] == 0:
        raise ValueError("constraints leave no free parameters")
    Xr = Xv @ R

    w = w0.copy()
    coef, xtx = _wls(Xr, yv, w)
    resid = yv - Xr @ coef
    if robust:
        for _ in range(2):
            w = _huber_reweight(resid, w0)
            coef, xtx = _wls(Xr, yv, w)
            resid = yv - Xr @ coef

    f = R @ coef
    dof = max(len(yv) - R.shape[1], 1)
    sigma2 = float((w * resid ** 2).sum() / w.sum()) * len(yv) / dof
    try:
        cov_g = np.linalg.inv(xtx) * sigma2
        se = np.sqrt(np.clip(np.diag(R @ cov_g @ R.T), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(f), np.nan)

    ss_res = float((w * resid ** 2).sum())
    y_bar = float((w * yv).sum() / w.sum())
    ss_tot = float((w * (yv - y_bar) ** 2).sum())
    return {
        "factor_returns": pd.Series(f, index=X.columns),
        "std_errors": pd.Series(se, index=X.columns),
        "residuals": pd.Series(resid, index=X.index),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > _EPS else np.nan,
        "n": int(len(yv)),
    }


# ── Design matrix ─────────────────────────────────────────────────────────────

def build_design(sids: pd.Index, style_row: pd.DataFrame,
                 industry: pd.Series | None, country: pd.Series | None,
                 caps: pd.Series) -> tuple[pd.DataFrame, list[tuple[str, str]], np.ndarray | None]:
    """Assemble [market | industries | countries | styles] and its constraints.

    Returns the design matrix, a (factor, group) list, and the constraint matrix
    A such that A f = 0 pins the cap-weighted industry and country returns to
    zero. Industry/country blocks with a single member on the date are folded
    away — a dummy that is on for one security is that security's residual, not
    a factor.
    """
    blocks: list[pd.DataFrame] = [pd.DataFrame({"Market": 1.0}, index=sids)]
    groups: list[tuple[str, str]] = [("Market", "Market")]
    constraint_rows: list[np.ndarray] = []

    def add_categorical(series: pd.Series | None, prefix: str, group: str) -> None:
        if series is None:
            return
        member = series.reindex(sids)
        counts = member.value_counts()
        keep = [c for c in counts.index if counts[c] >= 2 and str(c) not in ("", "nan", "None")]
        if len(keep) < 2:
            return
        dummies = pd.DataFrame(
            {f"{prefix}{c}": (member == c).astype(float) for c in sorted(keep)}, index=sids)
        blocks.append(dummies)
        groups.extend((col, group) for col in dummies.columns)
        # cap weight of each category, as the constraint's coefficients
        cap_by = pd.Series({f"{prefix}{c}": float(caps[member == c].sum()) for c in sorted(keep)})
        constraint_rows.append(("block", dummies.columns.tolist(), cap_by))

    add_categorical(industry, "IND_", "Industry")
    add_categorical(country, "CTY_", "Country")

    if not style_row.empty:
        blocks.append(style_row)
        groups.extend((col, "Style") for col in style_row.columns)

    X = pd.concat(blocks, axis=1)
    names = list(X.columns)

    A = np.zeros((len(constraint_rows), len(names)))
    for i, (_, cols, cap_by) in enumerate(constraint_rows):
        total = cap_by.sum()
        if total <= 0:
            continue
        for col in cols:
            A[i, names.index(col)] = cap_by[col] / total
    return X, groups, (A if len(constraint_rows) else None)


# ── Panel loop ────────────────────────────────────────────────────────────────

def run_panel(style_panels: dict[str, pd.DataFrame], fwd_returns: pd.DataFrame,
              caps: pd.DataFrame, industry: pd.Series | None, country: pd.Series | None,
              spec, progress=None) -> dict:
    """Estimate the model on every date that has both exposures and a forward return.

    Cheap by construction: X is (n_securities x ~40) but X'WX is only ~40x40, so
    a 2,000-name universe over 250 dates is a few hundred small solves.
    """
    dates = [d for d in style_panels[next(iter(style_panels))].index if d in fwd_returns.index] \
        if style_panels else []
    style_keys = list(style_panels)

    f_rows, se_rows, resid_rows, stats_rows = {}, {}, {}, []
    groups: list[tuple[str, str]] = []

    for i, date in enumerate(dates):
        if progress and (i % 10 == 0 or i == len(dates) - 1):
            progress(i / max(len(dates), 1), f"Cross-section {i + 1}/{len(dates)} · {date.date()}")

        style_row = pd.DataFrame({k: style_panels[k].loc[date] for k in style_keys})
        y = fwd_returns.loc[date]
        cap = caps.loc[date]
        ok = style_row.notna().all(axis=1) & y.notna() & cap.notna() & (cap > 0)
        sids = style_row.index[ok]
        if len(sids) < len(style_keys) + 10:
            continue

        X, grp, A = build_design(sids, style_row.loc[sids], industry, country, cap.loc[sids])
        try:
            out = solve_cross_section(X, y.loc[sids], cap.loc[sids], A,
                                      spec.regression_weight, spec.robust)
        except (np.linalg.LinAlgError, ValueError):
            continue

        groups = grp or groups
        f_rows[date] = out["factor_returns"]
        se_rows[date] = out["std_errors"]
        resid_rows[date] = out["residuals"]
        stats_rows.append({"date": date, "r2": out["r2"], "n": out["n"]})

    if not f_rows:
        raise ValueError(
            "No cross-section could be estimated. The usual cause is too few "
            "securities with both a valid exposure row and a forward return — "
            "widen the date range, lower the history requirement, or back-fill "
            "EOD prices with `capital-ingest eod --start <date>`."
        )

    factor_returns = pd.DataFrame(f_rows).T.sort_index()
    return {
        "factor_returns": factor_returns,
        "std_errors": pd.DataFrame(se_rows).T.sort_index().reindex(columns=factor_returns.columns),
        "specific_returns": pd.DataFrame(resid_rows).T.sort_index(),
        "fit_stats": pd.DataFrame(stats_rows).set_index("date").sort_index(),
        "factor_groups": dict(groups),
    }


# ── Returns-based exposures (ETFs, funds, anything without fundamentals) ──────

def returns_based_exposures(security_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                            style_keys: list[str], ridge: float = 1e-4) -> pd.DataFrame:
    """Style exposures for securities the cross-section cannot price directly.

    An ETF has no book value and no sector — but it does have returns, so we
    regress those on the *estimated* factor returns (a time-series regression:
    returns-based style analysis). The result is on the same scale as the
    cross-sectional exposures, which is what makes it legitimate to add an ETF
    and a single stock into one portfolio exposure number. A small ridge keeps
    the estimate stable when factor returns are collinear over a short window.

    Returns one row per security: the style betas, plus r2, n_obs and the
    residual volatility — the fund's own specific risk. Without that last column
    a diversified ETF would inherit the median *single-stock* specific risk and
    the portfolio's diversifiable share would be badly overstated.
    """
    cols = ["Market"] + [k for k in style_keys if k in factor_returns.columns]
    F = factor_returns[cols].dropna()
    if F.empty or security_returns.empty:
        return pd.DataFrame(columns=[*cols, "r2", "n_obs", "resid_vol"])

    rows = {}
    Fv = F.to_numpy(dtype=float)
    for sid in security_returns.columns:
        y = security_returns[sid].reindex(F.index)
        ok = y.notna().to_numpy()
        if ok.sum() < max(len(cols) + 10, 30):
            continue
        X, yv = Fv[ok], y.to_numpy(dtype=float)[ok]
        xtx = X.T @ X + ridge * np.eye(X.shape[1]) * np.trace(X.T @ X) / X.shape[1]
        try:
            beta = np.linalg.solve(xtx, X.T @ yv)
        except np.linalg.LinAlgError:
            continue
        resid = yv - X @ beta
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        rows[sid] = {**dict(zip(cols, beta)),
                     "r2": 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > _EPS else np.nan,
                     "n_obs": int(ok.sum()),
                     "resid_vol": float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan}
    return pd.DataFrame(rows).T if rows else pd.DataFrame(
        columns=[*cols, "r2", "n_obs", "resid_vol"])
