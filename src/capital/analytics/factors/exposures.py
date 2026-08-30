"""
Descriptor standardisation and style-factor construction.

Three ideas do most of the work here, and all three are Barra conventions worth
knowing because they change how the numbers should be read:

1. **Robust scaling before standardising.** Raw descriptors have fat tails (one
   company with a near-zero book value dominates a plain z-score). We centre on
   the median, scale by the MAD, clip at +/- `winsor_sigma`, and only then
   standardise. One outlier can no longer set the scale for the whole universe.

2. **Cap-weighted mean, equal-weighted standard deviation.** Exposures are
   centred so the *cap-weighted market portfolio has zero exposure to every
   style*, and scaled so a one-unit exposure means one cross-sectional standard
   deviation. That is what makes "the portfolio is +0.8 Momentum" a statement
   about the portfolio's tilt away from the market, not away from an equal-
   weighted average nobody holds.

3. **Missing values filled at the industry mean.** A security with no book value
   is not assumed to be average overall — it is assumed to be average *for its
   industry*, which is the weaker and more defensible assumption.

Everything is row-wise (per date) and therefore vectorised across dates.
"""
import numpy as np
import pandas as pd

from capital.analytics.factors.spec import STYLES, ModelSpec

_EPS = 1e-12


# ── Row-wise (per-date) primitives ────────────────────────────────────────────

def robust_z(panel: pd.DataFrame, winsor_sigma: float) -> pd.DataFrame:
    """Median/MAD z-score clipped at +/- winsor_sigma, computed per date."""
    median = panel.median(axis=1)
    mad = (panel.sub(median, axis=0)).abs().median(axis=1) * 1.4826
    # Fall back to the plain standard deviation where >50% of the row is tied
    # (MAD == 0), which happens with sparse or heavily rounded descriptors.
    scale = mad.where(mad > _EPS, panel.std(axis=1))
    z = panel.sub(median, axis=0).div(scale.replace(0, np.nan), axis=0)
    return z.clip(-winsor_sigma, winsor_sigma)


def cap_standardise(panel: pd.DataFrame, cap_weights: pd.DataFrame) -> pd.DataFrame:
    """Centre on the cap-weighted mean, scale by the equal-weighted std."""
    w = cap_weights.where(panel.notna())
    w_sum = w.sum(axis=1).replace(0, np.nan)
    mean = (panel * w).sum(axis=1) / w_sum
    centred = panel.sub(mean, axis=0)
    std = centred.std(axis=1).replace(0, np.nan)
    return centred.div(std, axis=0)


def fill_industry_mean(panel: pd.DataFrame, industry: pd.Series,
                       cap_weights: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps with the industry's cap-weighted mean, then the market's (0).

    `industry` maps security_id -> industry for the whole run rather than per
    date: GICS sectors are near-static, and a reclassification shifting a name's
    historical fill value is a smaller error than looping 250 cross-sections.
    """
    filled = panel.copy()
    for _, members in industry.groupby(industry):
        cols = [c for c in members.index if c in panel.columns]
        if len(cols) < 2:
            continue
        block = panel[cols]
        w = cap_weights[cols].where(block.notna())
        w_sum = w.sum(axis=1).replace(0, np.nan)
        group_mean = (block * w).sum(axis=1) / w_sum
        filled[cols] = block.where(block.notna(), group_mean, axis=0)
    return filled.fillna(0.0).where(panel.notna() | _has_any(panel), np.nan)


def _has_any(panel: pd.DataFrame) -> pd.DataFrame:
    """True wherever the date has at least one observation (so a fully empty
    cross-section stays NaN instead of collapsing to a row of zeros)."""
    any_row = panel.notna().any(axis=1)
    return pd.DataFrame(np.broadcast_to(any_row.to_numpy()[:, None], panel.shape),
                        index=panel.index, columns=panel.columns)


def orthogonalise(target: pd.DataFrame, bases: list[pd.DataFrame],
                  cap_weights: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional cap-weighted residual of `target` on `bases`, per date.

    Without this, Residual Volatility is mostly Beta wearing a different hat and
    the regression cannot tell the two apart. Removing the projection makes each
    style's factor return interpretable on its own.
    """
    if not bases:
        return target
    out = target.copy()
    stacked = np.stack([b.to_numpy(dtype=float) for b in bases], axis=-1)
    y_all = target.to_numpy(dtype=float)
    w_all = cap_weights.to_numpy(dtype=float)

    for i in range(len(target)):
        y, X, w = y_all[i], stacked[i], w_all[i]
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(w) & (w > 0)
        if ok.sum() < X.shape[1] + 5:
            continue
        Xo = np.column_stack([np.ones(ok.sum()), X[ok]])
        sw = np.sqrt(w[ok])
        try:
            coef, *_ = np.linalg.lstsq(Xo * sw[:, None], y[ok] * sw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = np.full_like(y, np.nan)
        resid[ok] = y[ok] - Xo @ coef
        out.iloc[i] = resid
    return out


# ── Style construction ────────────────────────────────────────────────────────

def build_styles(descriptors: dict[str, pd.DataFrame], cap_weights: pd.DataFrame,
                 industry: pd.Series, spec: ModelSpec) -> tuple[dict[str, pd.DataFrame], dict]:
    """Standardise descriptors, blend them into styles, orthogonalise, restandardise.

    Returns the style panels plus a coverage report naming every descriptor and
    style that had to be dropped and why — the model never silently substitutes
    a factor it could not actually measure.
    """
    report: dict = {"descriptors": {}, "dropped_descriptors": [],
                    "dropped_styles": {}, "styles": {}}

    # -- 1. standardise each descriptor -------------------------------------
    std: dict[str, pd.DataFrame] = {}
    for key, panel in descriptors.items():
        if panel is None or panel.empty:
            report["dropped_descriptors"].append({"descriptor": key, "reason": "no data"})
            continue
        coverage = panel.notna().mean(axis=1)
        usable = coverage >= spec.min_coverage
        mean_cov = float(coverage.mean()) if len(coverage) else 0.0
        report["descriptors"][key] = {
            "mean_coverage": round(mean_cov, 4),
            "dates_usable": int(usable.sum()),
            "dates_total": int(len(coverage)),
        }
        if not usable.any():
            report["dropped_descriptors"].append({
                "descriptor": key,
                "reason": f"coverage {mean_cov:.0%} never reaches the {spec.min_coverage:.0%} floor",
            })
            continue
        masked = panel.where(usable, np.nan)
        z = cap_standardise(robust_z(masked, spec.winsor_sigma), cap_weights)
        filled = fill_industry_mean(z, industry, cap_weights)

        # A descriptor with no cross-sectional variation standardises to NaN
        # (division by a zero standard deviation). That is not a coverage
        # problem — the data is there, it just says the same thing about every
        # security — so it has to be caught separately or it would poison its
        # style while the coverage report showed 100%.
        usable_dates = float(filled.notna().any(axis=1).mean())
        report["descriptors"][key]["dates_with_variation"] = round(usable_dates, 4)
        if usable_dates < spec.min_coverage:
            report["dropped_descriptors"].append({
                "descriptor": key,
                "reason": f"no cross-sectional variation on {1 - usable_dates:.0%} of dates "
                          f"(every security scores the same, so it cannot rank them)",
            })
            continue
        std[key] = filled

    # -- 2. Mid Cap is derived from standardised Size ------------------------
    if "midcap" in {d for s in spec.active_styles for d in s.weights} and "lncap" in std:
        cubed = std["lncap"] ** 3
        std["midcap"] = cap_standardise(robust_z(cubed, spec.winsor_sigma), cap_weights)

    # -- 3. blend descriptors into styles ------------------------------------
    styles: dict[str, pd.DataFrame] = {}
    for style in spec.active_styles:
        available = {k: w for k, w in style.weights.items() if k in std}
        if not available:
            report["dropped_styles"][style.key] = (
                "none of its descriptors are available: "
                + ", ".join(sorted(style.weights))
            )
            continue
        # Renormalise per cell, not per style: where one descriptor is missing
        # for one security on one date, the blend uses the rest at proportionally
        # larger weight instead of collapsing to NaN.
        total = sum(available.values())
        numerator = sum(std[k].fillna(0.0) * w for k, w in available.items())
        denominator = sum(std[k].notna().astype(float) * w for k, w in available.items())
        blend = numerator / denominator.where(denominator > 0)
        standardised = cap_standardise(blend, cap_weights)
        if not standardised.notna().to_numpy().any():
            report["dropped_styles"][style.key] = (
                "every descriptor standardised away (no cross-sectional variation)")
            continue
        styles[style.key] = standardised
        report["styles"][style.key] = {
            "descriptors_used": sorted(available),
            "descriptors_missing": sorted(set(style.weights) - set(available)),
            "weights": {k: round(w / total, 4) for k, w in available.items()},
        }

    # -- 4. orthogonalise, in declaration order so bases are already final ---
    for style in spec.active_styles:
        if style.key not in styles or not style.orthogonalise_to:
            continue
        bases = [styles[b] for b in style.orthogonalise_to if b in styles]
        if not bases:
            continue
        resid = orthogonalise(styles[style.key], bases, cap_weights)
        styles[style.key] = cap_standardise(resid, cap_weights)
        report["styles"][style.key]["orthogonalised_to"] = [
            b for b in style.orthogonalise_to if b in styles
        ]

    return styles, report


def style_labels(keys) -> dict[str, str]:
    return {k: STYLES[k].label for k in keys if k in STYLES}
