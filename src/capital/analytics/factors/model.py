"""
Model orchestration: data in, estimated factor model out.

`run_factor_model(spec, progress)` is the single entry point. It is written to
be run as a background job (see capital.jobs), so it reports progress at every
stage and never touches Dash, S3 or the browser.

Pipeline
--------
    universe  ->  descriptors  ->  styles  ->  cross-sectional regression
                                                 |
                     factor covariance  <--------+--------> specific risk
                                                 |
                                            diagnostics

Cost scales with the number of *estimation dates*, not securities: descriptors
are whole-matrix operations and each cross-section is a ~40x40 solve regardless
of whether the universe is 100 names or 2,000. The expensive part is loading and
aligning the price panel, which happens once.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capital.analytics.factors import descriptors as desc
from capital.analytics.factors import diagnostics as diag
from capital.analytics.factors import exposures as expo
from capital.analytics.factors import regression as reg
from capital.analytics.factors import risk as riskmod
from capital.analytics.factors.spec import DESCRIPTORS, STYLES, ModelSpec
from capital.data import loaders

_WARMUP_DAYS = 500          # calendar days of price history before `start`
_FREQ_ALIAS = {"B": "B", "D": "B", "W-FRI": "W-FRI", "W": "W-FRI", "ME": "ME", "M": "ME"}


@dataclass
class FactorModelResult:
    """Everything one run produces. Frames only — nothing Dash- or job-specific."""
    spec: ModelSpec
    summary: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    security_meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    style_panels: dict = field(default_factory=dict)
    exposure_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_std_errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    specific_returns: pd.DataFrame = field(default_factory=pd.DataFrame)
    fit_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_groups: dict = field(default_factory=dict)
    factor_cov: pd.DataFrame = field(default_factory=pd.DataFrame)
    factor_corr: pd.DataFrame = field(default_factory=pd.DataFrame)
    specific_risk: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)


def _noop(_frac: float, _msg: str) -> None:
    pass


# ── Universe ──────────────────────────────────────────────────────────────────

def _select_universe(spec: ModelSpec, prices: pd.DataFrame,
                     caps_daily: pd.DataFrame, master: pd.DataFrame) -> tuple[list[str], list[str], dict]:
    """Split the master into an estimation universe and a returns-priced tail.

    Estimation is single stocks only. ETFs are portfolios of the very securities
    being regressed — putting them in the cross-section double-counts their
    holdings and lets a fund's own factor bet contaminate the factor return it is
    supposed to be measured against. They come back in via
    `regression.returns_based_exposures`, priced *off* the finished model.
    """
    active = master[master["asset_type"] != "INDEX"].copy()
    history = prices.notna().sum()
    notes: dict = {}

    est = active[active["asset_type"].isin(spec.asset_types)]
    est_ids = [s for s in est["security_id"] if s in prices.columns]
    est_ids = [s for s in est_ids if history.get(s, 0) >= spec.min_history_days]
    notes["dropped_short_history"] = int(len(est) - len(est_ids))

    if spec.min_market_cap > 0 and not caps_daily.empty:
        last_cap = caps_daily.ffill().iloc[-1]
        keep = [s for s in est_ids if float(last_cap.get(s, np.nan) or np.nan) >= spec.min_market_cap]
        notes["dropped_small_cap"] = len(est_ids) - len(keep)
        est_ids = keep

    if spec.max_securities and len(est_ids) > spec.max_securities and not caps_daily.empty:
        last_cap = caps_daily.ffill().iloc[-1].reindex(est_ids)
        est_ids = list(last_cap.nlargest(spec.max_securities).index)
        notes["capped_to_max_securities"] = True

    other_ids: list[str] = []
    if spec.include_etfs:
        other = active[~active["asset_type"].isin(spec.asset_types)]
        other_ids = [s for s in other["security_id"]
                     if s in prices.columns and history.get(s, 0) >= 60]

    notes["n_estimation"] = len(est_ids)
    notes["n_returns_priced"] = len(other_ids)
    return est_ids, other_ids, notes


def _estimation_dates(spec: ModelSpec, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Nominal period ends snapped back to the last trading day that exists."""
    freq = _FREQ_ALIAS.get(spec.frequency, "W-FRI")
    nominal = pd.date_range(spec.start, spec.as_of, freq=freq)
    snapped = sorted({index.asof(d) for d in nominal if index.asof(d) is not pd.NaT})
    snapped = [d for d in snapped if pd.notna(d)]
    last = index[index <= pd.Timestamp(spec.as_of)]
    if len(last) and (not snapped or snapped[-1] != last[-1]):
        snapped.append(last[-1])          # always carry a live "today" cross-section
    return pd.DatetimeIndex(sorted(set(snapped)))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_factor_model(spec: ModelSpec, progress=None) -> FactorModelResult:
    progress = progress or _noop
    started = time.time()

    # -- 1. dates and price panel -------------------------------------------
    progress(0.02, "Loading price history")
    master = loaders.get_security_master()
    if master.empty:
        raise ValueError("security_master.csv is empty — nothing to estimate.")

    probe = loaders.get_close_matrix()
    if probe.empty:
        raise ValueError("No EOD prices in the store. Run `capital-ingest eod` first.")
    as_of = pd.Timestamp(spec.as_of) if spec.as_of else probe.index.max()
    start = pd.Timestamp(spec.start) if spec.start else as_of - pd.DateOffset(years=3)
    spec = spec.replace(as_of=as_of.date().isoformat(), start=start.date().isoformat())
    load_from = (start - pd.Timedelta(days=_WARMUP_DAYS)).date().isoformat()

    prices = probe.loc[probe.index >= load_from]
    prices = prices.loc[prices.index <= as_of]
    close = loaders.get_eod_matrix("close", start=load_from).reindex(prices.index)
    volume = loaders.get_eod_matrix("volume", start=load_from).reindex(prices.index)

    # -- 2. market cap panel (daily: weights the market and denominates turnover)
    progress(0.10, "Loading fundamentals")
    ric_to_sid = dict(zip(master["ric"], master["security_id"]))
    caps_ric = loaders.get_fundamentals_matrix("market_cap", start=load_from)
    caps_daily = _to_sid(caps_ric, ric_to_sid).reindex(prices.index).ffill()
    shares_ric = loaders.get_fundamentals_matrix("shares_outstanding", start=load_from)
    shares_daily = _to_sid(shares_ric, ric_to_sid).reindex(prices.index).ffill() \
        if not shares_ric.empty else None

    # -- 3. universe --------------------------------------------------------
    est_ids, other_ids, universe_notes = _select_universe(spec, prices, caps_daily, master)
    if len(est_ids) < 20:
        raise ValueError(
            f"Only {len(est_ids)} securities qualify for the estimation universe "
            f"(need 20+). Lower 'minimum history' or widen the asset types; a "
            f"cross-sectional model cannot be fitted on a handful of names."
        )
    est_dates = _estimation_dates(spec, prices.index)
    if len(est_dates) < 8:
        raise ValueError(f"Only {len(est_dates)} estimation dates in "
                         f"{spec.start}..{spec.as_of} at frequency {spec.frequency}.")

    px = prices[est_ids]
    caps = caps_daily.reindex(columns=est_ids)

    # Optional numeraire conversion. A universe spanning EUR, GBP, CHF and SEK
    # mixes FX moves into every price-based descriptor; converting the price
    # panel removes that. If the FX panel does not cover every currency the run
    # stays in local currency and the coverage report says so — country factors
    # then absorb most of the drift.
    numeraire = "local"
    if spec.numeraire != "local":
        converted, applied = desc.convert_prices_to_numeraire(
            px, master.set_index("security_id")["currency"],
            loaders.get_fx_rates(spec.numeraire))
        if applied:
            px, numeraire = converted, spec.numeraire
        else:
            log_note = (f"requested {spec.numeraire} but the store has no FX rates "
                        f"covering every currency in the universe")
            universe_notes["numeraire_fallback"] = log_note

    # -- 4. descriptors -----------------------------------------------------
    progress(0.18, f"Building descriptors · {len(est_ids)} securities")
    wanted = spec.required_descriptors()
    price_panels, byproducts = desc.build_price_descriptors(
        px, close.reindex(columns=est_ids), volume.reindex(columns=est_ids),
        caps, shares_daily.reindex(columns=est_ids) if shares_daily is not None else None,
        spec, wanted)

    progress(0.34, "Sampling fundamentals on estimation dates")
    fund_cols = spec.required_fund_columns()
    fund_long = loaders.get_fundamentals_asof(
        tuple(d.date().isoformat() for d in est_dates), tuple(fund_cols))
    fund_wide = {col: _to_sid(fund_long.pivot(index="date", columns="ric", values=col),
                              ric_to_sid).reindex(index=est_dates, columns=est_ids)
                 for col in fund_cols if col in fund_long.columns}
    fund_panels = desc.build_fundamental_descriptors(fund_wide, wanted)

    # Growth is derived from stored *level* series, so it needs the daily panel
    # (a year-on-year change cannot be read off estimation-date snapshots alone).
    growth_cols = {DESCRIPTORS[k].column for k in wanted
                   if k in DESCRIPTORS and DESCRIPTORS[k].transform == "yoy_growth"}
    if growth_cols:
        levels = {col: _to_sid(loaders.get_fundamentals_matrix(col, start=load_from),
                               ric_to_sid).reindex(index=prices.index, columns=est_ids).ffill()
                  for col in growth_cols}
        fund_panels.update({k: v.reindex(est_dates) for k, v in
                            desc.build_growth_descriptors(levels, wanted).items()})

    # lncap is a fundamentals column but behaves like a price descriptor (daily)
    if "lncap" in wanted:
        fund_panels["lncap"] = np.log(caps.where(caps > 0)).reindex(est_dates)

    panels = {**{k: v.reindex(est_dates) for k, v in price_panels.items() if v is not None},
              **fund_panels}

    # -- 5. styles ----------------------------------------------------------
    progress(0.42, "Standardising exposures")
    industry = _industry_map(fund_long, ric_to_sid, est_ids)
    country = master.set_index("security_id")["country"].reindex(est_ids) \
        if spec.country_factors else None
    cap_weights = caps.reindex(est_dates).fillna(0.0)
    style_panels, coverage = expo.build_styles(panels, cap_weights, industry, spec)
    if not style_panels:
        raise ValueError("No style factor survived the coverage checks — see the "
                         "coverage report. With the store as it ships, use the "
                         "'Core' preset until extended fundamentals are back-filled.")

    # -- 6. forward returns -------------------------------------------------
    progress(0.48, "Aligning forward returns")
    period_fwd = px.reindex(est_dates).pct_change(fill_method=None).shift(-1)
    horizons = {h: px.pct_change(h, fill_method=None).shift(-h).reindex(est_dates)
                 for h in spec.ic_horizons}

    # -- 7. cross-sectional regression --------------------------------------
    def reg_progress(frac, msg):
        progress(0.50 + 0.25 * frac, msg)

    panel = reg.run_panel(style_panels, period_fwd, caps.reindex(est_dates),
                          industry if spec.industry_factors else None,
                          country, spec, progress=reg_progress)

    # -- 8. risk model ------------------------------------------------------
    progress(0.78, "Estimating factor covariance and specific risk")
    cov = riskmod.factor_covariance(panel["factor_returns"], spec.cov_var_halflife,
                                    spec.cov_corr_halflife, spec.newey_west_lags,
                                    spec.frequency)
    size_last = style_panels.get("size")
    spec_risk = riskmod.specific_risk(
        panel["specific_returns"], spec.specific_halflife,
        size_last.iloc[-1] if size_last is not None and len(size_last) else None,
        spec.frequency)

    # -- 9. latest exposure matrix (styles + industry/country dummies) -------
    progress(0.84, "Assembling exposure matrix")
    last_date = max(d for d in style_panels[next(iter(style_panels))].index)
    style_last = pd.DataFrame({k: v.loc[last_date] for k, v in style_panels.items()})
    X_last, groups, _ = reg.build_design(
        style_last.dropna(how="all").index, style_last.dropna(how="all"),
        industry if spec.industry_factors else None, country,
        caps.loc[last_date].fillna(0.0))
    exposure_matrix = X_last.reindex(columns=panel["factor_returns"].columns).fillna(0.0)

    # -- 10. ETFs and anything else priced off the finished model ------------
    rb_meta = pd.DataFrame()
    if other_ids:
        progress(0.88, f"Fitting {len(other_ids)} returns-priced securities")
        other_ret = desc.simple_returns(prices[other_ids]).reindex(prices.index)
        period_other = prices[other_ids].reindex(est_dates).pct_change(fill_method=None)
        rb = reg.returns_based_exposures(period_other, panel["factor_returns"],
                                         list(style_panels))
        if not rb.empty:
            style_cols = [c for c in rb.columns if c in style_panels]
            for key in style_cols:
                style_panels[key] = pd.concat(
                    [style_panels[key],
                     pd.DataFrame({sid: rb.loc[sid, key] for sid in rb.index},
                                  index=style_panels[key].index)], axis=1)
            extra = pd.DataFrame(0.0, index=rb.index, columns=exposure_matrix.columns)
            for col in ["Market", *style_cols]:
                if col in rb.columns and col in extra.columns:
                    extra[col] = rb[col]
            exposure_matrix = pd.concat([exposure_matrix, extra])
            rb_meta = rb[["r2", "n_obs"]].rename(columns={"r2": "rb_r2", "n_obs": "rb_n_obs"})
            # A fund's specific risk is the residual of its own time-series fit —
            # never the median single stock's, which would overstate the
            # portfolio's diversifiable share by a wide margin.
            if "resid_vol" in rb.columns:
                scale = np.sqrt(riskmod.periods_per_year(spec.frequency))
                extra_risk = pd.DataFrame({
                    "sigma_raw": rb["resid_vol"] * scale,
                    "sigma": rb["resid_vol"] * scale,
                    "shrink_weight": 0.0,
                    "n_obs": rb["n_obs"],
                }).dropna(subset=["sigma"])
                spec_risk = pd.concat([spec_risk, extra_risk[~extra_risk.index.isin(spec_risk.index)]])
        del other_ret

    # -- 11. diagnostics ----------------------------------------------------
    progress(0.90, "Running robustness diagnostics")
    est_styles = {k: v[[c for c in v.columns if c in est_ids]] for k, v in style_panels.items()}
    ic_summary, ic_series = diag.information_coefficients(est_styles, horizons)
    q_means, q_spreads = diag.quantile_spreads(est_styles, period_fwd,
                                               caps.reindex(est_dates), spec.n_quantiles,
                                               spec.frequency)
    diagnostics = {
        "ic_summary": ic_summary,
        "ic_series": ic_series,
        "quantile_means": q_means,
        "quantile_spreads": q_spreads,
        "factor_performance": diag.factor_performance(panel["factor_returns"], spec.frequency),
        "subperiod_returns": diag.subperiod_returns(panel["factor_returns"]),
        "cumulative_returns": diag.cumulative_factor_returns(panel["factor_returns"]),
        "persistence": diag.exposure_persistence(est_styles),
        "vif": diag.variance_inflation(exposure_matrix, list(style_panels)).rename("vif")
                  .to_frame().reset_index(names="factor"),
        "security_stability": diag.security_exposure_stability(style_panels),
        "security_fit": diag.security_fit(
            px.reindex(est_dates).pct_change(fill_method=None),
            panel["specific_returns"], spec.frequency),
    }

    # -- 12. metadata and summary -------------------------------------------
    progress(0.93, "Packaging results")
    meta = _security_meta(master, est_ids, other_ids, caps, industry, exposure_matrix, rb_meta)
    coverage.update({
        "universe": universe_notes,
        "numeraire": numeraire,
        "estimation_dates": len(est_dates),
        "fundamentals_columns_available": list(loaders.get_fundamentals_columns()),
        "styles_estimated": list(style_panels),
    })
    summary = {
        "n_securities": len(est_ids) + len(other_ids),
        "n_estimation": len(est_ids),
        "n_periods": int(len(panel["factor_returns"])),
        "n_factors": int(panel["factor_returns"].shape[1]),
        "mean_r2": float(panel["fit_stats"]["r2"].mean()) if len(panel["fit_stats"]) else None,
        # Observations per estimated factor. Below ~10 the cross-section is thin
        # enough that industry/country dummies start fitting noise — the UI warns.
        "obs_per_factor": (round(float(panel["fit_stats"]["n"].mean())
                                 / max(panel["factor_returns"].shape[1], 1), 1)
                           if len(panel["fit_stats"]) else None),
        "start": spec.start, "as_of": spec.as_of, "frequency": spec.frequency,
        "last_cross_section": str(last_date.date()),
        "runtime_seconds": round(time.time() - started, 1),
    }

    return FactorModelResult(
        spec=spec, summary=summary, coverage=coverage, security_meta=meta,
        style_panels=style_panels, exposure_matrix=exposure_matrix,
        factor_returns=panel["factor_returns"], factor_std_errors=panel["std_errors"],
        specific_returns=panel["specific_returns"], fit_stats=panel["fit_stats"],
        factor_groups={**panel["factor_groups"], **dict(groups)},
        factor_cov=cov, factor_corr=riskmod.factor_correlation(cov),
        specific_risk=spec_risk, diagnostics=diagnostics,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_sid(frame: pd.DataFrame, ric_to_sid: dict) -> pd.DataFrame:
    """Relabel a RIC-keyed panel onto security_ids, dropping unmapped columns.

    Fundamentals keep rows under retired RICs after an exchange migration; those
    have no master row and are exactly what should be dropped here.
    """
    if frame.empty:
        return frame
    cols = {c: ric_to_sid[c] for c in frame.columns if c in ric_to_sid}
    out = frame[list(cols)].rename(columns=cols)
    return out.loc[:, ~out.columns.duplicated()]


def _industry_map(fund_long: pd.DataFrame, ric_to_sid: dict, sids: list[str]) -> pd.Series:
    if fund_long.empty or "gics_sector" not in fund_long.columns:
        return pd.Series("Unclassified", index=sids)
    latest = fund_long.sort_values("date").groupby("ric")["gics_sector"].last()
    mapped = {ric_to_sid[r]: (v or "Unclassified") for r, v in latest.items() if r in ric_to_sid}
    out = pd.Series(mapped).reindex(sids)
    return out.replace("", "Unclassified").fillna("Unclassified")


def _security_meta(master: pd.DataFrame, est_ids: list[str], other_ids: list[str],
                   caps: pd.DataFrame, industry: pd.Series,
                   exposure_matrix: pd.DataFrame, rb_meta: pd.DataFrame) -> pd.DataFrame:
    ids = list(est_ids) + list(other_ids)
    meta = master.set_index("security_id").reindex(ids)[
        ["ric", "ticker", "name", "currency", "asset_type", "country"]].copy()
    last_cap = caps.ffill().iloc[-1] if len(caps) else pd.Series(dtype=float)
    meta["market_cap"] = last_cap.reindex(ids)
    meta["sector"] = industry.reindex(ids).fillna("Unclassified")
    meta["method"] = ["cross-sectional" if s in set(est_ids) else "returns-based" for s in ids]
    meta["in_model"] = [s in exposure_matrix.index for s in ids]
    if not rb_meta.empty:
        meta = meta.join(rb_meta)
    return meta.reset_index(names="security_id")
