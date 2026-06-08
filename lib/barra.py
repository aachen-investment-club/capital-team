"""
Barra-style cross-sectional factor model.

Style factors : Value, Momentum, Growth, Volatility, Liquidity
Industry      : GICS sector dummies

The exposure matrix can always be built from fundamentals + EOD prices.
Factor-return estimation (WLS) is optional — it requires enough securities
with both price history AND fundamentals to be over-determined.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=RuntimeWarning)

STYLE_FACTORS = ["Value", "Momentum", "Growth", "Volatility", "Liquidity"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_float(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan


def _winsorise(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / (std + 1e-8) if std > 0 else s * 0.0


def _nearest_price(pdf: pd.DataFrame, p_col: str, target: pd.Timestamp) -> float:
    sub = pdf[pdf["date"] <= target]
    return float(sub[p_col].iloc[-1]) if not sub.empty else np.nan


# ── Exposure matrix ───────────────────────────────────────────────────────────

def build_exposure_matrix(
    records: list,
    eod_prices: dict,
    as_of_date,
    fund_history: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """
    Build factor exposure matrix from plain-Python records.

    Parameters
    ----------
    records      : list of dicts with keys ric, ticker, gics_sector,
                   pb_ratio, market_cap, shares_outstanding
    eod_prices   : {ric_or_ticker: eod_DataFrame}
    as_of_date   : date
    fund_history : full fundamentals DataFrame (all dates) used to compute
                   market-cap-based momentum/growth when EOD prices are absent

    Returns
    -------
    DataFrame indexed by ticker; columns = STYLE_FACTORS + Industry_* dummies.
    """
    as_of    = pd.Timestamp(as_of_date)
    date_1m  = as_of - pd.DateOffset(months=1)
    date_12m = as_of - pd.DateOffset(months=12)

    # Build market_cap time series lookup from fund_history keyed by ric
    mcap_history: dict = {}
    if fund_history is not None and not fund_history.empty:
        fh = fund_history.copy()
        fh["date"] = pd.to_datetime(fh["date"])
        for ric_key, grp in fh.groupby("ric"):
            grp = grp.sort_values("date")[["date", "market_cap"]].dropna()
            if not grp.empty:
                mcap_history[str(ric_key)] = grp

    rows = []
    for rec in records:
        ric    = str(rec.get("ric")         or "")
        ticker = str(rec.get("ticker")      or "")
        sector = str(rec.get("gics_sector") or "")

        # ── Price-based factors via EOD prices (most accurate) ─────────────
        momentum = growth = volatility = np.nan
        eod = None
        for key in (ric, ticker):
            if key and isinstance(eod_prices, dict) and key in eod_prices:
                cand = eod_prices[key]
                if isinstance(cand, pd.DataFrame) and not cand.empty:
                    eod = cand
                    break

        if eod is not None:
            pdf   = eod.copy()
            pdf["date"] = pd.to_datetime(pdf["date"])
            pdf   = pdf.sort_values("date")
            p_col = "adj_close" if "adj_close" in pdf.columns else "close"

            p_now  = _nearest_price(pdf, p_col, as_of)
            p_1m   = _nearest_price(pdf, p_col, date_1m)
            p_12m  = _nearest_price(pdf, p_col, date_12m)

            if np.isfinite(p_12m) and p_12m > 0:
                if np.isfinite(p_1m):
                    momentum = p_1m  / p_12m - 1.0
                if np.isfinite(p_now):
                    growth   = p_now / p_12m - 1.0

            recent = pdf[pdf["date"] <= as_of].tail(252)
            if len(recent) >= 20:
                volatility = float(recent[p_col].pct_change().dropna().std() * np.sqrt(252))

        # ── Fallback: market_cap time series as price proxy ────────────────
        # market_cap = price × shares_outstanding; when shares are stable,
        # percentage changes in market_cap equal percentage changes in price.
        if (not np.isfinite(momentum) or not np.isfinite(growth) or not np.isfinite(volatility)) and ric in mcap_history:
            mdf    = mcap_history[ric]
            mc_now = _nearest_price(mdf, "market_cap", as_of)
            mc_1m  = _nearest_price(mdf, "market_cap", date_1m)
            # For 12m base: use nearest available on or before 12m ago;
            # if no data that old, fall back to oldest available row.
            mc_12m = _nearest_price(mdf, "market_cap", date_12m)
            if not np.isfinite(mc_12m):
                oldest = mdf[mdf["date"] <= as_of]
                if not oldest.empty:
                    mc_12m = float(oldest["market_cap"].iloc[0])
            if np.isfinite(mc_12m) and mc_12m > 0:
                if not np.isfinite(momentum) and np.isfinite(mc_1m) and mc_12m > 0:
                    momentum = mc_1m  / mc_12m - 1.0
                if not np.isfinite(growth) and np.isfinite(mc_now) and mc_12m > 0:
                    growth   = mc_now / mc_12m - 1.0
            if not np.isfinite(volatility):
                sub = mdf[mdf["date"] <= as_of].tail(252)
                if len(sub) >= 20:
                    volatility = float(sub["market_cap"].pct_change().dropna().std() * np.sqrt(252))

        # Fundamental factors
        pb_f  = _to_float(rec.get("pb_ratio"))
        value = (1.0 / (pb_f + 1e-8)) if np.isfinite(pb_f) and pb_f > 0 else np.nan

        mcap_f = _to_float(rec.get("market_cap"))
        # Liquidity proxy: log(market_cap) — larger cap = more liquid.
        # shares_outstanding is often missing from data providers; log(mcap) is the
        # standard fallback used in single-factor Barra-lite implementations.
        if np.isfinite(mcap_f) and mcap_f > 0:
            liquidity = float(np.log(mcap_f))
        else:
            liquidity = np.nan

        clean_sector = sector.strip() if sector else ""
        rows.append({
            "ticker":      ticker,
            "ric":         ric,
            "gics_sector": clean_sector if clean_sector else "Other",
            "market_cap":  mcap_f,
            "Value":       value,
            "Momentum":    momentum,
            "Growth":      growth,
            "Volatility":  volatility,
            "Liquidity":   liquidity,
        })

    df = pd.DataFrame(rows).set_index("ticker")
    if df.empty:
        return df

    # Z-score + winsorise each style factor across the universe
    for factor in STYLE_FACTORS:
        col = df[factor].dropna()
        if len(col) > 2:
            normed = _zscore(_winsorise(col))
            df[factor] = normed
        # Rows with no data stay NaN — do not fill

    # Industry dummies
    dummies = pd.get_dummies(df["gics_sector"], prefix="Industry").astype(float)
    df = pd.concat([df, dummies], axis=1).drop(columns=["gics_sector", "ric"], errors="ignore")
    return df


# ── Portfolio / universe summaries ────────────────────────────────────────────

def portfolio_weighted_exposure(
    exposure_matrix: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Weighted-average style factor exposures of the portfolio.
    Weights are normalised internally; missing tickers are ignored.
    """
    cols   = [c for c in STYLE_FACTORS if c in exposure_matrix.columns]
    common = exposure_matrix.index.intersection(weights.index)
    if common.empty or not cols:
        return pd.Series(dtype=float)
    w = weights.loc[common]
    w = w / w.sum()
    return exposure_matrix.loc[common, cols].fillna(0).T.dot(w)


def universe_factor_stats(exposure_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and std of each style factor across the estimation universe.
    Useful as a market-level benchmark.
    """
    cols = [c for c in STYLE_FACTORS if c in exposure_matrix.columns]
    return pd.DataFrame({
        "Mean":    exposure_matrix[cols].mean(),
        "Std Dev": exposure_matrix[cols].std(),
        "# Valid": exposure_matrix[cols].count(),
    })


def sector_exposures(exposure_matrix: pd.DataFrame) -> pd.DataFrame:
    """Average style factor exposures grouped by GICS sector."""
    cols = [c for c in STYLE_FACTORS if c in exposure_matrix.columns]
    ind_cols = [c for c in exposure_matrix.columns if c.startswith("Industry_")]
    if not ind_cols:
        return pd.DataFrame()
    sector_series = exposure_matrix[ind_cols].idxmax(axis=1).str.replace("Industry_", "", regex=False)
    df = exposure_matrix[cols].copy()
    df["Sector"] = sector_series
    # Drop catch-all / unknown buckets
    df = df[~df["Sector"].isin(["Other", "Unknown", ""])]
    return df.groupby("Sector")[cols].mean().round(3)


# ── Factor return estimation (optional) ──────────────────────────────────────

def estimate_factor_returns(
    exposure_matrix: pd.DataFrame,
    total_returns: pd.Series,
    market_caps: pd.Series,
) -> tuple[pd.Series, object]:
    """
    WLS cross-sectional regression R = X*f + e, weights = sqrt(market_cap).
    Raises ValueError if there are not enough complete observations.
    """
    factor_cols = [c for c in exposure_matrix.columns
                   if c.startswith("Industry_") or c in STYLE_FACTORS]

    full = pd.concat([
        total_returns.rename("R"),
        exposure_matrix[factor_cols],
        market_caps.rename("mcap"),
    ], axis=1).dropna()

    if len(full) < len(factor_cols) + 2:
        raise ValueError(
            f"Only {len(full)} complete observations for {len(factor_cols)} factors. "
            "Backfill EOD prices with ingest_eod.py --start 2020-01-01 to enable this."
        )

    w   = np.sqrt(full["mcap"].clip(lower=0).values)
    y   = full["R"].astype(float).values
    X   = full[factor_cols].astype(float).values
    fit = sm.WLS(y, X, weights=w).fit()
    return pd.Series(fit.params, index=factor_cols), fit


# ── Attribution (requires factor returns) ────────────────────────────────────

def portfolio_attribution(
    exposure_matrix: pd.DataFrame,
    factor_returns: pd.Series,
    weights: pd.Series,
) -> dict:
    """Decompose portfolio return into factor contributions."""
    factor_cols = factor_returns.index
    common = exposure_matrix.index.intersection(weights.index)
    if common.empty:
        return {}
    w = weights.loc[common]
    w = w / w.sum()
    X_port               = exposure_matrix.loc[common, factor_cols].fillna(0)
    portfolio_exposure   = X_port.T.dot(w)
    factor_contributions = portfolio_exposure * factor_returns.reindex(portfolio_exposure.index).fillna(0)
    return {
        "factor_contributions": factor_contributions,
        "portfolio_exposure":   portfolio_exposure,
        "total_factor_return":  float(factor_contributions.sum()),
    }
