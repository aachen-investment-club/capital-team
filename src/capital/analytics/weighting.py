"""
Portfolio optimisation — efficient frontier, weight generation, and stats.
All calculation logic lives here; pages must not import cvxpy directly.
"""
import numpy as np
import pandas as pd
import cvxpy as cvx


# ── Data preparation ──────────────────────────────────────────────────────────

def build_basket_stats(
    security_master: pd.DataFrame,
    eod_prices_dict: dict,
    bench_daily_returns: pd.Series,
    backtest_from,
) -> list[dict]:
    """Build per-security stat dicts for use by the optimiser and frontier.

    Parameters
    ----------
    security_master     : DataFrame with columns security_id, ticker, name
    eod_prices_dict     : {security_id: DataFrame(date, adj_close)}
    bench_daily_returns : Series indexed by date — benchmark (e.g. SPX) daily returns
    backtest_from       : date — history window start (inclusive)

    Returns
    -------
    List of dicts: Name, Symbol, Daily Rets, Ann Return, Ann Vol, Beta
    Excludes securities with fewer than 20 trading days of data.
    """
    stats = []
    bench = bench_daily_returns.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench[bench.index >= pd.Timestamp(backtest_from)]

    for _, row in security_master.iterrows():
        sid = row["security_id"]
        df = eod_prices_dict.get(sid)
        if df is None or df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= pd.Timestamp(backtest_from)].sort_values("date")

        if len(df) < 20:
            continue

        price_col = "adj_close" if "adj_close" in df.columns else "close"
        rets = df.set_index("date")[price_col].pct_change().dropna()
        rets.name = row["ticker"]

        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)

        # Beta vs benchmark — align on common dates
        common = rets.index.intersection(bench.index)
        if len(common) >= 20:
            r_sec = rets.loc[common].values
            r_bm  = bench.loc[common].values
            cov_matrix = np.cov(r_sec, r_bm)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
        else:
            beta = 1.0

        stats.append({
            "Name":       row.get("name", row["ticker"]),
            "Symbol":     row["ticker"],
            "Daily Rets": rets,
            "Ann Return": ann_ret,
            "Ann Vol":    ann_vol,
            "Beta":       beta,
        })

    return stats


# ── Optimiser ─────────────────────────────────────────────────────────────────

def get_weights(
    basket_stats: list[dict],
    max_beta: float = 1.0,
    risk_aversion: float = 0.5,
) -> np.ndarray:
    """Mean-variance optimisation with a relaxation loop for infeasibility.

    Constraints (relaxed iteratively if infeasible):
      - Fully invested: sum(w) == 1
      - Long-only: w >= 0.05
      - Concentration cap: w <= 0.25 (relaxes to 0.35)
      - Portfolio beta: betas @ w <= max_beta
    """
    n = len(basket_stats)
    if n == 0:
        raise ValueError("No securities available for optimisation.")

    returns = np.array([b["Ann Return"] for b in basket_stats])
    betas   = np.array([b["Beta"]       for b in basket_stats])

    daily_rets_df = pd.concat([b["Daily Rets"] for b in basket_stats], axis=1).dropna()
    cov = daily_rets_df.cov().values

    w = cvx.Variable(n)
    risk_penalty = cvx.quad_form(w, cov)

    current_max_beta = max_beta
    current_ra       = risk_aversion
    current_max_w    = 0.25

    for attempt in range(25):
        constraints = [
            cvx.sum(w) == 1,
            w <= current_max_w,
            w >= 0.05,
            betas @ w <= current_max_beta,
        ]
        prob = cvx.Problem(cvx.Maximize(returns @ w - current_ra * risk_penalty), constraints)
        prob.solve()

        if w.value is not None:
            return w.value

        current_max_beta += 0.1
        current_max_w    += 0.05
        current_ra        = max(0.01, current_ra - 0.01)

    raise ValueError(
        "Optimisation infeasible after 25 relaxation attempts. "
        "Check that EOD data is available and contains no NaNs."
    )


# ── Efficient frontier ────────────────────────────────────────────────────────

def efficient_frontier(
    basket_stats: list[dict],
    n_points: int = 40,
) -> tuple[list[float], list[float]]:
    """Compute the efficient frontier by sweeping risk-aversion values.

    Returns (vols, rets) — both annualised, sorted by vol ascending.
    Uses a long-only, fully-invested constraint (no beta cap on the frontier).
    """
    daily_rets_df = pd.concat(
        [b["Daily Rets"] for b in basket_stats], axis=1
    ).dropna()
    mu = daily_rets_df.mean().values * 252
    S  = daily_rets_df.cov().values * 252
    n  = len(mu)

    frontier_vols = []
    frontier_rets = []

    for ra in np.logspace(-3, 2, n_points):
        w = cvx.Variable(n)
        prob = cvx.Problem(
            cvx.Maximize(mu @ w - ra * cvx.quad_form(w, S)),
            [cvx.sum(w) == 1, w >= 0],
        )
        prob.solve()
        if w.value is not None:
            r = float(mu @ w.value)
            v = float(np.sqrt(w.value.T @ S @ w.value))
            frontier_rets.append(r)
            frontier_vols.append(v)

    if not frontier_vols:
        return [], []

    points = sorted(zip(frontier_vols, frontier_rets))
    vols, rets = zip(*points)
    return list(vols), list(rets)


# ── Portfolio statistics ──────────────────────────────────────────────────────

def portfolio_stats(
    basket_stats: list[dict],
    weights: np.ndarray,
) -> dict:
    """Return annualised vol, return, and beta for a weighted portfolio."""
    daily_rets_df = pd.concat(
        [b["Daily Rets"] for b in basket_stats], axis=1
    ).dropna()
    w = np.array(weights)
    mu = daily_rets_df.mean().values * 252
    S  = daily_rets_df.cov().values * 252
    betas = np.array([b["Beta"] for b in basket_stats])

    return {
        "Ann Return": float(mu @ w),
        "Ann Vol":    float(np.sqrt(w.T @ S @ w)),
        "Beta":       float(betas @ w),
    }


# ── Scheme comparison ─────────────────────────────────────────────────────────

def build_price_matrix(
    security_master: pd.DataFrame,
    get_eod_prices_fn,
    cache_version: str,
) -> pd.DataFrame:
    """Daily adj_close for all active non-index securities over the last 1 year."""
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=1)
    cols = {}
    for _, row in security_master.iterrows():
        eod = get_eod_prices_fn(row["security_id"], cache_version)
        if eod.empty:
            continue
        eod = eod.copy()
        eod["date"] = pd.to_datetime(eod["date"])
        col = "adj_close" if "adj_close" in eod.columns else "close"
        s = eod[eod["date"] >= cutoff].set_index("date")[col].dropna()
        if len(s) >= 30:
            cols[row["ticker"]] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index().ffill()


def scheme_nav(
    prices: pd.DataFrame,
    scheme: str,
    rebal_freq: str | None,
    spx_rets: pd.Series | None = None,
) -> pd.Series:
    """Simulate a portfolio under a given weighting scheme.

    Parameters
    ----------
    prices     : daily price matrix (tickers as columns)
    scheme     : 'equal' | 'inv_vol' | 'momentum' | 'beta'
    rebal_freq : pandas offset string ('B', 'W-FRI', 'MS') or None for buy-and-hold
    spx_rets   : required for scheme='beta' (rolling 63d beta, weight inversely proportional)
    """
    prices = prices.astype(float)
    if prices.empty:
        return pd.Series(dtype=float)

    rets = prices.pct_change().fillna(0)
    n    = len(prices.columns)
    nav  = pd.Series(index=prices.index, dtype=float)
    nav.iloc[0] = 1.0

    rebal_set = (
        set(pd.date_range(prices.index[0], prices.index[-1], freq=rebal_freq).normalize())
        if rebal_freq else set()
    )

    spx_aligned = None
    if scheme == "beta" and spx_rets is not None:
        spx_aligned = spx_rets.reindex(rets.index).fillna(0)

    current_w = None

    for i in range(1, len(prices)):
        dt = prices.index[i].normalize()

        if current_w is None or (rebal_freq and dt in rebal_set):
            lookback      = prices.iloc[max(0, i - 63):i]
            lookback_rets = rets.iloc[max(0, i - 63):i]

            if scheme == "equal":
                w = np.ones(n) / n

            elif scheme == "inv_vol":
                vols = lookback.pct_change().std().replace(0, np.nan)
                inv  = 1.0 / vols
                w    = (inv / inv.sum()).fillna(1 / n).values

            elif scheme == "momentum":
                ret_63  = (lookback.iloc[-1] / lookback.iloc[0] - 1).replace([np.inf, -np.inf], np.nan)
                clipped = ret_63.clip(lower=0).fillna(0)
                total   = clipped.sum()
                w = (clipped / total).values if total > 0 else np.ones(n) / n

            elif scheme == "beta" and spx_aligned is not None:
                spx_lb  = spx_aligned.iloc[max(0, i - 63):i]
                spx_var = float(spx_lb.var())
                betas   = pd.Series(index=prices.columns, dtype=float)
                for col in prices.columns:
                    cov = float(lookback_rets[col].cov(spx_lb))
                    betas[col] = cov / spx_var if spx_var > 1e-12 else 0.0
                betas    = betas.clip(lower=1e-6)
                inv_beta = (1.0 / betas).replace([np.inf, -np.inf], np.nan)
                total    = inv_beta.sum()
                w = (inv_beta / total).fillna(1 / n).values if total > 0 else np.ones(n) / n

            else:
                w = np.ones(n) / n

            current_w = w

        nav.iloc[i] = nav.iloc[i - 1] * (1 + float(np.dot(current_w, rets.iloc[i].values)))

    return nav.rename(scheme)


def scheme_nav_fixed(prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Simulate a buy-and-hold portfolio with fixed static weights (no rebalancing)."""
    prices = prices.astype(float)
    rets   = prices.pct_change().fillna(0)
    w      = np.array(weights)
    if w.sum() > 0:
        w = w / w.sum()
    nav = pd.Series(index=prices.index, dtype=float)
    nav.iloc[0] = 1.0
    for i in range(1, len(prices)):
        nav.iloc[i] = nav.iloc[i - 1] * (1 + float(np.dot(w, rets.iloc[i].values)))
    return nav


def scheme_stats(nav: pd.Series, label: str) -> dict:
    """Return summary stats for a NAV series."""
    s       = nav.dropna()
    rets    = s.pct_change().dropna()
    ann_ret = float(rets.mean() * 252)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    dd      = float((s / s.cummax() - 1).min())
    total   = float(s.iloc[-1] / s.iloc[0] - 1)
    return {
        "Scheme":       label,
        "Total return": total,
        "Ann return":   ann_ret,
        "Ann vol":      ann_vol,
        "Sharpe":       sharpe,
        "Max drawdown": dd,
    }
