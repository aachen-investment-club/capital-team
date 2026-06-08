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
