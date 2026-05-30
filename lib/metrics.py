import numpy as np
import pandas as pd


def cumulative_returns(returns: pd.DataFrame, return_col: str = "daily_return") -> pd.DataFrame:
    """Wide DataFrame (date index, ticker columns) of cumulative returns."""
    wide = returns.pivot(index="date", columns="ticker", values=return_col)
    return (1 + wide).cumprod() - 1


def correlation_matrix(returns: pd.DataFrame, return_col: str = "daily_return") -> pd.DataFrame:
    """Pairwise return correlation matrix."""
    wide = returns.pivot(index="date", columns="ticker", values=return_col)
    return wide.corr()


def monte_carlo_var(
    returns: pd.DataFrame,
    return_col: str = "daily_return",
    confidence: float = 0.95,
    horizon: int = 1,
    n_sim: int = 10_000,
) -> dict[str, float]:
    """
    Parametric Monte Carlo VaR per ticker at the given confidence level.
    Returns a dict of {ticker: VaR} where VaR is a negative number (loss).
    Stub: uses normal distribution fitted to historical returns.
    """
    wide = returns.pivot(index="date", columns="ticker", values=return_col).dropna()
    mu = wide.mean()
    sigma = wide.std()
    rng = np.random.default_rng(42)
    sims = rng.normal(mu.values, sigma.values, size=(n_sim, len(mu))) * np.sqrt(horizon)
    var = np.percentile(sims, (1 - confidence) * 100, axis=0)
    return dict(zip(mu.index.tolist(), var.tolist()))
