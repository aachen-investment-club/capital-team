import pandas as pd


def cumulative_returns(returns: pd.DataFrame, return_col: str = "daily_return") -> pd.DataFrame:
    """Wide DataFrame (date index, ticker columns) of cumulative returns."""
    wide = returns.pivot(index="date", columns="ticker", values=return_col)
    return (1 + wide).cumprod() - 1
