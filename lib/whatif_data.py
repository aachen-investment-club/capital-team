"""
yfinance loader for What-If scenarios.
Kept separate from lib/data.py because it hits a public API, not S3/DuckDB.
"""
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import date, timedelta


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_returns(ticker: str) -> pd.Series:
    """Daily close-to-close returns for *ticker* over the past year.

    Returns a Series indexed by date (timezone-naive), named after the ticker.
    Raises ValueError if yfinance returns no data.
    """
    end   = date.today()
    start = end - timedelta(days=365)

    raw = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned by yfinance for '{ticker}'.")

    close = raw["Close"].squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    returns = close.pct_change().dropna()
    returns.name = ticker
    return returns
