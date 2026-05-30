"""
Generate realistic placeholder Parquet data in ./data/raw/.
Run once before first launch: python scripts/generate_placeholder_data.py
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "BRK.B"]
PRICES = [190.0, 415.0, 170.0, 195.0, 510.0, 875.0, 245.0, 205.0, 155.0, 395.0]

rng = np.random.default_rng(42)

# ── Trading dates (≈2 years) ────────────────────────────────────────────────
dates = pd.bdate_range("2023-01-03", "2024-12-31")
n_days, n_tickers = len(dates), len(TICKERS)

# ── Correlated daily returns ─────────────────────────────────────────────────
market = rng.normal(0.0003, 0.009, n_days)
betas_arr = rng.uniform(0.6, 1.4, n_tickers)
idio_vol = rng.uniform(0.007, 0.018, n_tickers)
idio = rng.normal(0, 1, (n_days, n_tickers)) * idio_vol
ret_matrix = market[:, None] * betas_arr[None, :] + idio

returns_df = pd.DataFrame(
    [(dates[i].date(), TICKERS[j], float(ret_matrix[i, j]))
     for i in range(n_days) for j in range(n_tickers)],
    columns=["date", "ticker", "daily_return"],
)
returns_df.to_parquet(RAW_DIR / "returns.parquet", index=False)
print(f"returns.parquet     {len(returns_df):>7,} rows")

# ── Positions ────────────────────────────────────────────────────────────────
weights = rng.dirichlet(np.ones(n_tickers) * 3)
total_nav = 5_000_000.0
market_values = weights * total_nav
shares = np.round(market_values / PRICES)

positions_df = pd.DataFrame({
    "ticker": TICKERS,
    "shares": shares,
    "price": PRICES,
    "weight": np.round(weights, 6),
    "market_value": np.round(market_values, 2),
})
positions_df.to_parquet(RAW_DIR / "positions.parquet", index=False)
print(f"positions.parquet   {len(positions_df):>7,} rows")

# ── Factor betas ─────────────────────────────────────────────────────────────
factor_betas_df = pd.DataFrame({
    "ticker": TICKERS,
    "market_beta": np.round(betas_arr, 3),
    "value_beta": np.round(rng.uniform(-0.5, 0.5, n_tickers), 3),
    "momentum_beta": np.round(rng.uniform(-0.3, 0.3, n_tickers), 3),
    "quality_beta": np.round(rng.uniform(-0.2, 0.4, n_tickers), 3),
})
factor_betas_df.to_parquet(RAW_DIR / "factor_betas.parquet", index=False)
print(f"factor_betas.parquet{len(factor_betas_df):>7,} rows")

# ── Trade log ────────────────────────────────────────────────────────────────
trade_dates = pd.bdate_range("2024-01-02", "2024-12-31", freq="W-WED")
trade_tickers = rng.choice(TICKERS, size=len(trade_dates))
trade_actions = rng.choice(["BUY", "SELL"], size=len(trade_dates), p=[0.6, 0.4])
trade_shares = rng.integers(10, 200, size=len(trade_dates)).astype(float)
ticker_price = dict(zip(TICKERS, PRICES))
trade_prices = np.array([ticker_price[t] * (1 + rng.normal(0, 0.005)) for t in trade_tickers])

trade_log_df = pd.DataFrame({
    "date": [d.date() for d in trade_dates],
    "ticker": trade_tickers,
    "action": trade_actions,
    "shares": trade_shares,
    "price": np.round(trade_prices, 2),
    "value": np.round(trade_shares * trade_prices, 2),
})
trade_log_df.to_parquet(RAW_DIR / "trade_log.parquet", index=False)
print(f"trade_log.parquet   {len(trade_log_df):>7,} rows")

print(f"\nAll data written to {RAW_DIR}")
