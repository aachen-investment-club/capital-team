"""
Generate realistic placeholder Parquet data in ./data/raw/.

Run once before first launch:  python scripts/generate_placeholder_data.py
Then build derived tables:     python -m precompute.build_derived

Raw tables produced
───────────────────
portfolio_and_benchmarks.parquet  date, ticker, index_value, daily_return
instrument_metadata.parquet       symbol, name, isin, ccy, category, theme
daily_positions.parquet           date, symbol, pct_nav
position_returns.parquet          date, symbol, daily_return
trade_log.parquet                 date, symbol, name, action, shares, price, value
returns.parquet                   date, ticker, daily_return          (legacy — kept for factor model loaders)
positions.parquet                 ticker, shares, price, weight, market_value (legacy)
factor_betas.parquet              ticker, market_beta, …              (legacy)
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# (symbol, name, isin, ccy, category, theme, base_pct_nav, daily_mu, daily_sigma)
POSITION_DEFS: list[tuple] = [
    ("AAPL",  "Apple Inc.",            "US0378331005", "USD", "Equities",    "Technology",    8.5, 0.00060, 0.0140),
    ("MSFT",  "Microsoft Corp.",       "US5949181045", "USD", "Equities",    "Technology",    7.2, 0.00050, 0.0120),
    ("NVDA",  "NVIDIA Corp.",          "US67066G1040", "USD", "Equities",    "Technology",    6.8, 0.00090, 0.0220),
    ("ASML",  "ASML Holding N.V.",     "NL0010273215", "EUR", "Equities",    "Technology",    5.1, 0.00040, 0.0130),
    ("GOOGL", "Alphabet Inc.",         "US02079K3059", "USD", "Equities",    "Technology",    4.9, 0.00040, 0.0120),
    ("JPM",   "JPMorgan Chase & Co.", "US46625H1005", "USD", "Equities",    "Financials",    5.5, 0.00040, 0.0110),
    ("GS",    "Goldman Sachs Group",  "US38141G1040", "USD", "Equities",    "Financials",    3.8, 0.00030, 0.0120),
    ("MC.PA", "LVMH Moët Hennessy",   "FR0000121014", "EUR", "Equities",    "Consumer",      4.2, 0.00020, 0.0110),
    ("NESN",  "Nestlé S.A.",          "CH0012221716", "CHF", "Equities",    "Consumer",      3.5, 0.00010, 0.0090),
    ("XOM",   "Exxon Mobil Corp.",    "US30231G1022", "USD", "Equities",    "Energy",        3.2, 0.00030, 0.0130),
    ("BP.L",  "BP PLC",               "GB0007980591", "GBP", "Equities",    "Energy",        2.3, 0.00020, 0.0120),
    ("TLT",   "iShares 20+ Yr Tsy",  "US4642874329", "USD", "Bonds",       "Fixed Income", 12.0, 0.00010, 0.0060),
    ("HYG",   "iShares HY Corp Bond","US4642886034", "USD", "Bonds",       "Fixed Income",  5.5, 0.00020, 0.0040),
    ("GLD",   "SPDR Gold Shares",    "US78463V1070", "USD", "Commodities", "Commodities",   7.0, 0.00030, 0.0090),
    ("CASH",  "Cash & Equivalents",  "N/A",          "EUR", "Cash",        "Cash",          14.5, 0.00004, 0.0001),
]

BENCHMARK_PARAMS: dict[str, tuple[float, float]] = {
    "SPY":        (0.00035, 0.0090),
    "QQQ":        (0.00055, 0.0130),
    "STOXX50":    (0.00025, 0.0095),
    "MSCI_WORLD": (0.00030, 0.0080),
}

DATES = pd.bdate_range(start="2024-01-02", end="2026-05-30")


def _sim(mu: float, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    daily = rng.normal(mu, sigma, len(DATES))
    return daily, np.cumprod(1.0 + daily)


# ── instrument_metadata.parquet ──────────────────────────────────────────────
meta_df = pd.DataFrame([
    {"symbol": sym, "name": name, "isin": isin, "ccy": ccy, "category": cat, "theme": theme}
    for sym, name, isin, ccy, cat, theme, *_ in POSITION_DEFS
])
meta_df.to_parquet(RAW_DIR / "instrument_metadata.parquet", index=False)
print(f"instrument_metadata.parquet       {len(meta_df):>5,} rows")

# ── position_returns.parquet ─────────────────────────────────────────────────
# Simulate independent GBM for each position (seed = k + 100 to match original mock)
pos_daily: dict[str, np.ndarray] = {}
pos_cumidx: dict[str, np.ndarray] = {}

pos_return_rows = []
for k, (sym, *_, mu, sigma) in enumerate(POSITION_DEFS):
    daily, cumidx = _sim(mu, sigma, seed=k + 100)
    pos_daily[sym] = daily
    pos_cumidx[sym] = cumidx
    for i, d in enumerate(DATES):
        pos_return_rows.append({"date": d, "symbol": sym, "daily_return": float(daily[i])})

position_returns_df = pd.DataFrame(pos_return_rows)
position_returns_df.to_parquet(RAW_DIR / "position_returns.parquet", index=False)
print(f"position_returns.parquet          {len(position_returns_df):>5,} rows")

# ── daily_positions.parquet ──────────────────────────────────────────────────
# pct_nav evolves proportionally with position returns; normalised so total = 100% each day
pos_rows = []
for i, d in enumerate(DATES):
    raw = {
        sym: POSITION_DEFS[k][6] * pos_cumidx[sym][i]
        for k, (sym, *_) in enumerate(POSITION_DEFS)
    }
    total = sum(raw.values())
    for sym, val in raw.items():
        pos_rows.append({"date": d, "symbol": sym, "pct_nav": val / total * 100.0})

daily_positions_df = pd.DataFrame(pos_rows)
daily_positions_df.to_parquet(RAW_DIR / "daily_positions.parquet", index=False)
print(f"daily_positions.parquet           {len(daily_positions_df):>5,} rows")

# ── portfolio_and_benchmarks.parquet ─────────────────────────────────────────
pb_rows = []

# Portfolio: base-weight-average of position daily returns
base_weights = np.array([p[6] for p in POSITION_DEFS])
base_weights = base_weights / base_weights.sum()
all_daily = np.column_stack([pos_daily[sym] for sym, *_ in POSITION_DEFS])
port_daily = all_daily @ base_weights
port_index = np.cumprod(1.0 + port_daily)
for i, d in enumerate(DATES):
    pb_rows.append({
        "date": d, "ticker": "PORTFOLIO",
        "index_value": float(port_index[i]),
        "daily_return": float(port_daily[i]),
    })

for j, (bm, (mu, sigma)) in enumerate(BENCHMARK_PARAMS.items()):
    daily, cumidx = _sim(mu, sigma, seed=j + 10)
    for i, d in enumerate(DATES):
        pb_rows.append({
            "date": d, "ticker": bm,
            "index_value": float(cumidx[i]),
            "daily_return": float(daily[i]),
        })

portfolio_and_benchmarks_df = pd.DataFrame(pb_rows)
portfolio_and_benchmarks_df.to_parquet(RAW_DIR / "portfolio_and_benchmarks.parquet", index=False)
print(f"portfolio_and_benchmarks.parquet  {len(portfolio_and_benchmarks_df):>5,} rows")

# ── trade_log.parquet ────────────────────────────────────────────────────────
rng = np.random.default_rng(99)
tradeable = [(sym, name) for sym, name, *_ in POSITION_DEFS if sym not in ("CASH", "TLT", "HYG")]
trade_rows = []
for month_start in pd.date_range("2024-01-01", "2026-05-01", freq="MS"):
    month_end = month_start + pd.offsets.MonthEnd(0)
    bdays = pd.bdate_range(start=month_start, end=month_end)
    if len(bdays) == 0:
        continue
    n_trades = int(rng.integers(2, 5))
    chosen_idx = rng.choice(len(bdays), size=min(n_trades, len(bdays)), replace=False)
    for trade_day in sorted(bdays[chosen_idx]):
        sym, name = tradeable[int(rng.integers(0, len(tradeable)))]
        shares = int(rng.integers(50, 500))
        price = round(float(rng.uniform(20.0, 500.0)), 2)
        trade_rows.append({
            "date": pd.Timestamp(trade_day),
            "symbol": sym, "name": name,
            "action": "BUY" if rng.random() > 0.35 else "SELL",
            "shares": shares, "price": price,
            "value": round(shares * price, 2),
        })

trade_log_df = (
    pd.DataFrame(trade_rows)
    .sort_values("date", ascending=False)
    .reset_index(drop=True)
)
trade_log_df.to_parquet(RAW_DIR / "trade_log.parquet", index=False)
print(f"trade_log.parquet                 {len(trade_log_df):>5,} rows")

# ── Legacy tables (kept for factor model loaders) ─────────────────────────────
LEGACY_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "BRK.B"]
LEGACY_PRICES  = [190.0,  415.0,  170.0,  195.0,  510.0,  875.0,  245.0,  205.0, 155.0,  395.0]
legacy_rng = np.random.default_rng(42)
legacy_dates = pd.bdate_range("2023-01-03", "2024-12-31")
n_days, n_t = len(legacy_dates), len(LEGACY_TICKERS)

market = legacy_rng.normal(0.0003, 0.009, n_days)
betas_arr = legacy_rng.uniform(0.6, 1.4, n_t)
idio = legacy_rng.normal(0, 1, (n_days, n_t)) * legacy_rng.uniform(0.007, 0.018, n_t)
ret_matrix = market[:, None] * betas_arr[None, :] + idio

returns_df = pd.DataFrame(
    [(legacy_dates[i].date(), LEGACY_TICKERS[j], float(ret_matrix[i, j]))
     for i in range(n_days) for j in range(n_t)],
    columns=["date", "ticker", "daily_return"],
)
returns_df.to_parquet(RAW_DIR / "returns.parquet", index=False)
print(f"returns.parquet (legacy)          {len(returns_df):>5,} rows")

weights = legacy_rng.dirichlet(np.ones(n_t) * 3)
total_nav = 5_000_000.0
market_values = weights * total_nav
pd.DataFrame({
    "ticker": LEGACY_TICKERS,
    "shares": np.round(market_values / LEGACY_PRICES),
    "price": LEGACY_PRICES,
    "weight": np.round(weights, 6),
    "market_value": np.round(market_values, 2),
}).to_parquet(RAW_DIR / "positions.parquet", index=False)
print(f"positions.parquet (legacy)        {n_t:>5,} rows")

pd.DataFrame({
    "ticker": LEGACY_TICKERS,
    "market_beta": np.round(betas_arr, 3),
    "value_beta": np.round(legacy_rng.uniform(-0.5, 0.5, n_t), 3),
    "momentum_beta": np.round(legacy_rng.uniform(-0.3, 0.3, n_t), 3),
    "quality_beta": np.round(legacy_rng.uniform(-0.2, 0.4, n_t), 3),
}).to_parquet(RAW_DIR / "factor_betas.parquet", index=False)
print(f"factor_betas.parquet (legacy)     {n_t:>5,} rows")

print(f"\nAll data written to {RAW_DIR}")
print("Next: python -m precompute.build_derived")
