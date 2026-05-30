"""
THE DATA CONTRACT — all data access goes through here.
Pages must never import boto3, duckdb, or reference S3/file paths directly.
"""
import os
import pathlib

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_S3_BUCKET = os.getenv("S3_BUCKET", "")
_RAW_PREFIX = os.getenv("RAW_PREFIX", str(_ROOT / "data" / "raw"))
_DERIVED_PREFIX = os.getenv("DERIVED_PREFIX", str(_ROOT / "data" / "derived"))


def _path(prefix: str, table: str) -> str:
    if _S3_BUCKET:
        return f"s3://{_S3_BUCKET}/{prefix}/{table}.parquet"
    return f"{prefix}/{table}.parquet"


@st.cache_resource
def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    if _S3_BUCKET:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        region = os.getenv("AWS_REGION", "us-east-1")
        con.execute(f"SET s3_region='{region}';")
    return con


@st.cache_data(ttl=3600)
def get_returns(start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Daily returns. Columns: date, ticker, daily_return."""
    path = _path(_RAW_PREFIX, "returns")
    clauses = ["1=1"]
    if start_date:
        clauses.append(f"date >= '{start_date}'")
    if end_date:
        clauses.append(f"date <= '{end_date}'")
    where = " AND ".join(clauses)
    return _con().execute(
        f"SELECT date, ticker, daily_return FROM read_parquet('{path}') WHERE {where} ORDER BY date"
    ).df()


@st.cache_data(ttl=3600)
def get_positions() -> pd.DataFrame:
    """Current holdings. Columns: ticker, shares, price, weight, market_value."""
    path = _path(_RAW_PREFIX, "positions")
    return _con().execute(
        f"SELECT ticker, shares, price, weight, market_value FROM read_parquet('{path}') ORDER BY weight DESC"
    ).df()


@st.cache_data(ttl=3600)
def get_factor_betas() -> pd.DataFrame:
    """Current factor exposures. Columns: ticker, market_beta, value_beta, momentum_beta, quality_beta."""
    path = _path(_RAW_PREFIX, "factor_betas")
    return _con().execute(
        f"SELECT ticker, market_beta, value_beta, momentum_beta, quality_beta FROM read_parquet('{path}')"
    ).df()


@st.cache_data(ttl=3600)
def get_trade_log() -> pd.DataFrame:
    """Recent trades. Columns: date, ticker, action, shares, price, value."""
    path = _path(_RAW_PREFIX, "trade_log")
    return _con().execute(
        f"SELECT date, ticker, action, shares, price, value FROM read_parquet('{path}') ORDER BY date DESC"
    ).df()


@st.cache_data(ttl=3600)
def get_cumulative_returns() -> pd.DataFrame:
    """Precomputed cumulative returns. Columns: date, ticker, cumulative_return."""
    path = _path(_DERIVED_PREFIX, "cumulative_returns")
    return _con().execute(
        f"SELECT date, ticker, cumulative_return FROM read_parquet('{path}') ORDER BY date"
    ).df()


@st.cache_data(ttl=3600)
def get_rolling_vol() -> pd.DataFrame:
    """Precomputed 21-day rolling volatility (annualised). Columns: date, ticker, rolling_vol_21d."""
    path = _path(_DERIVED_PREFIX, "rolling_vol")
    return _con().execute(
        f"SELECT date, ticker, rolling_vol_21d FROM read_parquet('{path}') ORDER BY date"
    ).df()


# ── Mock data for Performance page ───────────────────────────────────────────
# (symbol, name, isin, ccy, category, theme, base_pct_nav, daily_mu, daily_sigma)
_POSITION_DEFS: list[tuple] = [
    ("AAPL",  "Apple Inc.",             "US0378331005", "USD", "Equities",    "Technology",   8.5,  0.00060, 0.0140),
    ("MSFT",  "Microsoft Corp.",        "US5949181045", "USD", "Equities",    "Technology",   7.2,  0.00050, 0.0120),
    ("NVDA",  "NVIDIA Corp.",           "US67066G1040", "USD", "Equities",    "Technology",   6.8,  0.00090, 0.0220),
    ("ASML",  "ASML Holding N.V.",      "NL0010273215", "EUR", "Equities",    "Technology",   5.1,  0.00040, 0.0130),
    ("GOOGL", "Alphabet Inc.",          "US02079K3059", "USD", "Equities",    "Technology",   4.9,  0.00040, 0.0120),
    ("JPM",   "JPMorgan Chase & Co.",   "US46625H1005", "USD", "Equities",    "Financials",   5.5,  0.00040, 0.0110),
    ("GS",    "Goldman Sachs Group",    "US38141G1040", "USD", "Equities",    "Financials",   3.8,  0.00030, 0.0120),
    ("MC.PA", "LVMH Moët Hennessy",     "FR0000121014", "EUR", "Equities",    "Consumer",     4.2,  0.00020, 0.0110),
    ("NESN",  "Nestlé S.A.",            "CH0012221716", "CHF", "Equities",    "Consumer",     3.5,  0.00010, 0.0090),
    ("XOM",   "Exxon Mobil Corp.",      "US30231G1022", "USD", "Equities",    "Energy",       3.2,  0.00030, 0.0130),
    ("BP.L",  "BP PLC",                "GB0007980591", "GBP", "Equities",    "Energy",       2.3,  0.00020, 0.0120),
    ("TLT",   "iShares 20+ Yr Tsy",    "US4642874329", "USD", "Bonds",       "Fixed Income", 12.0, 0.00010, 0.0060),
    ("HYG",   "iShares HY Corp Bond",  "US4642886034", "USD", "Bonds",       "Fixed Income",  5.5, 0.00020, 0.0040),
    ("GLD",   "SPDR Gold Shares",      "US78463V1070", "USD", "Commodities", "Commodities",   7.0, 0.00030, 0.0090),
    ("CASH",  "Cash & Equivalents",    "N/A",          "EUR", "Cash",        "Cash",         14.5, 0.00004, 0.0001),
]

_MOCK_DATES = pd.bdate_range(start="2024-01-02", end="2026-05-30")

_BENCHMARK_PARAMS: dict[str, tuple[float, float]] = {
    "SPY":        (0.00035, 0.0090),
    "QQQ":        (0.00055, 0.0130),
    "STOXX50":    (0.00025, 0.0095),
    "MSCI_WORLD": (0.00030, 0.0080),
}


def _mock_index(mu: float, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    daily = rng.normal(mu, sigma, len(_MOCK_DATES))
    return np.cumprod(1.0 + daily), daily


@st.cache_data
def get_portfolio_and_benchmarks() -> pd.DataFrame:
    """
    Portfolio + benchmark daily index values.
    Columns: date, ticker, index_value, daily_return
    ticker='PORTFOLIO' is the fund; SPY/QQQ/STOXX50/MSCI_WORLD are benchmarks.
    """
    rows: list[dict] = []
    idx, dr = _mock_index(0.00045, 0.0085, seed=0)
    for i, d in enumerate(_MOCK_DATES):
        rows.append({"date": d, "ticker": "PORTFOLIO", "index_value": float(idx[i]), "daily_return": float(dr[i])})
    for j, (bm, (mu, sigma)) in enumerate(_BENCHMARK_PARAMS.items()):
        idx, dr = _mock_index(mu, sigma, seed=j + 10)
        for i, d in enumerate(_MOCK_DATES):
            rows.append({"date": d, "ticker": bm, "index_value": float(idx[i]), "daily_return": float(dr[i])})
    return pd.DataFrame(rows)


@st.cache_data
def get_daily_weightings_history() -> pd.DataFrame:
    """
    Daily weightings for all positions over the full mock history (no theme column).
    Join with get_theme_mappings() to add theme/basket data.
    Columns: date, symbol, name, isin, ccy, category, pct_nav, cumulative_return
    cumulative_return is since-inception (relative to 2024-01-02).
    """
    rows: list[dict] = []
    for k, (sym, name, isin, ccy, cat, _, base_nav, mu, sigma) in enumerate(_POSITION_DEFS):
        idx, _ = _mock_index(mu, sigma, seed=k + 100)
        for i, d in enumerate(_MOCK_DATES):
            rows.append({
                "date": d, "symbol": sym, "name": name, "isin": isin,
                "ccy": ccy, "category": cat,
                "_raw": base_nav * float(idx[i]),
                "cumulative_return": float(idx[i]) - 1.0,
            })
    df = pd.DataFrame(rows)
    totals = df.groupby("date")["_raw"].transform("sum")
    df["pct_nav"] = df["_raw"] / totals * 100.0
    return df.drop(columns=["_raw"])


@st.cache_data(ttl=300)
def get_theme_mappings() -> pd.DataFrame:
    """
    Theme/basket assignments per symbol.
    Columns: symbol, theme

    TODO: replace mock with DynamoDB when table is ready.
    Production pattern (register with DuckDB for SQL JOIN):
        _con().register("theme_mappings", get_theme_mappings())
        df = _con().execute(
            "SELECT w.*, t.theme FROM read_parquet('{path}') w "
            "LEFT JOIN theme_mappings t ON w.symbol = t.symbol"
        ).df()
    """
    return pd.DataFrame(
        [{"symbol": s, "theme": t} for s, _, _, _, _, t, *_ in _POSITION_DEFS]
    )
    # ── DynamoDB path (enable when DDB_TABLE is configured) ──────────────────
    # import boto3
    # from boto3.dynamodb.conditions import Attr
    # ddb = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))
    # table = ddb.Table(os.getenv("DDB_TABLE"))
    # items: list[dict] = []
    # resp = table.scan(
    #     FilterExpression=Attr("symbol").exists() & Attr("theme").exists(),
    #     ProjectionExpression="symbol, #t",
    #     ExpressionAttributeNames={"#t": "theme"},
    # )
    # items.extend(resp["Items"])
    # while "LastEvaluatedKey" in resp:
    #     resp = table.scan(
    #         FilterExpression=Attr("symbol").exists() & Attr("theme").exists(),
    #         ProjectionExpression="symbol, #t",
    #         ExpressionAttributeNames={"#t": "theme"},
    #         ExclusiveStartKey=resp["LastEvaluatedKey"],
    #     )
    #     items.extend(resp["Items"])
    # return pd.DataFrame(items)[["symbol", "theme"]] if items else pd.DataFrame(columns=["symbol", "theme"])


@st.cache_data
def get_performance_trade_log() -> pd.DataFrame:
    """
    Mock trade log covering 2024-01 through 2026-05.
    Columns: date, symbol, name, action, shares, price, value
    """
    rng = np.random.default_rng(99)
    tradeable = [(s, n) for s, n, *_ in _POSITION_DEFS if s not in ("CASH", "TLT", "HYG")]
    rows: list[dict] = []
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
            rows.append({
                "date": pd.Timestamp(trade_day),
                "symbol": sym, "name": name,
                "action": "BUY" if rng.random() > 0.35 else "SELL",
                "shares": shares, "price": price,
                "value": round(shares * price, 2),
            })
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
