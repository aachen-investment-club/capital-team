"""
THE DATA CONTRACT — all data access goes through here.
Pages must never import boto3, duckdb, or reference S3/file paths directly.
"""
import os
import pathlib

import duckdb
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
