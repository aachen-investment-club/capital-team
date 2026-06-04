"""
THE DATA CONTRACT — all data access goes through here.
Pages must never import boto3, duckdb, or reference S3/file paths directly.
"""
import os
import pathlib

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

_S3_BUCKET      = os.getenv("S3_BUCKET", "")
_RAW_PREFIX     = os.getenv("RAW_PREFIX",     "history/raw"     if _S3_BUCKET else str(_ROOT / "data" / "raw"))
_DERIVED_PREFIX = os.getenv("DERIVED_PREFIX", "history/derived" if _S3_BUCKET else str(_ROOT / "data" / "derived"))
_AWS_REGION     = os.getenv("AWS_REGION", "eu-central-1")
_DDB_TABLE      = os.getenv("DDB_TABLE", "")


def _path(prefix: str, table: str) -> str:
    if _S3_BUCKET:
        return f"s3://{_S3_BUCKET}/{prefix}/{table}.parquet"
    return f"{prefix}/{table}.parquet"

@st.cache_resource(ttl=3600)
def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    if _S3_BUCKET:
        print(f"[data] connecting DuckDB → s3://{_S3_BUCKET} ({_AWS_REGION})")
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"SET s3_region='{_AWS_REGION}';")
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key and secret_key:
            con.execute(f"SET s3_access_key_id='{access_key}';")
            con.execute(f"SET s3_secret_access_key='{secret_key}';")
            session_token = os.getenv("AWS_SESSION_TOKEN")
            if session_token:
                con.execute(f"SET s3_session_token='{session_token}';")
            print("[data] credentials: env vars")
        else:
            import boto3
            creds = boto3.Session().get_credentials()
            if creds:
                creds = creds.get_frozen_credentials()
                con.execute(f"SET s3_access_key_id='{creds.access_key}';")
                con.execute(f"SET s3_secret_access_key='{creds.secret_key}';")
                if creds.token:
                    con.execute(f"SET s3_session_token='{creds.token}';")
                print("[data] credentials: boto3 session")
            else:
                print("[data] WARNING: no credentials found")
    else:
        print(f"[data] connecting DuckDB → local ({_DERIVED_PREFIX})")
    return con


def _log(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print(f"[data] {name}: EMPTY")
        return df
    date_col = next((c for c in ("date", "trade_date") if c in df.columns), None)
    if date_col:
        lo, hi = df[date_col].min(), df[date_col].max()
        print(f"[data] {name}: {len(df)} rows  {lo:%Y-%m-%d} → {hi:%Y-%m-%d}")
    else:
        print(f"[data] {name}: {len(df)} rows")
    return df


# ── Performance page loaders ──────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_portfolio_and_benchmarks() -> pd.DataFrame:
    """Portfolio + benchmark daily index values (normalised to 1.0 at inception).
    Columns: date, ticker, index_value, daily_return
    ticker = 'PORTFOLIO' | 'SPX' | 'MSCI_WORLD' | 'MSCI_EUROPE' | '60_40'
    """
    path = _path(_DERIVED_PREFIX, "portfolio_and_benchmarks")
    print(f"[data] loading portfolio_and_benchmarks from {path}")
    df = _con().execute(
        f"SELECT date, ticker, index_value, daily_return FROM read_parquet('{path}') ORDER BY ticker, date"
    ).df()
    return _log("portfolio_and_benchmarks", df)


@st.cache_data(ttl=3600)
def get_daily_weightings_history() -> pd.DataFrame:
    """Daily position weights + returns for all holdings including cash.
    Columns: date, symbol, name, isin, ccy, category, pct_nav,
             cumulative_return, daily_return
    Built by precompute/build_derived.py — refresh with:
        python -m precompute.build_derived
    """
    path = _path(_DERIVED_PREFIX, "daily_weightings")
    print(f"[data] loading daily_weightings from {path}")
    df = _con().execute(
        f"SELECT date, symbol, name, isin, ccy, category, pct_nav, cumulative_return, daily_return"
        f" FROM read_parquet('{path}') ORDER BY symbol, date"
    ).df()
    return _log("daily_weightings", df)


@st.cache_data(ttl=300)
def get_theme_mappings() -> pd.DataFrame:
    """Theme/basket assignment per symbol.
    Reads from DynamoDB (fund-baskets) when DDB_TABLE is set,
    otherwise falls back to category as proxy.
    Columns: symbol, theme
    """
    if _DDB_TABLE:
        return _theme_mappings_from_ddb()
    return pd.DataFrame(columns=["symbol", "theme"])


def _theme_mappings_from_ddb() -> pd.DataFrame:
    import boto3
    from boto3.dynamodb.conditions import Attr
    ddb  = boto3.resource("dynamodb", region_name=_AWS_REGION)
    tbl  = ddb.Table(_DDB_TABLE)
    items: list[dict] = []
    resp = tbl.scan(
        FilterExpression=Attr("active").eq(True),
        ProjectionExpression="symbol, theme",
    )
    items.extend(resp["Items"])
    while "LastEvaluatedKey" in resp:
        resp = tbl.scan(
            FilterExpression=Attr("active").eq(True),
            ProjectionExpression="symbol, theme",
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        items.extend(resp["Items"])
    return pd.DataFrame(items)[["symbol", "theme"]] if items else pd.DataFrame(columns=["symbol", "theme"])


@st.cache_data(ttl=3600)
def get_trade_log() -> pd.DataFrame:
    """Trade history from IBKR Flex Query, with same-day fills merged per symbol.
    Fractional-share fills (0 commission) are combined with the main order so each
    row represents one trade event, not individual broker fills.

    Columns: trade_date, symbol, name, isin, currency, asset_type,
             buy_sell, quantity, entry_exit_price, effective_price, proceeds, commission
    """
    from lib.ibkr import load_trade_log
    raw = load_trade_log()
    agg = (
        raw.groupby(
            ["trade_date", "symbol", "name", "isin", "currency", "asset_type", "buy_sell"],
            as_index=False,
        )
        .agg(quantity=("quantity", "sum"),
             proceeds=("proceeds", "sum"),
             commission=("commission", "sum"))
    )
    # Execution price per share (proceeds are negative for buys, positive for sells)
    agg["entry_exit_price"] = (-agg["proceeds"] / agg["quantity"]).round(4)
    # Effective price: execution price adjusted for transaction cost per share
    agg["effective_price"] = ((-agg["proceeds"] + agg["commission"].abs()) / agg["quantity"]).round(4)
    return _log("trade_log", agg.sort_values("trade_date", ascending=False).reset_index(drop=True))
