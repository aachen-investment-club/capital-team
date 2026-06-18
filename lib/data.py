"""
THE DATA CONTRACT — all data access goes through here.
Pages must never import boto3, duckdb, or reference S3/file paths directly.
"""
import json
import os
import pathlib

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

_S3_BUCKET        = os.getenv("S3_BUCKET", "")
_EOD_PREFIX       = "history/eod_prices"        if _S3_BUCKET else str(_ROOT / "data" / "eod_prices")
_PORTFOLIO_PREFIX = "history/portfolio/derived"  if _S3_BUCKET else str(_ROOT / "data" / "derived")
_AWS_REGION       = os.getenv("AWS_REGION", "eu-central-1")
_DDB_TABLE        = os.getenv("DDB_TABLE", "")
_CONFIG_DIR       = _ROOT / "config"


def _path(prefix: str, table: str, ext: str = "parquet") -> str:
    if _S3_BUCKET:
        return f"s3://{_S3_BUCKET}/{prefix}/{table}.{ext}"
    return f"{prefix}/{table}.{ext}"



@st.cache_resource(ttl=3600)
def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    if _S3_BUCKET:
        print(f"[data] connecting DuckDB -> s3://{_S3_BUCKET} ({_AWS_REGION})")
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
        print(f"[data] connecting DuckDB -> local ({_PORTFOLIO_PREFIX})")
    return con


def _log(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print(f"[data] {name}: EMPTY")
        return df
    date_col = next((c for c in ("date", "trade_date") if c in df.columns), None)
    if date_col:
        lo, hi = df[date_col].min(), df[date_col].max()
        print(f"[data] {name}: {len(df)} rows  {lo} -> {hi}")
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
    path = _path(_PORTFOLIO_PREFIX, "portfolio_and_benchmarks", "json")
    print(f"[data] loading portfolio_and_benchmarks from {path}")
    df = _con().execute(
        f"SELECT date, ticker, index_value, daily_return"
        f" FROM read_json_auto('{path}') ORDER BY ticker, date"
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    return _log("portfolio_and_benchmarks", df)


@st.cache_data(ttl=3600)
def get_daily_weightings_history() -> pd.DataFrame:
    """Daily position weights + returns for all holdings including cash.
    Columns: date, symbol, name, isin, ccy, category, pct_nav,
             cumulative_return, daily_return
    """
    path = _path(_PORTFOLIO_PREFIX, "daily_weightings", "json")
    print(f"[data] loading daily_weightings from {path}")
    df = _con().execute(
        f"SELECT date, symbol, name, isin, ccy, category, pct_nav,"
        f" cumulative_return, daily_return"
        f" FROM read_json_auto('{path}') ORDER BY symbol, date"
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    return _log("daily_weightings", df)


@st.cache_data(ttl=300)
def get_theme_mappings() -> pd.DataFrame:
    """Theme/basket assignment per symbol.
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
    return (
        pd.DataFrame(items)[["symbol", "theme"]]
        if items
        else pd.DataFrame(columns=["symbol", "theme"])
    )


# ── EOD price loaders ─────────────────────────────────────────────────────────

def _eod_path(security_id: str) -> str:
    partition = f"security_id={security_id}/data.parquet"
    if _S3_BUCKET:
        return f"s3://{_S3_BUCKET}/{_EOD_PREFIX}/{partition}"
    return str(pathlib.Path(_EOD_PREFIX) / partition)


@st.cache_data(ttl=60)
def _eod_data_version() -> str:
    """Return the ingestion timestamp from _version.json (refreshes every 60s)."""
    try:
        if _S3_BUCKET:
            import boto3
            s3  = boto3.client("s3", region_name=_AWS_REGION)
            obj = s3.get_object(Bucket=_S3_BUCKET, Key=f"{_EOD_PREFIX}/_version.json")
            data = json.loads(obj["Body"].read())
        else:
            p = pathlib.Path(_EOD_PREFIX) / "_version.json"
            if not p.exists():
                return ""
            data = json.loads(p.read_text())
        return data.get("ingested_at", "")
    except Exception:
        return ""


@st.cache_data(ttl=3600)
def get_eod_prices(security_id: str, cache_version: str = "") -> pd.DataFrame:
    """Daily EOD OHLCV for one security, sorted by date ascending.
    Columns: security_id, date, open, high, low, close, adj_close, volume
    Returns an empty DataFrame (no error) if no data has been ingested yet.
    """
    path = _eod_path(security_id)
    try:
        df = _con().execute(
            f"SELECT * FROM read_parquet('{path}') ORDER BY date"
        ).df()
    except Exception:
        return pd.DataFrame(
            columns=["security_id", "date", "open", "high", "low",
                     "close", "adj_close", "volume"]
        )
    return _log("eod_prices", df)


@st.cache_data(ttl=3600)
def get_security_master() -> pd.DataFrame:
    """Active securities from security_master.csv (S3 when deployed, local otherwise).
    Columns: security_id, ric, ticker, isin, name, currency, asset_type
    Sorted by ticker.
    """
    import io as _io
    if _S3_BUCKET:
        try:
            import boto3 as _boto3
            s3c = _boto3.client("s3", region_name=_AWS_REGION)
            obj = s3c.get_object(Bucket=_S3_BUCKET, Key="config/security_master.csv")
            raw = _io.BytesIO(obj["Body"].read())
            df  = pd.read_csv(raw, dtype=str)
        except Exception as e:
            print(f"[data] S3 security_master failed ({e}), falling back to local")
            df = pd.read_csv(_CONFIG_DIR / "security_master.csv", dtype=str)
    else:
        csv_path = _CONFIG_DIR / "security_master.csv"
        if not csv_path.exists():
            return pd.DataFrame(
                columns=["security_id", "ric", "ticker", "isin",
                         "name", "currency", "asset_type"]
            )
        df = pd.read_csv(csv_path, dtype=str)

    df["active"] = df["active"].str.lower().isin(("true", "1", "yes"))
    df = (
        df[df["active"]]
        .drop(columns=["active"])
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    return _log("security_master", df)


@st.cache_data(ttl=3600)
def get_fundamentals() -> pd.DataFrame:
    """Daily fundamentals snapshot for the Barra universe.
    Columns: date, ric, ticker, gics_sector, pb_ratio, market_cap, shares_outstanding
    Populated by scripts/ingest_fundamentals.py.
    Returns empty DataFrame if not yet ingested.
    """
    if _S3_BUCKET:
        path = f"s3://{_S3_BUCKET}/history/fundamentals/data.parquet"
    else:
        path = str(_ROOT / "data" / "fundamentals" / "data.parquet")
    try:
        df = _con().execute(
            f"SELECT * FROM read_parquet('{path}') ORDER BY ric, date"
        ).df()
        df["date"] = pd.to_datetime(df["date"])
        return _log("fundamentals", df)
    except Exception:
        return pd.DataFrame(
            columns=["date", "ric", "ticker", "gics_sector",
                     "pb_ratio", "market_cap", "shares_outstanding"]
        )


@st.cache_data(ttl=3600)
def get_trade_log() -> pd.DataFrame:
    """Trade history from IBKR Flex Query, with same-day fills merged per symbol.
    Columns: trade_date, symbol, name, isin, currency, asset_type,
             buy_sell, quantity, entry_exit_price, effective_price, proceeds, commission
    """
    if _S3_BUCKET:
        path = f"s3://{_S3_BUCKET}/history/portfolio/trade_log.json"
    else:
        path = str(_ROOT / "data" / "trade_log.json")
    print(f"[data] loading trade_log from {path}")
    raw = _con().execute(f"SELECT * FROM read_json_auto('{path}')").df()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])

    agg = (
        raw.groupby(
            ["trade_date", "symbol", "name", "isin",
             "currency", "asset_type", "buy_sell"],
            as_index=False,
        )
        .agg(
            quantity=("quantity", "sum"),
            proceeds=("proceeds", "sum"),
            commission=("commission", "sum"),
        )
    )
    agg["entry_exit_price"] = (-agg["proceeds"] / agg["quantity"]).round(4)
    agg["effective_price"]  = (
        (-agg["proceeds"] + agg["commission"].abs()) / agg["quantity"]
    ).round(4)
    return _log(
        "trade_log",
        agg.sort_values("trade_date", ascending=False).reset_index(drop=True),
    )


# ── External market data ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_market_prices(ticker: str) -> pd.Series:
    """Daily close price for an external market ticker (SPY, IWM, TLT, HYG, VIX, etc.).
    Stored under history/market_data/{ticker}.parquet by the nightly Lambda.
    Falls back to yfinance if not yet on S3 (e.g. first run before Lambda has executed).
    Returns a Series indexed by date.
    """
    if _S3_BUCKET:
        path = f"s3://{_S3_BUCKET}/history/market_data/{ticker}.parquet"
    else:
        path = str(_ROOT / "data" / "market_data" / f"{ticker}.parquet")
    try:
        df = _con().execute(
            f"SELECT date, close FROM read_parquet('{path}') ORDER BY date"
        ).df()
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["close"].dropna()
        return _log(f"market_prices/{ticker}", s.to_frame()).index.map(lambda _: s).iloc[0] if False else s
    except Exception:
        # Fallback to yfinance when S3 data not yet available
        try:
            import yfinance as yf
            yf_ticker = f"^{ticker}" if ticker == "VIX" else ticker
            raw = yf.download(yf_ticker, period="5y", progress=False, auto_adjust=True)["Close"]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            raw.index = pd.to_datetime(raw.index)
            print(f"[data] market_prices/{ticker}: yfinance fallback ({len(raw)} rows)")
            return raw.dropna().sort_index()
        except Exception as e:
            print(f"[data] market_prices/{ticker}: both S3 and yfinance failed — {e}")
            return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def get_market_ohlcv(ticker: str) -> pd.DataFrame:
    """Daily OHLCV for an external market ticker. Returns DataFrame with date index."""
    if _S3_BUCKET:
        path = f"s3://{_S3_BUCKET}/history/market_data/{ticker}.parquet"
    else:
        path = str(_ROOT / "data" / "market_data" / f"{ticker}.parquet")
    try:
        df = _con().execute(
            f"SELECT * FROM read_parquet('{path}') ORDER BY date"
        ).df()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")
    except Exception:
        pass
    try:
        import yfinance as yf
        yf_ticker = "^VIX" if ticker == "VIX" else ticker
        raw = yf.download(yf_ticker, period="10y", progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame(columns=["close", "volume"])
        # flatten MultiIndex columns (newer yfinance returns (field, ticker) tuples)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw.columns = [str(c).lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        df = pd.DataFrame({
            "close":  raw["close"].squeeze(),
            "volume": raw["volume"].squeeze() if "volume" in raw.columns else 0.0,
        })
        return df
    except Exception:
        return pd.DataFrame(columns=["close", "volume"])


@st.cache_data(ttl=3600)
def get_fred_series(series_id: str) -> pd.Series:
    """A FRED time series (e.g. BAMLH0A0HYM2 for HY OAS).
    Stored under history/market_data/FRED_{series_id}.parquet by the nightly Lambda.
    Falls back to live FRED CSV endpoint if not yet on S3.
    Returns a Series indexed by date.
    """
    if _S3_BUCKET:
        path = f"s3://{_S3_BUCKET}/history/market_data/FRED_{series_id}.parquet"
    else:
        path = str(_ROOT / "data" / "market_data" / f"FRED_{series_id}.parquet")
    try:
        df = _con().execute(
            f"SELECT date, value FROM read_parquet('{path}') ORDER BY date"
        ).df()
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["value"].dropna()
        print(f"[data] fred/{series_id}: {len(s)} rows from S3")
        return s
    except Exception:
        # Fallback to live FRED CSV
        try:
            import urllib.request as _url, io as _io, numpy as _np
            u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            with _url.urlopen(u, timeout=30) as r:
                raw = pd.read_csv(_io.BytesIO(r.read()))
            raw.columns = ["date", "value"]
            raw["date"]  = pd.to_datetime(raw["date"])
            raw["value"] = pd.to_numeric(raw["value"].replace(".", _np.nan), errors="coerce")
            s = raw.dropna().set_index("date")["value"].sort_index()
            print(f"[data] fred/{series_id}: FRED fallback ({len(s)} rows)")
            return s
        except Exception as e:
            print(f"[data] fred/{series_id}: both S3 and FRED failed — {e}")
            return pd.Series(dtype=float)
