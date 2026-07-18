"""
THE DATA CONTRACT — all data access goes through here.

Pages and analytics must never import boto3 or duckdb, and must never
reference S3 paths or the DB file directly. Loaders return ready DataFrames
and are memoized on the store's data_version (see cache.py), so the first
call after a nightly ingest pays the read and everyone else hits the cache.

Sources:
- Local DuckDB store (market.duckdb): EOD prices, fundamentals, market data,
  FRED series, derived tables. Written only by `capital-ingest`.
- S3 JSONs under history/portfolio/: the stable outputs of the IBKR ingest,
  shared with the club website.
- DynamoDB (fund-baskets): theme mappings.
"""
import io
import os

import duckdb
import pandas as pd

from capital.data import store
from capital.data.cache import cached_by_version
from capital.settings import settings


def _log(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print(f"[data] {name}: EMPTY")
        return df
    date_col = next((c for c in ("date", "trade_date") if c in df.columns), None)
    if date_col:
        print(f"[data] {name}: {len(df)} rows  {df[date_col].min()} -> {df[date_col].max()}")
    else:
        print(f"[data] {name}: {len(df)} rows")
    return df


# ── S3 JSON access (stable website outputs; not in the DuckDB store) ─────────

def _portfolio_json(table: str) -> str:
    if settings.s3_bucket:
        return f"s3://{settings.s3_bucket}/{settings.derived_prefix}/{table}.json" \
            if table in ("portfolio_and_benchmarks", "daily_weightings") \
            else f"s3://{settings.s3_bucket}/{settings.portfolio_prefix}/{table}.json"
    return str(settings.root / "data" / "derived" / f"{table}.json") \
        if table in ("portfolio_and_benchmarks", "daily_weightings") \
        else str(settings.root / "data" / f"{table}.json")


def _mem_con() -> duckdb.DuckDBPyConnection:
    """Short-lived in-memory DuckDB with S3 access, for reading the S3 JSONs."""
    con = duckdb.connect(":memory:")
    if settings.s3_bucket:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"SET s3_region='{settings.aws_region}';")
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            con.execute(f"SET s3_access_key_id='{os.environ['AWS_ACCESS_KEY_ID']}';")
            con.execute(f"SET s3_secret_access_key='{os.environ['AWS_SECRET_ACCESS_KEY']}';")
            if os.getenv("AWS_SESSION_TOKEN"):
                con.execute(f"SET s3_session_token='{os.environ['AWS_SESSION_TOKEN']}';")
        else:
            import boto3
            creds = boto3.Session().get_credentials()
            if creds:
                creds = creds.get_frozen_credentials()
                con.execute(f"SET s3_access_key_id='{creds.access_key}';")
                con.execute(f"SET s3_secret_access_key='{creds.secret_key}';")
                if creds.token:
                    con.execute(f"SET s3_session_token='{creds.token}';")
    return con


# ── Portfolio loaders (S3 website outputs) ────────────────────────────────────

@cached_by_version
def get_portfolio_and_benchmarks() -> pd.DataFrame:
    """Portfolio + benchmark daily index values (normalised to 1.0 at inception).
    Columns: date, ticker, index_value, daily_return
    ticker = 'PORTFOLIO' | 'SPX' | 'MSCI_WORLD' | 'MSCI_EUROPE' | '60_40'
    """
    con = _mem_con()
    try:
        df = con.execute(
            "SELECT date, ticker, index_value, daily_return"
            f" FROM read_json_auto('{_portfolio_json('portfolio_and_benchmarks')}')"
            " ORDER BY ticker, date"
        ).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return _log("portfolio_and_benchmarks", df)


@cached_by_version
def get_daily_weightings_history() -> pd.DataFrame:
    """Daily position weights + returns for all holdings including cash.
    Columns: date, symbol, name, isin, ccy, category, pct_nav,
             cumulative_return, daily_return
    """
    con = _mem_con()
    try:
        df = con.execute(
            "SELECT date, symbol, name, isin, ccy, category, pct_nav,"
            " cumulative_return, daily_return"
            f" FROM read_json_auto('{_portfolio_json('daily_weightings')}')"
            " ORDER BY symbol, date"
        ).df()
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return _log("daily_weightings", df)


@cached_by_version
def get_trade_log() -> pd.DataFrame:
    """Trade history from IBKR Flex Query, with same-day fills merged per symbol.
    Columns: trade_date, symbol, name, isin, currency, asset_type,
             buy_sell, quantity, entry_exit_price, effective_price, proceeds, commission
    """
    con = _mem_con()
    try:
        raw = con.execute(
            f"SELECT * FROM read_json_auto('{_portfolio_json('trade_log')}')"
        ).df()
    finally:
        con.close()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    agg = (
        raw.groupby(
            ["trade_date", "symbol", "name", "isin", "currency", "asset_type", "buy_sell"],
            as_index=False,
        )
        .agg(quantity=("quantity", "sum"), proceeds=("proceeds", "sum"),
             commission=("commission", "sum"))
    )
    agg["entry_exit_price"] = (-agg["proceeds"] / agg["quantity"]).round(4)
    agg["effective_price"] = ((-agg["proceeds"] + agg["commission"].abs()) / agg["quantity"]).round(4)
    return _log("trade_log", agg.sort_values("trade_date", ascending=False).reset_index(drop=True))


def get_theme_mappings() -> pd.DataFrame:
    """Theme/basket assignment per symbol. Columns: symbol, theme"""
    if not settings.ddb_baskets_table:
        return pd.DataFrame(columns=["symbol", "theme"])
    import boto3
    from boto3.dynamodb.conditions import Attr
    tbl = boto3.resource("dynamodb", region_name=settings.aws_region).Table(settings.ddb_baskets_table)
    items: list[dict] = []
    resp = tbl.scan(FilterExpression=Attr("active").eq(True), ProjectionExpression="symbol, theme")
    items.extend(resp["Items"])
    while "LastEvaluatedKey" in resp:
        resp = tbl.scan(FilterExpression=Attr("active").eq(True),
                        ProjectionExpression="symbol, theme",
                        ExclusiveStartKey=resp["LastEvaluatedKey"])
        items.extend(resp["Items"])
    return pd.DataFrame(items)[["symbol", "theme"]] if items else pd.DataFrame(columns=["symbol", "theme"])


# ── Security master ───────────────────────────────────────────────────────────

@cached_by_version
def get_security_master() -> pd.DataFrame:
    """Active securities from security_master.csv (S3 when deployed, local otherwise).
    Columns: security_id, ric, ticker, isin, name, currency, asset_type
    Sorted by ticker.
    """
    if settings.s3_bucket:
        try:
            import boto3
            s3c = boto3.client("s3", region_name=settings.aws_region)
            obj = s3c.get_object(Bucket=settings.s3_bucket, Key="config/security_master.csv")
            df = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str)
        except Exception as e:
            print(f"[data] S3 security_master failed ({e}), falling back to local")
            df = pd.read_csv(settings.config_dir / "security_master.csv", dtype=str)
    else:
        csv_path = settings.config_dir / "security_master.csv"
        if not csv_path.exists():
            return pd.DataFrame(columns=["security_id", "ric", "ticker", "isin",
                                         "name", "currency", "asset_type"])
        df = pd.read_csv(csv_path, dtype=str)

    df["active"] = df["active"].str.lower().isin(("true", "1", "yes"))
    df = df[df["active"]].drop(columns=["active"]).sort_values("ticker").reset_index(drop=True)
    return _log("security_master", df)


# ── Store loaders (DuckDB file) ───────────────────────────────────────────────

@cached_by_version
def get_eod_prices(security_id: str) -> pd.DataFrame:
    """Daily EOD OHLCV for one security, sorted by date ascending.
    Columns: security_id, date, open, high, low, close, adj_close, volume
    Returns an empty DataFrame (no error) if no data has been ingested yet.
    """
    cols = ["security_id", "date", "open", "high", "low", "close", "adj_close", "volume"]
    if not store.db_exists():
        return pd.DataFrame(columns=cols)
    df = store.query(
        "SELECT security_id, date, open, high, low, close, adj_close, volume"
        " FROM eod_prices WHERE security_id = ? ORDER BY date",
        [security_id],
    )
    df["date"] = pd.to_datetime(df["date"])
    return _log(f"eod_prices/{security_id}", df)


@cached_by_version
def get_fundamentals() -> pd.DataFrame:
    """Daily fundamentals snapshot for the Barra universe.
    Columns: date, ric, ticker, gics_sector, pb_ratio, market_cap, shares_outstanding
    """
    cols = ["date", "ric", "ticker", "gics_sector", "pb_ratio", "market_cap", "shares_outstanding"]
    if not store.db_exists():
        return pd.DataFrame(columns=cols)
    df = store.query(
        "SELECT date, ric, ticker, gics_sector, pb_ratio, market_cap, shares_outstanding"
        " FROM fundamentals ORDER BY ric, date"
    )
    df["date"] = pd.to_datetime(df["date"])
    return _log("fundamentals", df)


@cached_by_version
def get_market_prices(ticker: str) -> pd.Series:
    """Daily close for an external market ticker (SPY, IWM, TLT, HYG, VIX, …).
    Falls back to yfinance if the store has no rows yet. Series indexed by date.
    """
    if store.db_exists():
        df = store.query("SELECT date, close FROM market_data WHERE ticker = ? ORDER BY date", [ticker])
        if len(df):
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["close"].dropna()
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
        print(f"[data] market_prices/{ticker}: store and yfinance failed — {e}")
        return pd.Series(dtype=float)


@cached_by_version
def get_market_ohlcv(ticker: str) -> pd.DataFrame:
    """Daily OHLCV for an external market ticker. DataFrame with date index."""
    if store.db_exists():
        df = store.query("SELECT * FROM market_data WHERE ticker = ? ORDER BY date", [ticker])
        if len(df):
            df["date"] = pd.to_datetime(df["date"])
            return df.drop(columns=["ticker"]).set_index("date")
    try:
        import yfinance as yf
        yf_ticker = "^VIX" if ticker == "VIX" else ticker
        raw = yf.download(yf_ticker, period="10y", progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame(columns=["close", "volume"])
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        raw.columns = [str(c).lower() for c in raw.columns]
        raw.index = pd.to_datetime(raw.index)
        return pd.DataFrame({
            "close": raw["close"].squeeze(),
            "volume": raw["volume"].squeeze() if "volume" in raw.columns else 0.0,
        })
    except Exception:
        return pd.DataFrame(columns=["close", "volume"])


@cached_by_version
def get_fred_series(series_id: str) -> pd.Series:
    """A FRED time series (e.g. BAMLH0A0HYM2 for HY OAS). Series indexed by date.
    Falls back to the live FRED CSV endpoint if the store has no rows yet.
    """
    if store.db_exists():
        df = store.query("SELECT date, value FROM fred WHERE series_id = ? ORDER BY date", [series_id])
        if len(df):
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["value"].dropna()
    try:
        import urllib.request as _url
        import numpy as np
        u = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with _url.urlopen(u, timeout=30) as r:
            raw = pd.read_csv(io.BytesIO(r.read()))
        raw.columns = ["date", "value"]
        raw["date"] = pd.to_datetime(raw["date"])
        raw["value"] = pd.to_numeric(raw["value"].replace(".", np.nan), errors="coerce")
        s = raw.dropna().set_index("date")["value"].sort_index()
        print(f"[data] fred/{series_id}: FRED fallback ({len(s)} rows)")
        return s
    except Exception as e:
        print(f"[data] fred/{series_id}: store and FRED failed — {e}")
        return pd.Series(dtype=float)


# ── Bulk / cross-sectional loaders (the 1k+ universe hot path) ────────────────

@cached_by_version
def get_close_matrix(security_ids: tuple[str, ...] | None = None,
                     start: str | None = None) -> pd.DataFrame:
    """Wide matrix of adj_close (columns = security_id, index = date).
    One store query regardless of universe size — never loop get_eod_prices.
    """
    if not store.db_exists():
        return pd.DataFrame()
    sql = "SELECT security_id, date, adj_close FROM eod_prices"
    conds, params = [], []
    if security_ids:
        conds.append(f"security_id IN ({', '.join('?' * len(security_ids))})")
        params.extend(security_ids)
    if start:
        conds.append("date >= ?")
        params.append(start)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    df = store.query(sql, params)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="security_id", values="adj_close").sort_index()


@cached_by_version
def get_returns_matrix(security_ids: tuple[str, ...] | None = None,
                       start: str | None = None) -> pd.DataFrame:
    """Wide matrix of simple daily returns from the derived daily_returns table
    (falls back to pct_change on the close matrix if derived isn't built yet).
    """
    if store.db_exists():
        sql = "SELECT security_id, date, ret FROM daily_returns"
        conds, params = [], []
        if security_ids:
            conds.append(f"security_id IN ({', '.join('?' * len(security_ids))})")
            params.extend(security_ids)
        if start:
            conds.append("date >= ?")
            params.append(start)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        df = store.query(sql, params)
        if len(df):
            df["date"] = pd.to_datetime(df["date"])
            return df.pivot(index="date", columns="security_id", values="ret").sort_index()
    closes = get_close_matrix(security_ids, start)
    return closes.pct_change() if len(closes) else pd.DataFrame()


@cached_by_version
def get_universe_snapshot(as_of: str | None = None) -> pd.DataFrame:
    """One row per security as of a date (default: latest): last close plus the
    derived rolling stats. Feeds screens without touching per-security series.
    Columns: security_id, date, close, adj_close, vol_20d, vol_60d, mom_12_1,
             max_dd_1y, adv_20d
    """
    if not store.db_exists():
        return pd.DataFrame()
    cond = "WHERE date <= ?" if as_of else ""
    params = [as_of, as_of] if as_of else []  # cond appears twice below
    df = store.query(f"""
        WITH last_px AS (
            SELECT security_id, arg_max(close, date) AS close,
                   arg_max(adj_close, date) AS adj_close, max(date) AS date
            FROM eod_prices {cond} GROUP BY security_id
        ),
        last_stats AS (
            SELECT security_id, arg_max(vol_20d, date) AS vol_20d,
                   arg_max(vol_60d, date) AS vol_60d, arg_max(mom_12_1, date) AS mom_12_1,
                   arg_max(max_dd_1y, date) AS max_dd_1y, arg_max(adv_20d, date) AS adv_20d
            FROM rolling_stats {cond} GROUP BY security_id
        )
        SELECT p.security_id, p.date, p.close, p.adj_close,
               s.vol_20d, s.vol_60d, s.mom_12_1, s.max_dd_1y, s.adv_20d
        FROM last_px p LEFT JOIN last_stats s USING (security_id)
        ORDER BY p.security_id
    """, params)
    df["date"] = pd.to_datetime(df["date"])
    return _log("universe_snapshot", df)


def data_version() -> str:
    """Expose the store version for pages that key their own caches on it."""
    from capital.data.cache import data_version as _v
    return _v()
