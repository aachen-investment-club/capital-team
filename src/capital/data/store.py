"""
DuckDB store access.

Concurrency contract (DuckDB allows many read-only processes OR one writer):
- Readers (dashboard, loaders) call `query()` which opens a short-lived
  read-only connection per call — open cost is ~ms, never hold a connection.
- The nightly ingest is the SOLE writer; it uses `write_connection()` which
  retries while a straggler reader holds the file.

If lock contention ever becomes a problem in practice, the escalation path is
write-to-`market.duckdb.new` + atomic `os.replace` — not built until needed.
"""
import json
import time
from datetime import datetime, timezone

import duckdb
import pandas as pd

from capital.settings import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS eod_prices (
    security_id TEXT NOT NULL,
    date        DATE NOT NULL,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    adj_close   DOUBLE,
    volume      BIGINT,
    PRIMARY KEY (security_id, date)
);
CREATE TABLE IF NOT EXISTS fundamentals (
    ric                TEXT NOT NULL,
    date               DATE NOT NULL,
    ticker             TEXT,
    gics_sector        TEXT,
    pb_ratio           DOUBLE,
    market_cap         DOUBLE,
    shares_outstanding DOUBLE,
    PRIMARY KEY (ric, date)
);
CREATE TABLE IF NOT EXISTS market_data (
    ticker TEXT NOT NULL,
    date   DATE NOT NULL,
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume BIGINT,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS fred (
    series_id TEXT NOT NULL,
    date      DATE NOT NULL,
    value     DOUBLE,
    PRIMARY KEY (series_id, date)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
-- Derived tables, built nightly by `capital-ingest derived`
CREATE TABLE IF NOT EXISTS daily_returns (
    security_id TEXT NOT NULL,
    date        DATE NOT NULL,
    ret         DOUBLE,
    log_ret     DOUBLE,
    PRIMARY KEY (security_id, date)
);
CREATE TABLE IF NOT EXISTS rolling_stats (
    security_id  TEXT NOT NULL,
    date         DATE NOT NULL,
    vol_20d      DOUBLE,
    vol_60d      DOUBLE,
    mom_12_1     DOUBLE,
    max_dd_1y    DOUBLE,
    adv_20d      DOUBLE,
    PRIMARY KEY (security_id, date)
);
"""


# Additive column migrations, applied by write_connection() on every open.
#
# The fundamentals table grows as new LSEG descriptor fields are wired up (the
# factor model asks the store which columns exist and drops descriptors it
# cannot build — see data.loaders.get_fundamentals_columns). ADD COLUMN IF NOT
# EXISTS makes this idempotent, so an older store file upgrades in place on the
# next ingest and a newer one is untouched.
MIGRATIONS = [
    f"ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS {col} DOUBLE"
    for col in (
        "pe_ratio",          # trailing P/E            -> Earnings Yield
        "ps_ratio",          # price / sales           -> Earnings Yield
        "pcf_ratio",         # price / cash flow       -> Earnings Yield
        "dividend_yield",    # trailing 12m yield      -> Dividend Yield
        "roe",               # return on equity        -> Profitability
        "roa",               # return on assets        -> Profitability
        "gross_margin",      # gross profit / revenue  -> Profitability
        "debt_to_ev",        # total debt / enterprise value -> Leverage
        "debt_to_assets",    # retired: not entitled on our LSEG account
        "revenue_ttm",       # level; growth is derived from the series
        "eps_ttm",           # level; growth is derived from the series
    )
]


def db_exists() -> bool:
    return settings.db_path.exists()


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only query against the store, returning a DataFrame."""
    con = duckdb.connect(str(settings.db_path), read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def write_connection(retry_seconds: int = 600) -> duckdb.DuckDBPyConnection:
    """Open the store read-write (sole-writer ingest only). Ensures schema.

    Retries while another process holds the file, up to `retry_seconds`.
    """
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            con = duckdb.connect(str(settings.db_path))
            break
        except duckdb.IOException:
            if time.monotonic() >= deadline:
                raise
            time.sleep(5)
    con.execute(SCHEMA)
    for statement in MIGRATIONS:
        con.execute(statement)
    return con


def data_version() -> str:
    """Ingestion timestamp used as the cache key for every loader."""
    if not db_exists():
        return ""
    try:
        df = query("SELECT value FROM meta WHERE key = 'data_version'")
        return df["value"].iloc[0] if len(df) else ""
    except duckdb.Error:
        return ""


def get_meta(key: str) -> str:
    """Read one meta value (read-only). Empty string when absent."""
    if not db_exists():
        return ""
    try:
        df = query("SELECT value FROM meta WHERE key = ?", [key])
        return df["value"].iloc[0] if len(df) else ""
    except duckdb.Error:
        return ""


def set_meta(con: duckdb.DuckDBPyConnection, key: str, value: str) -> None:
    """Write one meta value (writer connection only)."""
    con.execute(
        "INSERT INTO meta VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [key, value],
    )


def bump_data_version(con: duckdb.DuckDBPyConnection) -> str:
    """Set meta.data_version to now (UTC ISO). Called at the end of ingest."""
    version = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute(
        "INSERT INTO meta VALUES ('data_version', ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [version],
    )
    return version


def flush(table: str, frames: list[pd.DataFrame], bump: bool = False) -> int:
    """Upsert a batch and *release the writer immediately*.

    DuckDB's write lock is exclusive at the file level: while it is held, no
    other process can open the database even read-only, so the dashboard goes
    dark. A long backfill must therefore write in short bursts rather than hold
    one connection for its whole run — see ingest.eod.run_eod.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    con = write_connection()
    try:
        n = upsert(con, table, df)
        if bump:
            bump_data_version(con)
        return n
    finally:
        con.close()


def coverage(table: str, key: str) -> pd.DataFrame:
    """Earliest and latest date held per key — the basis for `--resume`."""
    if not db_exists():
        return pd.DataFrame(columns=[key, "lo", "hi", "n"])
    try:
        return query(f"SELECT {key}, min(date) AS lo, max(date) AS hi, count(*) AS n "
                     f"FROM {table} GROUP BY {key}")
    except duckdb.Error:
        return pd.DataFrame(columns=[key, "lo", "hi", "n"])


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Idempotent insert-or-replace of a DataFrame into `table` (PK-deduped)."""
    if df.empty:
        return 0
    cols = ", ".join(df.columns)
    con.register("_upsert_df", df)
    con.execute(
        f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _upsert_df"
    )
    con.unregister("_upsert_df")
    return len(df)


def table_stats() -> pd.DataFrame:
    """Row count and date range per table — used by backfill/ingest logging."""
    rows = []
    for t, datecol in [("eod_prices", "date"), ("fundamentals", "date"),
                       ("market_data", "date"), ("fred", "date"),
                       ("daily_returns", "date"), ("rolling_stats", "date")]:
        try:
            df = query(f"SELECT count(*) AS n, min({datecol}) AS lo, max({datecol}) AS hi FROM {t}")
            rows.append({"table": t, **df.iloc[0].to_dict()})
        except duckdb.Error:
            rows.append({"table": t, "n": 0, "lo": None, "hi": None})
    return pd.DataFrame(rows)
