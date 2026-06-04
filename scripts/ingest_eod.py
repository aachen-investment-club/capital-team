#!/usr/bin/env python3
"""
EOD price backfill and incremental update via LSEG Data Library.

Storage: hive-partitioned Parquet
  local: {RAW_PREFIX}/eod_prices/security_id={id}/data.parquet
  S3:    s3://{S3_BUCKET}/{RAW_PREFIX}/eod_prices/security_id={id}/data.parquet

Universe: config/security_master.csv — add a row to extend; no code changes needed.

Modes:
  Full backfill (first run):    fetches from --start to today for each active security
  Incremental (subsequent runs): fetches only dates newer than existing Parquet per security
  Retry failures:               re-runs only securities that failed in the previous run

LSEG session:
  Reads credentials from lseg-data.config.json (searched in current dir, then ~/lseg-data.config.json).
  See https://developers.lseg.com/en/api-catalog/lseg-data-platform/lseg-data-library-for-python
  for config file format. LSEG_APP_KEY and LSEG_SESSION_TYPE must also match what is in that file.

RIC validation:
  The ric column in security_master.csv must be verified against LSEG Workspace before running.
  An invalid RIC returns empty data (no exception), which is logged as EMPTY and skipped.

Usage:
  python scripts/ingest_eod.py                        # full backfill / incremental
  python scripts/ingest_eod.py --dry-run              # show plan, no fetches or writes
  python scripts/ingest_eod.py --retry-failures       # retry securities from eod_failures.json
  python scripts/ingest_eod.py --start 2015-01-01     # override backfill start date
  python scripts/ingest_eod.py --end 2024-12-31       # override end date (default: today)
  python scripts/ingest_eod.py --delay 1.0 --max-rps 1  # tighter rate limiting
"""
import argparse
import functools
import io
import json
import logging
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
pd.set_option("future.no_silent_downcasting", True)  # silence lseg-data FutureWarning
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONFIG_DIR = _ROOT / "config"
load_dotenv(_ROOT / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingest_eod")

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    start_date: str = "2010-01-01"        # full-backfill horizon
    end_date: str = ""                     # empty = today
    delay_between_requests: float = 0.5   # flat seconds added after every call
    max_rps: float = 2.0                  # max requests per second (before flat delay)
    max_retries: int = 5                  # per-security retry cap
    retry_base_delay: float = 2.0         # seconds; doubled each attempt
    retry_max_delay: float = 120.0        # cap on computed backoff
    retry_jitter: float = 0.3            # random fraction of delay added as jitter
    failures_file: str = str(_ROOT / "eod_failures.json")
    dry_run: bool = False
    retry_failures_only: bool = False

    def effective_end_date(self) -> str:
        return self.end_date or date.today().isoformat()

# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket style: enforces min_interval between calls + a flat delay."""

    def __init__(self, max_rps: float, min_delay: float) -> None:
        self._min_interval = 1.0 / max_rps if max_rps > 0 else 0.0
        self._min_delay = min_delay
        self._last: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        sleep = max(0.0, self._min_interval - elapsed) + self._min_delay
        if sleep > 0:
            time.sleep(sleep)
        self._last = time.monotonic()

# ── Retry decorator ───────────────────────────────────────────────────────────

def _with_retry(cfg: Config):
    """Decorator: exponential backoff + jitter on any exception."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(cfg.max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == cfg.max_retries:
                        break
                    delay = min(cfg.retry_base_delay * (2 ** attempt), cfg.retry_max_delay)
                    delay += random.uniform(0.0, cfg.retry_jitter * delay)
                    log.warning(
                        "Attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1, cfg.max_retries, exc, delay,
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator

# ── LSEG fetch ────────────────────────────────────────────────────────────────

# Two separate field groups: LSEG rejects parameterized field syntax (e.g.
# TR.CLOSEPRICE(Adjust=0R)) as a field name. The adjusted close is fetched in a
# second call using parameters={"Adjust": "0R"}, which applies the total-return
# adjustment via the request-level parameters dict instead.
_OHLCV_FIELDS = ["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.CLOSEPRICE", "TR.VOLUME"]
_OHLCV_COLS   = ["open", "high", "low", "close", "volume"]

# Canonical PyArrow schema used for every EOD partition file.
_SCHEMA = pa.schema([
    pa.field("security_id", pa.string()),
    pa.field("date",        pa.date32()),
    pa.field("open",        pa.float64()),
    pa.field("high",        pa.float64()),
    pa.field("low",         pa.float64()),
    pa.field("close",       pa.float64()),
    pa.field("adj_close",   pa.float64()),
    pa.field("volume",      pa.int64()),
])


def _open_lseg_session() -> None:
    """Open a LSEG Data Library session from lseg-data.config.json."""
    import lseg.data as ld  # noqa: PLC0415 — optional dep, imported lazily
    ld.open_session()
    log.info("LSEG session opened")


def _close_lseg_session() -> None:
    import lseg.data as ld
    try:
        ld.close_session()
        log.info("LSEG session closed")
    except Exception:
        pass


def _fetch_ric(ric: str, security_id: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily EOD OHLCV history for one RIC and return a normalised DataFrame.

    adj_close is set equal to close (unadjusted) because the LSEG desktop session
    does not support the Adjust=0R parameter on the get_history UDF endpoint.
    The column is preserved in the schema so adjusted prices can be backfilled
    later without a schema migration.

    Returns an empty DataFrame (not an error) when the RIC has no data for the
    requested range — e.g. a bad/unresolvable RIC or a holiday-only window.
    Raises on network/auth errors so the retry wrapper can handle them.
    """
    import lseg.data as ld

    raw = ld.get_history(
        universe=[ric],
        fields=_OHLCV_FIELDS,
        start=start,
        end=end,
        interval="1D",
    )

    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["security_id", "date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    df = raw.copy()
    df.columns = _OHLCV_COLS
    df.index.name = "date"
    df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["security_id"] = security_id
    df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")   # placeholder
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    return df[["security_id", "date", "open", "high", "low", "close", "adj_close", "volume"]]

# ── Storage helpers ───────────────────────────────────────────────────────────

def _local_path(raw_prefix: str, security_id: str) -> pathlib.Path:
    return pathlib.Path(raw_prefix) / "eod_prices" / f"security_id={security_id}" / "data.parquet"


def _s3_key(raw_prefix: str, security_id: str) -> str:
    return f"{raw_prefix}/eod_prices/security_id={security_id}/data.parquet"


def _read_existing_local(path: pathlib.Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _read_existing_s3(s3_client, bucket: str, key: str) -> Optional[pd.DataFrame]:
    from botocore.exceptions import ClientError
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    except Exception as exc:
        log.warning("Could not read existing S3 object %s: %s", key, exc)
        return None


def _write_parquet_local(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _write_parquet_s3(df: pd.DataFrame, s3_client, bucket: str, key: str) -> None:
    table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())


def _write_version(s3_client, s3_bucket: str, raw_prefix: str, max_date: str) -> None:
    payload = json.dumps({
        "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_date": max_date,
    }).encode()
    if s3_bucket:
        key = f"{raw_prefix}/eod_prices/_version.json"
        s3_client.put_object(Bucket=s3_bucket, Key=key, Body=payload)
        log.info("Version file written → s3://%s/%s", s3_bucket, key)
    else:
        p = pathlib.Path(raw_prefix) / "eod_prices" / "_version.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        log.info("Version file written → %s", p)

# ── Failures file ─────────────────────────────────────────────────────────────

def _load_failures_file(path: str) -> list[str]:
    p = pathlib.Path(path)
    if not p.exists():
        log.error("Failures file not found: %s", path)
        sys.exit(1)
    data = json.loads(p.read_text())
    return [item["security_id"] for item in data.get("failures", [])]


def _save_failures_file(path: str, failures: list[dict]) -> None:
    pathlib.Path(path).write_text(json.dumps({"failures": failures}, indent=2))

# ── Main run ──────────────────────────────────────────────────────────────────

def run(cfg: Config) -> None:
    # Load universe
    csv_path = _CONFIG_DIR / "security_master.csv"
    if not csv_path.exists():
        log.error("Security master not found: %s", csv_path)
        sys.exit(1)

    universe = pd.read_csv(csv_path, dtype=str)
    universe["active"] = universe["active"].str.lower().isin(("true", "1", "yes"))
    universe = universe[universe["active"]].reset_index(drop=True)

    if universe.empty:
        log.warning("No active securities in security_master.csv")
        return

    if cfg.retry_failures_only:
        failure_ids = _load_failures_file(cfg.failures_file)
        if not failure_ids:
            log.info("No failures to retry")
            return
        universe = universe[universe["security_id"].isin(failure_ids)].reset_index(drop=True)
        log.info("Retrying %d failed security(s)", len(universe))

    # Storage config — mirrors lib/data.py env var resolution
    s3_bucket  = os.getenv("S3_BUCKET", "")
    raw_prefix = os.getenv(
        "RAW_PREFIX",
        "history/raw" if s3_bucket else str(_ROOT / "data" / "raw"),
    )
    s3_client = None
    if s3_bucket:
        import boto3
        s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-central-1"))

    end_date = cfg.effective_end_date()
    log.info(
        "Run: %d securities | start=%s end=%s | s3=%s | dry_run=%s",
        len(universe), cfg.start_date, end_date,
        s3_bucket or "(local)", cfg.dry_run,
    )

    if cfg.dry_run:
        log.info("DRY RUN — no data will be fetched or written")

    # Wrap fetch with retry
    @_with_retry(cfg)
    def _fetch(ric: str, security_id: str, start: str, end: str) -> pd.DataFrame:
        return _fetch_ric(ric, security_id, start, end)

    limiter = RateLimiter(cfg.max_rps, cfg.delay_between_requests)

    succeeded_tickers: list[str] = []
    failed: list[dict] = []
    skipped = 0
    max_dates: list[str] = []

    if not cfg.dry_run:
        _open_lseg_session()

    try:
        for _, row in universe.iterrows():
            sec_id = row["security_id"]
            ric    = row["ric"]
            ticker = row["ticker"]

            # Determine fetch range from existing state
            existing: Optional[pd.DataFrame] = None
            if s3_bucket:
                existing = _read_existing_s3(s3_client, s3_bucket, _s3_key(raw_prefix, sec_id))
            else:
                existing = _read_existing_local(_local_path(raw_prefix, sec_id))

            if existing is not None and not existing.empty:
                max_existing = str(existing["date"].max())
                next_day = (date.fromisoformat(max_existing) + timedelta(days=1)).isoformat()
                if next_day > end_date:
                    log.info("[SKIP]  %-6s (%s)  up to date (max: %s)", ticker, sec_id, max_existing)
                    skipped += 1
                    continue
                fetch_start = next_day
                log.info("[INCR]  %-6s (%s)  %s → %s", ticker, sec_id, fetch_start, end_date)
            else:
                fetch_start = cfg.start_date
                log.info("[FULL]  %-6s (%s)  %s → %s", ticker, sec_id, fetch_start, end_date)

            if cfg.dry_run:
                continue

            try:
                limiter.wait()
                new_df = _fetch(ric, sec_id, fetch_start, end_date)

                if new_df.empty:
                    log.warning("[EMPTY] %-6s (%s)  no data returned — verify RIC", ticker, sec_id)
                    skipped += 1
                    continue

                # Merge with existing and deduplicate
                if existing is not None and not existing.empty:
                    combined = pd.concat([existing, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["security_id", "date"], keep="last")
                else:
                    combined = new_df

                combined = combined.sort_values("date").reset_index(drop=True)

                # Write
                if s3_bucket:
                    key = _s3_key(raw_prefix, sec_id)
                    _write_parquet_s3(combined, s3_client, s3_bucket, key)
                    log.info(
                        "[OK]    %-6s (%s)  %d rows → s3://%s/%s",
                        ticker, sec_id, len(combined), s3_bucket, key,
                    )
                else:
                    local = _local_path(raw_prefix, sec_id)
                    _write_parquet_local(combined, local)
                    log.info(
                        "[OK]    %-6s (%s)  %d rows → %s",
                        ticker, sec_id, len(combined), local,
                    )

                max_dates.append(str(combined["date"].max()))
                succeeded_tickers.append(ticker)

            except Exception as exc:
                log.error("[FAIL]  %-6s (%s)  %s", ticker, sec_id, exc, exc_info=True)
                failed.append({"security_id": sec_id, "ticker": ticker, "error": str(exc)})

    finally:
        if not cfg.dry_run:
            _close_lseg_session()

    # Version file — update only when at least one security was written
    if max_dates and not cfg.dry_run:
        _write_version(s3_client, s3_bucket, raw_prefix, max(max_dates))

    # Failures file
    if failed:
        _save_failures_file(cfg.failures_file, failed)
        log.warning("%d failure(s) written to %s", len(failed), cfg.failures_file)
    elif not cfg.retry_failures_only and pathlib.Path(cfg.failures_file).exists():
        pathlib.Path(cfg.failures_file).unlink()  # clean run — remove stale failures

    # Summary
    n_ok   = len(succeeded_tickers)
    n_fail = len(failed)
    print("\n" + "─" * 64)
    if succeeded_tickers:
        print(f"  SUCCEEDED  {n_ok:>4}   {', '.join(succeeded_tickers)}")
    else:
        print(f"  SUCCEEDED  {n_ok:>4}")
    print(f"  SKIPPED    {skipped:>4}")
    if failed:
        failed_tickers = [f["ticker"] for f in failed]
        print(f"  FAILED     {n_fail:>4}   {', '.join(failed_tickers)}")
        print(f"             see {cfg.failures_file}")
    if cfg.dry_run:
        print(f"  (dry run — nothing fetched or written)")
    print("─" * 64 + "\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> Config:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start",           default="2010-01-01", metavar="YYYY-MM-DD",
                   help="backfill start date (default: 2010-01-01)")
    p.add_argument("--end",             default="",           metavar="YYYY-MM-DD",
                   help="end date (default: today)")
    p.add_argument("--delay",           type=float, default=0.5, metavar="SECS",
                   help="flat seconds to sleep after each request (default: 0.5)")
    p.add_argument("--max-rps",         type=float, default=2.0,
                   help="max requests per second (default: 2.0)")
    p.add_argument("--max-retries",     type=int,   default=5,
                   help="max retry attempts per security (default: 5)")
    p.add_argument("--retry-failures",  action="store_true",
                   help="re-run only securities listed in eod_failures.json")
    p.add_argument("--dry-run",         action="store_true",
                   help="show plan without fetching or writing anything")
    args = p.parse_args()
    return Config(
        start_date=args.start,
        end_date=args.end,
        delay_between_requests=args.delay,
        max_rps=args.max_rps,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        retry_failures_only=args.retry_failures,
    )


if __name__ == "__main__":
    run(_parse_args())
