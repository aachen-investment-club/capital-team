"""
Nightly EOD price ingestion Lambda.

Fetches the last few trading days for every active security in the master
with a SINGLE ld.get_history() call, then appends new rows to each
security's Parquet partition on S3.

Trigger: EventBridge Scheduler — daily after market close (e.g. 22:00 CET).

Environment variables (set in Lambda console):
  S3_BUCKET       — target bucket (e.g. aic-fund-public-data)
  AWS_REGION      — default eu-central-1
  LSEG_APP_KEY    — LSEG Data Platform application key
  LSEG_USERNAME   — RDP username (email)
  LSEG_PASSWORD   — RDP password
  LOOKBACK_DAYS   — days to fetch; default 5 (covers weekends + holidays)
"""
import io
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

S3_BUCKET     = os.environ["S3_BUCKET"]
AWS_REGION    = os.environ.get("AWS_REGION", "eu-central-1")
EOD_PREFIX    = "history/eod_prices"
MASTER_KEY    = "config/security_master.csv"
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "5"))

_OHLCV_FIELDS = ["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.CLOSEPRICE", "TR.VOLUME"]
_OHLCV_COLS   = ["open", "high", "low", "close", "volume"]

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

s3 = boto3.client("s3", region_name=AWS_REGION)


# ── LSEG session ──────────────────────────────────────────────────────────────

def _open_lseg_session() -> None:
    """Open a LSEG Data Platform (RDP) session using env-var credentials."""
    import lseg.data as ld
    from lseg.data.session import platform as lseg_platform

    sess = lseg_platform.Definition(
        app_key=os.environ["LSEG_APP_KEY"],
        grant=lseg_platform.GrantPassword(
            username=os.environ["LSEG_USERNAME"],
            password=os.environ["LSEG_PASSWORD"],
        ),
        signon_control=True,   # force-close other open sessions for this account
    ).get_session()
    sess.open()
    ld.session.set_default(sess)
    log.info("LSEG session opened")


def _close_lseg_session() -> None:
    import lseg.data as ld
    try:
        ld.close_session()
    except Exception:
        pass


# ── Data fetch ────────────────────────────────────────────────────────────────

def _fetch_all(rics: list[str], start: str, end: str) -> pd.DataFrame:
    """Single ld.get_history() call for all RICs.

    Returns a DataFrame with MultiIndex (Instruments, Date) rows and
    OHLCV fields as columns. Returns an empty DataFrame on failure.
    """
    import lseg.data as ld
    raw = ld.get_history(
        universe=rics,
        fields=_OHLCV_FIELDS,
        start=start,
        end=end,
        interval="1D",
    )
    return raw if raw is not None else pd.DataFrame()


def _normalise(df: pd.DataFrame, ric: str, ric_to_id: dict[str, str]) -> tuple[str, pd.DataFrame] | None:
    """Convert a single-instrument OHLCV DataFrame to the canonical schema."""
    sec_id = ric_to_id.get(str(ric))
    if not sec_id or df.empty:
        return None
    df = df.copy()
    df.columns = _OHLCV_COLS
    df.index.name = "date"
    df = df.reset_index()
    df["date"]        = pd.to_datetime(df["date"]).dt.date
    df["security_id"] = sec_id
    df["adj_close"]   = pd.to_numeric(df["close"], errors="coerce")
    df["volume"]      = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        return None
    return sec_id, df[["security_id", "date", "open", "high", "low", "close", "adj_close", "volume"]]


def _per_security(raw: pd.DataFrame, ric_to_id: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Reshape the multi-universe response into {security_id: ohlcv_df}."""
    if raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}

    # lseg-data returns Layout B: DatetimeIndex rows, MultiIndex(Instrument, Field) columns.
    if isinstance(raw.columns, pd.MultiIndex):
        for ric in raw.columns.get_level_values(0).unique():
            out = _normalise(raw[ric], ric, ric_to_id)
            if out:
                result[out[0]] = out[1]
    elif isinstance(raw.index, pd.MultiIndex):
        # Layout A fallback: MultiIndex(Instruments, Date) rows
        for ric in raw.index.get_level_values(0).unique():
            out = _normalise(raw.xs(ric, level=0), ric, ric_to_id)
            if out:
                result[out[0]] = out[1]
    else:
        log.warning("Unrecognised response structure — shape=%s cols=%s",
                    raw.shape, raw.columns.tolist())

    return result


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_key(security_id: str) -> str:
    return f"{EOD_PREFIX}/security_id={security_id}/data.parquet"


def _read_existing(security_id: str) -> pd.DataFrame | None:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=_s3_key(security_id))
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _write_parquet(df: pd.DataFrame, security_id: str) -> None:
    table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="snappy")
    s3.put_object(Bucket=S3_BUCKET, Key=_s3_key(security_id), Body=buf.getvalue())


def _write_version(max_date: str) -> None:
    payload = json.dumps({
        "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_date":    max_date,
    })
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{EOD_PREFIX}/_version.json",
        Body=payload,
        ContentType="application/json",
    )


# ── Handler ───────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    # Load active universe from S3
    obj    = s3.get_object(Bucket=S3_BUCKET, Key=MASTER_KEY)
    master = pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=str)
    master["active"] = master["active"].str.lower().isin(("true", "1", "yes"))
    master = master[master["active"]].reset_index(drop=True)

    if master.empty:
        log.warning("Security master is empty — nothing to fetch")
        return {"status": "skipped", "reason": "empty universe"}

    rics       = master["ric"].tolist()
    ric_to_id  = dict(zip(master["ric"], master["security_id"]))
    end        = date.today().isoformat()
    start      = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()

    log.info("Fetching %d RICs | %s → %s", len(rics), start, end)

    _open_lseg_session()
    try:
        raw = pd.DataFrame()
        for attempt in range(1, 4):
            try:
                raw = _fetch_all(rics, start, end)
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                wait = 10 * attempt
                log.warning("Fetch attempt %d/3 failed: %s — retrying in %ds", attempt, exc, wait)
                time.sleep(wait)
    finally:
        _close_lseg_session()

    if raw.empty:
        log.warning("LSEG returned no data")
        return {"status": "empty"}

    new_data = _per_security(raw, ric_to_id)
    log.info("Received data for %d / %d securities", len(new_data), len(rics))

    succeeded, skipped, failed = [], [], []
    max_dates: list[str] = []

    for sec_id, new_df in new_data.items():
        if new_df.empty:
            skipped.append(sec_id)
            continue
        try:
            existing = _read_existing(sec_id)
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["security_id", "date"], keep="last")
            else:
                combined = new_df
            combined = combined.sort_values("date").reset_index(drop=True)
            _write_parquet(combined, sec_id)
            max_date = str(combined["date"].max())
            max_dates.append(max_date)
            succeeded.append(sec_id)
            log.info("  %-8s  %d rows  max=%s", sec_id, len(combined), max_date)
        except Exception as exc:
            log.error("  %-8s  FAILED: %s", sec_id, exc, exc_info=True)
            failed.append(sec_id)

    if max_dates:
        _write_version(max(max_dates))
        log.info("_version.json updated → max_date=%s", max(max_dates))

    summary = {"succeeded": len(succeeded), "skipped": len(skipped), "failed": len(failed)}
    log.info("Done: %s", summary)

    if failed:
        raise RuntimeError(f"Failed securities: {failed}")

    return summary
