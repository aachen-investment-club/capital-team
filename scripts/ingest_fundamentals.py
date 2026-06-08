#!/usr/bin/env python3
"""
Fundamental data ingestion for Barra factor model.

Fetches daily time-series of:
  TR.PriceToBVPerShare   — P/B ratio  (value factor)
  TR.CompanyMarketCap    — market cap  (WLS weights + liquidity)
  TR.SharesOutstanding   — shares out  (liquidity factor)
  TR.GICSSector          — sector      (industry dummies, fetched once as snapshot)

Universe: config/barra_universe.csv  UNION  config/security_master.csv (non-INDEX rows).
  Deduplication is by RIC, so portfolio securities appear in both files without issue.

Storage:
  local:  data/fundamentals/data.parquet
  S3:     s3://{S3_BUCKET}/history/fundamentals/data.parquet

Schema (Parquet):
  date             date32
  ric              string
  ticker           string
  gics_sector      string    (nullable — ETFs often have no GICS sector)
  pb_ratio         float64
  market_cap       float64   (native currency, as returned by LSEG)
  shares_outstanding float64
  volume           float64   (daily volume from price history, already in EOD parquet
                              but replicated here for convenience)

Usage:
  python scripts/ingest_fundamentals.py                    # backfill 1 year + today
  python scripts/ingest_fundamentals.py --start 2023-01-01 # custom window
  python scripts/ingest_fundamentals.py --dry-run
"""
import argparse
import io
import logging
import os
import pathlib
import sys
from datetime import date, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

_ROOT       = pathlib.Path(__file__).resolve().parent.parent
_CONFIG_DIR = _ROOT / "config"
load_dotenv(_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ingest_fundamentals")

_SCHEMA = pa.schema([
    pa.field("date",               pa.date32()),
    pa.field("ric",                pa.string()),
    pa.field("ticker",             pa.string()),
    pa.field("gics_sector",        pa.string()),
    pa.field("pb_ratio",           pa.float64()),
    pa.field("market_cap",         pa.float64()),
    pa.field("shares_outstanding", pa.float64()),
])

_HIST_FIELDS = ["TR.PriceToBVPerShare", "TR.CompanyMarketCap", "TR.SharesOutstanding"]
_COL_MAP     = {
    "Price To Book Value Per Share": "pb_ratio",
    "Company Market Cap":            "market_cap",
    "Shares Outstanding":            "shares_outstanding",
    # LSEG sometimes returns abbreviated column names
    "PriceToBVPerShare":             "pb_ratio",
    "CompanyMarketCap":              "market_cap",
    "SharesOutstanding":             "shares_outstanding",
}


# ── Universe loading ──────────────────────────────────────────────────────────

def _load_universe() -> pd.DataFrame:
    """Merge barra_universe.csv and security_master.csv, deduplicate by RIC."""
    barra_path  = _CONFIG_DIR / "barra_universe.csv"
    master_path = _CONFIG_DIR / "security_master.csv"

    parts = []
    if barra_path.exists():
        bu = pd.read_csv(barra_path, comment="#", dtype=str)
        bu = bu[["ric", "ticker", "name"]].copy()
        parts.append(bu)

    if master_path.exists():
        sm = pd.read_csv(master_path, dtype=str)
        sm["active"] = sm["active"].str.lower().isin(("true", "1", "yes"))
        # Exclude benchmark indices — no fundamentals available for them
        sm = sm[sm["active"] & (sm["asset_type"] != "INDEX")]
        sm = sm[["ric", "ticker", "name"]].copy()
        parts.append(sm)

    if not parts:
        log.error("No universe files found in config/")
        sys.exit(1)

    universe = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ric"])
    log.info("Universe: %d unique RICs", len(universe))
    return universe


# ── LSEG helpers ──────────────────────────────────────────────────────────────

def _open_session() -> None:
    import lseg.data as ld
    ld.open_session()
    log.info("LSEG session opened")


def _close_session() -> None:
    import lseg.data as ld
    try:
        ld.close_session()
    except Exception:
        pass


def _fetch_gics(rics: list[str]) -> dict[str, str]:
    """Fetch GICS sector snapshot for all RICs. Returns {ric: sector}."""
    import lseg.data as ld
    try:
        df = ld.get_data(universe=rics, fields=["TR.GICSSector"])
        if df is None or df.empty:
            return {}
        # Normalise column name
        sector_col = next(
            (c for c in df.columns if "gics" in c.lower() or "sector" in c.lower()), None
        )
        if sector_col is None:
            return {}
        inst_col = next((c for c in df.columns if "instrument" in c.lower()), df.columns[0])
        return dict(zip(df[inst_col].str.upper(), df[sector_col].fillna("")))
    except Exception as exc:
        log.warning("GICS fetch failed: %s", exc)
        return {}


def _fetch_history_batch(
    rics: list[str], start: str, end: str, batch_size: int = 10
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily P/B, market cap, shares outstanding for `rics` over [start, end].
    Returns {ric: DataFrame(date, pb_ratio, market_cap, shares_outstanding)}.
    Batches requests to stay within LSEG rate limits.
    """
    import lseg.data as ld
    result: dict[str, pd.DataFrame] = {}

    for i in range(0, len(rics), batch_size):
        batch = rics[i : i + batch_size]
        log.info("  batch %d/%d  (%d RICs)", i // batch_size + 1,
                 (len(rics) + batch_size - 1) // batch_size, len(batch))
        try:
            raw = ld.get_history(
                universe=batch,
                fields=_HIST_FIELDS,
                start=start,
                end=end,
                interval="1D",
            )
            if raw is None or raw.empty:
                continue

            # lseg-data returns MultiIndex columns (Instrument, Field)
            if isinstance(raw.columns, pd.MultiIndex):
                for ric in raw.columns.get_level_values(0).unique():
                    sub = raw[ric].copy()
                    sub = _normalise_cols(sub)
                    sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
                    if "date" not in sub.columns:
                        sub.insert(0, "date", sub.index)
                    sub["date"] = pd.to_datetime(sub["date"]).dt.date
                    result[str(ric)] = sub
            else:
                # Single RIC or flat structure
                sub = raw.copy()
                sub = _normalise_cols(sub)
                sub = sub.reset_index().rename(columns={"Date": "date"})
                sub["date"] = pd.to_datetime(sub["date"]).dt.date
                ric = batch[0]
                result[str(ric)] = sub

        except Exception as exc:
            log.warning("  batch failed: %s", exc)

    return result


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename LSEG column names to our canonical names."""
    rename = {}
    for col in df.columns:
        for lseg_name, our_name in _COL_MAP.items():
            if lseg_name.lower().replace(" ", "") in col.lower().replace(" ", ""):
                rename[col] = our_name
                break
    return df.rename(columns=rename)


# ── Storage ───────────────────────────────────────────────────────────────────

def _local_path() -> pathlib.Path:
    return _ROOT / "data" / "fundamentals" / "data.parquet"


def _s3_key() -> str:
    return "history/fundamentals/data.parquet"


def _read_existing(s3_bucket: str, s3_client) -> pd.DataFrame | None:
    if s3_bucket:
        from botocore.exceptions import ClientError
        try:
            obj = s3_client.get_object(Bucket=s3_bucket, Key=_s3_key())
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
    else:
        p = _local_path()
        return pd.read_parquet(p) if p.exists() else None


def _write(df: pd.DataFrame, s3_bucket: str, s3_client) -> None:
    # Ensure schema compatibility
    for col in ("pb_ratio", "market_cap", "shares_outstanding"):
        if col not in df.columns:
            df[col] = float("nan")
    for col in ("gics_sector", "ric", "ticker"):
        if col not in df.columns:
            df[col] = ""

    df["date"] = pd.to_datetime(df["date"]).dt.date
    table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)

    if s3_bucket:
        buf = io.BytesIO()
        pq.write_table(table, buf, compression="snappy")
        s3_client.put_object(Bucket=s3_bucket, Key=_s3_key(), Body=buf.getvalue())
        log.info("Written → s3://%s/%s  (%d rows)", s3_bucket, _s3_key(), len(df))
    else:
        p = _local_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, p, compression="snappy")
        log.info("Written → %s  (%d rows)", p, len(df))


def _to_float(val) -> float:
    """Convert a value to float, returning NaN for None, NaN, or empty string."""
    if val is None:
        return float("nan")
    try:
        v = float(val)
        return v
    except (ValueError, TypeError):
        return float("nan")


# ── Main run ──────────────────────────────────────────────────────────────────

def run(start_date: str, end_date: str, dry_run: bool) -> None:
    universe = _load_universe()
    rics     = universe["ric"].tolist()
    ric_to_ticker = dict(zip(universe["ric"], universe["ticker"]))

    s3_bucket = os.getenv("S3_BUCKET", "")
    s3_client = None
    if s3_bucket:
        import boto3
        s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-central-1"))

    log.info("Run: %d RICs | %s → %s | s3=%s | dry_run=%s",
             len(rics), start_date, end_date, s3_bucket or "(local)", dry_run)

    if dry_run:
        log.info("DRY RUN — no data fetched or written")
        return

    _open_session()
    try:
        # 1. GICS sector snapshot (static — re-fetch each run to catch any changes)
        log.info("Fetching GICS sectors…")
        gics_map = _fetch_gics(rics)
        log.info("  got sectors for %d / %d RICs", len(gics_map), len(rics))

        # 2. Time-series fundamentals
        log.info("Fetching historical fundamentals %s → %s…", start_date, end_date)
        hist_data = _fetch_history_batch(rics, start_date, end_date)
        log.info("  received data for %d / %d RICs", len(hist_data), len(rics))
    finally:
        _close_session()

    if not hist_data:
        log.warning("No data returned — nothing to write")
        return

    # 3. Assemble rows
    rows = []
    for ric, df in hist_data.items():
        ticker = ric_to_ticker.get(ric, ric)
        sector = gics_map.get(ric.upper(), "")
        for _, row in df.iterrows():
            rows.append({
                "date":               row.get("date"),
                "ric":                ric,
                "ticker":             ticker,
                "gics_sector":        sector,
                "pb_ratio":           _to_float(row.get("pb_ratio")),
                "market_cap":         _to_float(row.get("market_cap")),
                "shares_outstanding": _to_float(row.get("shares_outstanding")),
            })

    new_df = pd.DataFrame(rows).dropna(subset=["date"])
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date
    log.info("Assembled %d new rows across %d securities", len(new_df), new_df["ric"].nunique())

    # 4. Merge with existing and deduplicate
    existing = _read_existing(s3_bucket, s3_client)
    if existing is not None and not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ric"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["ric", "date"]).reset_index(drop=True)
    _write(combined, s3_bucket, s3_client)
    log.info("Done — %d total rows, %d unique securities", len(combined), combined["ric"].nunique())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start",   default=(date.today() - timedelta(days=365)).isoformat(),
                   metavar="YYYY-MM-DD", help="start date (default: 1 year ago)")
    p.add_argument("--end",     default=date.today().isoformat(),
                   metavar="YYYY-MM-DD", help="end date (default: today)")
    p.add_argument("--dry-run", action="store_true", help="show plan only")
    args = p.parse_args()
    run(args.start, args.end, args.dry_run)
