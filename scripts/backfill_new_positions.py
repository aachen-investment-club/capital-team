"""
Backfill historical EOD data for new positions COPAl (SEC_012) and PURR (SEC_013).
Fetches from yfinance, formats to match Lambda EOD schema, uploads to S3.

Run from repo root:
    python scripts/backfill_new_positions.py
"""
import io
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET  = os.environ["S3_BUCKET"]
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")
EOD_PREFIX = "history/eod_prices"

_EOD_SCHEMA = pa.schema([
    pa.field("security_id", pa.string()),
    pa.field("date",        pa.date32()),
    pa.field("open",        pa.float64()),
    pa.field("high",        pa.float64()),
    pa.field("low",         pa.float64()),
    pa.field("close",       pa.float64()),
    pa.field("adj_close",   pa.float64()),
    pa.field("volume",      pa.int64()),
])

SECURITIES = [
    # (security_id, yfinance_ticker)
    ("SEC_012", "COPA.L"),   # WisdomTree Copper ETP — London
    ("SEC_013", "PURR"),     # Hyperliquid Strategies — Nasdaq
]

s3 = boto3.client("s3", region_name=AWS_REGION)


def fetch_and_upload(security_id: str, yf_ticker: str) -> None:
    print(f"\n── {security_id} ({yf_ticker}) ──")

    raw = yf.download(yf_ticker, period="max", progress=False, auto_adjust=False)
    if raw.empty:
        print(f"  ERROR: yfinance returned no data for {yf_ticker}")
        return

    # Flatten MultiIndex columns (newer yfinance)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw.columns = [str(c).lower() for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index)

    df = pd.DataFrame({
        "security_id": security_id,
        "date":        raw.index.date,
        "open":        pd.to_numeric(raw.get("open"),     errors="coerce"),
        "high":        pd.to_numeric(raw.get("high"),     errors="coerce"),
        "low":         pd.to_numeric(raw.get("low"),      errors="coerce"),
        "close":       pd.to_numeric(raw.get("close"),    errors="coerce"),
        "adj_close":   pd.to_numeric(raw.get("adj close", raw.get("close")), errors="coerce"),
        "volume":      pd.to_numeric(raw.get("volume"),   errors="coerce").fillna(0).astype("int64"),
    })
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    print(f"  {len(df)} rows  {df['date'].min()} → {df['date'].max()}")

    # Write parquet
    buf = io.BytesIO()
    pq.write_table(
        pa.Table.from_pandas(df, schema=_EOD_SCHEMA, preserve_index=False),
        buf, compression="snappy",
    )

    key = f"{EOD_PREFIX}/security_id={security_id}/data.parquet"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Uploaded → s3://{S3_BUCKET}/{key}")


if __name__ == "__main__":
    for sid, ticker in SECURITIES:
        try:
            fetch_and_upload(sid, ticker)
        except Exception as e:
            print(f"  FAILED {sid}: {e}")
    print("\nDone.")
