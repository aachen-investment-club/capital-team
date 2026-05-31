"""
One-off backfill: reads nav_history.json from S3 and writes Hive-partitioned
portfolio_and_benchmarks Parquet files to S3 (one file per date).

Uses the AWS CLI for all S3 operations (avoids credential chain issues locally).
Safe to run multiple times — existing partitions are overwritten, nav_history.json is never touched.

Usage:
    python scripts/backfill_nav_to_parquet.py
    python scripts/backfill_nav_to_parquet.py --local   # write to data/raw/ instead of S3
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import pandas as pd

S3_BUCKET  = "aic-fund-public-data"
AWS_REGION = "eu-central-1"
NAV_KEY    = "history/nav_history.json"
RAW_PREFIX = "history/raw"

BENCHMARK_MAP = {
    "spxClose":        "SPX",
    "msciWorldClose":  "MSCI_WORLD",
    "msciEuropeClose": "MSCI_EUROPE",
    "sixtyFortyClose": "60_40",
}

ROOT = pathlib.Path(__file__).resolve().parent.parent


def aws(*args: str) -> str:
    result = subprocess.run(
        ["aws", "--region", AWS_REGION, *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def load_nav_history() -> list[dict]:
    raw = aws("s3", "cp", f"s3://{S3_BUCKET}/{NAV_KEY}", "-")
    return json.loads(raw)


def build_partition_df(records: list[dict], i: int) -> pd.DataFrame:
    entry = records[i]
    rows  = []

    # ── PORTFOLIO ─────────────────────────────────────────────────────────────
    today_nav = entry["fundNav"]
    prev_nav  = records[i - 1]["fundNav"] if i > 0 else today_nav
    port_dr   = (today_nav - prev_nav) / prev_nav if i > 0 else 0.0
    rows.append({
        "ticker":       "PORTFOLIO",
        "index_value":  round(today_nav, 6),
        "daily_return": round(port_dr, 6),
    })

    # ── BENCHMARKS ────────────────────────────────────────────────────────────
    base = records[0]
    for field, ticker in BENCHMARK_MAP.items():
        today_price = entry.get(field)
        if today_price is None:
            continue
        base_price = base.get(field, today_price)
        index_val  = today_price / base_price if base_price else 1.0

        prev_price = records[i - 1].get(field) if i > 0 else None
        bm_dr      = (today_price - prev_price) / prev_price if prev_price else 0.0

        rows.append({
            "ticker":       ticker,
            "index_value":  round(index_val, 6),
            "daily_return": round(bm_dr, 6),
        })

    return pd.DataFrame(rows)


def write_local(df: pd.DataFrame, date: str) -> None:
    path = ROOT / "data" / RAW_PREFIX / "portfolio_and_benchmarks" / f"date={date}" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"  → {path.relative_to(ROOT)}")


def write_s3(df: pd.DataFrame, date: str, tmp_dir: str) -> None:
    local = pathlib.Path(tmp_dir) / f"{date}.parquet"
    df.to_parquet(local, index=False, engine="pyarrow")
    s3_key = f"s3://{S3_BUCKET}/{RAW_PREFIX}/portfolio_and_benchmarks/date={date}/data.parquet"
    aws("s3", "cp", str(local), s3_key)
    print(f"  → {s3_key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Write to data/raw/ instead of S3")
    args = parser.parse_args()

    print("Loading nav_history.json from S3…")
    history = sorted(load_nav_history(), key=lambda h: h["date"])
    print(f"  {len(history)} records: {history[0]['date']} → {history[-1]['date']}\n")

    print("Writing portfolio_and_benchmarks partitions…")
    if args.local:
        for i, entry in enumerate(history):
            df = build_partition_df(history, i)
            write_local(df, entry["date"])
    else:
        with tempfile.TemporaryDirectory() as tmp:
            for i, entry in enumerate(history):
                df = build_partition_df(history, i)
                write_s3(df, entry["date"], tmp)

    print(f"\nBackfill complete — {len(history)} partitions written.")


if __name__ == "__main__":
    main()
