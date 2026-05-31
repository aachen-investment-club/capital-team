"""
Backfill historical equity + FX positions to S3 as CSV.

Run ONCE before the new fund-data-ingestion lambda is deployed,
so precompute/build_derived.py has data from fund inception.

Reads from:
  data/ibkr/prior_positions.parquet
  data/ibkr/open_positions.parquet
  data/ibkr/fx_positions.parquet
  data/ibkr/trade_log.parquet

Writes to S3 (CSV, matching the lambda output format):
  history/raw/daily_equity_positions/date=YYYY-MM-DD/data.csv
  history/raw/daily_fx_positions/date=YYYY-MM-DD/data.csv

AWS credentials must be available (env vars, SSO, or IAM role).

Usage:
    S3_BUCKET=aic-fund-public-data python scripts/backfill_positions_to_s3.py
"""
import os
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import boto3
import pandas as pd

from lib.ibkr import (
    load_prior_positions, load_trade_log, load_open_positions, load_fx_positions,
    build_daily_positions, build_cash_positions, _shares_from_trades,
)

S3_BUCKET  = os.environ.get("S3_BUCKET",  "aic-fund-public-data")
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")
RAW_PREFIX = "history/raw"

s3 = boto3.client("s3", region_name=AWS_REGION)


def _write_csv(df: pd.DataFrame, key: str) -> None:
    s3.put_object(Bucket=S3_BUCKET, Key=key,
                  Body=df.to_csv(index=False).encode(),
                  ContentType="text/csv")
    print(f"  wrote s3://{S3_BUCKET}/{key}  ({len(df)} rows)")


def _key_exists(key: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False


def backfill_equity_positions() -> None:
    print("[daily_equity_positions]")
    prior   = load_prior_positions()
    trades  = load_trade_log()
    opens   = load_open_positions()
    shares  = _shares_from_trades(trades)

    # Combine prior + open positions, resolved with shares
    eq = build_daily_positions(prior, trades, opens)

    for dt, group in eq.groupby("date"):
        date_str = dt.strftime("%Y-%m-%d")
        key = f"{RAW_PREFIX}/daily_equity_positions/date={date_str}/data.csv"
        if _key_exists(key):
            print(f"  {date_str}: already present — skipped")
            continue
        rows = group[["symbol", "name", "isin", "currency", "asset_type",
                       "fx_rate_to_base", "price", "shares", "value_eur"]].copy()
        rows.columns = ["symbol", "name", "isin", "currency", "asset_type",
                        "fx_rate_to_base", "mark_price", "position", "value_eur"]
        cost_map = {op["symbol"]: op.get("cost_basis_price", 0) for _, op in opens.iterrows()}
        rows["cost_basis_price"] = rows["symbol"].map(cost_map).fillna(0)
        rows["fifo_pnl_unrealized"] = 0.0
        _write_csv(rows, key)


def backfill_fx_positions() -> None:
    print("[daily_fx_positions]")
    prior   = load_prior_positions()
    fx_snap = load_fx_positions()
    latest_fx = fx_snap[fx_snap["date"] == fx_snap["date"].max()]

    # Get all equity dates to know which dates need FX position rows
    all_dates = pd.DatetimeIndex(sorted(prior["date"].unique()))
    cash_pos  = build_cash_positions(prior, latest_fx, all_dates)

    for dt, group in cash_pos.groupby("date"):
        date_str = dt.strftime("%Y-%m-%d")
        key = f"{RAW_PREFIX}/daily_fx_positions/date={date_str}/data.csv"
        if _key_exists(key):
            print(f"  {date_str}: already present — skipped")
            continue

        fx_map = {row["fx_currency"]: row for _, row in latest_fx.iterrows()}
        rows = []
        for _, cash_row in group.iterrows():
            ccy = cash_row["currency"]
            snap = fx_map.get(ccy, {})
            rows.append({
                "fx_currency":   ccy,
                "quantity":      cash_row["shares"],
                "cost_price":    snap.get("cost_price", 1.0 if ccy == "EUR" else 0.0),
                "close_price":   cash_row["fx_rate_to_base"],
                "value_eur":     cash_row["value_eur"],
                "unrealized_pl": snap.get("unrealized_pl", 0.0) if ccy != "EUR" else 0.0,
            })
        _write_csv(pd.DataFrame(rows), key)


if __name__ == "__main__":
    if not S3_BUCKET:
        print("ERROR: S3_BUCKET env var must be set")
        sys.exit(1)
    print(f"Backfilling to s3://{S3_BUCKET}/{RAW_PREFIX}/\n")
    backfill_equity_positions()
    print()
    backfill_fx_positions()
    print("\nDone. Now run: python -m precompute.build_derived")
