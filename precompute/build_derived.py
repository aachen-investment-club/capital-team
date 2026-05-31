"""
Nightly precompute job: reads raw Parquet from S3 (or local data/),
writes derived Parquet to S3 (or local data/derived/).

Run locally:   python -m precompute.build_derived
Run on S3:     S3_BUCKET=aic-fund-public-data ... python -m precompute.build_derived
"""
import io
import json
import os
import pathlib
import sys

import duckdb
import pandas as pd

_ROOT           = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_S3_BUCKET      = os.getenv("S3_BUCKET", "")
_RAW_PREFIX     = os.getenv("RAW_PREFIX",     str(_ROOT / "data" / "raw"))
_DERIVED_PREFIX = os.getenv("DERIVED_PREFIX", str(_ROOT / "data" / "derived"))
_AWS_REGION     = os.getenv("AWS_REGION", "eu-central-1")
_NAV_KEY        = "history/nav_history.json"

_con = duckdb.connect(":memory:")
if _S3_BUCKET:
    _con.execute("INSTALL httpfs; LOAD httpfs;")
    _con.execute(f"SET s3_region='{_AWS_REGION}';")


def _raw_partitioned(table: str) -> str:
    if _S3_BUCKET:
        return f"s3://{_S3_BUCKET}/{_RAW_PREFIX}/{table}/**/*.parquet"
    return str(_ROOT / "data" / _RAW_PREFIX / table / "**" / "*.parquet")


def _write(df: pd.DataFrame, table: str) -> None:
    if _S3_BUCKET:
        import boto3
        key = f"{_DERIVED_PREFIX}/{table}.parquet"
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow")
        boto3.client("s3", region_name=_AWS_REGION).put_object(
            Bucket=_S3_BUCKET, Key=key, Body=buf.getvalue()
        )
        print(f"  wrote s3://{_S3_BUCKET}/{key}  ({len(df):,} rows)")
    else:
        derived_dir = pathlib.Path(_DERIVED_PREFIX)
        derived_dir.mkdir(parents=True, exist_ok=True)
        path = derived_dir / f"{table}.parquet"
        df.to_parquet(path, index=False)
        print(f"  wrote {path}  ({len(df):,} rows)")


def _load_nav_history() -> pd.DataFrame:
    """Load nav_history.json in full (all columns stored by fund-data-ingestion lambda)."""
    if _S3_BUCKET:
        import boto3
        obj = boto3.client("s3", region_name=_AWS_REGION).get_object(
            Bucket=_S3_BUCKET, Key=_NAV_KEY
        )
        records = json.loads(obj["Body"].read())
    else:
        nav_path = _ROOT / "data" / "nav_history.json"
        if not nav_path.exists():
            return pd.DataFrame()
        records = json.loads(nav_path.read_text())

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── portfolio_and_benchmarks ──────────────────────────────────────────────────

# Mapping from nav_history.json field → dashboard ticker name
_BENCHMARK_FIELDS = {
    "spxClose":        "SPX",
    "msciWorldClose":  "MSCI_WORLD",
    "msciEuropeClose": "MSCI_EUROPE",
    "sixtyFortyClose": "60_40",
}


def build_portfolio_and_benchmarks() -> None:
    """
    Build portfolio_and_benchmarks from nav_history.json.

    nav_history.json is written daily by fund-data-ingestion and contains:
      fundNav      — portfolio NAV normalised to 1.0 at inception
      spxClose     — SPX closing price
      msciWorldClose / msciEuropeClose / sixtyFortyClose — benchmark closes

    Benchmark index values are normalised to 1.0 at the first date in the file
    so all series start at the same base for meaningful comparison.
    """
    nav = _load_nav_history()
    if nav.empty:
        print("  nav_history.json not found — skipped")
        return

    rows = []

    # PORTFOLIO: fundNav is already normalised to 1.0 at inception
    port = nav[["date", "fundNav"]].dropna().copy()
    port["prev"] = port["fundNav"].shift(1).fillna(port["fundNav"].iloc[0])
    for _, r in port.iterrows():
        rows.append({
            "date":         r["date"],
            "ticker":       "PORTFOLIO",
            "index_value":  round(r["fundNav"], 6),
            "daily_return": round((r["fundNav"] - r["prev"]) / r["prev"], 6),
        })

    # BENCHMARKS: normalise raw close prices to 1.0 at first available date
    for field, ticker in _BENCHMARK_FIELDS.items():
        if field not in nav.columns:
            continue
        bm = nav[["date", field]].dropna(subset=[field]).copy()
        if bm.empty:
            continue
        base = bm[field].iloc[0]
        bm["index_value"]  = (bm[field] / base).round(6)
        bm["daily_return"] = bm[field].pct_change().fillna(0.0).round(6)
        for _, r in bm.iterrows():
            rows.append({
                "date":         r["date"],
                "ticker":       ticker,
                "index_value":  r["index_value"],
                "daily_return": r["daily_return"],
            })

    out = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    _write(out, "portfolio_and_benchmarks")


# ── daily_weightings (equity + cash, NAV-normalised) ─────────────────────────

def build_daily_weightings() -> None:
    """
    Build daily_weightings from local IBKR Parquet files.

    Columns: date, symbol, name, isin, ccy, category, pct_nav, cumulative_return, daily_return

    Equity rows: from prior_positions + open_positions + trade_log
    Cash rows  : CASH_EUR / CASH_GBP / CASH_USD derived from fx_positions quantities
                 and daily FX rates extracted from equity prior positions

    pct_nav is relative to total NAV = equity_value + cash_value per day.
    The S3 rawNav is the authoritative check — if available it is used as the
    NAV denominator so weights reconcile exactly with the public NAV report.
    """
    from lib.ibkr import (
        load_prior_positions, load_trade_log, load_open_positions, load_fx_positions,
        build_daily_positions, build_cash_positions, compute_weightings,
    )

    prior  = load_prior_positions()
    trades = load_trade_log()
    opens  = load_open_positions()
    fx_snap = load_fx_positions()

    # Latest FX snapshot (most recent date in fx_positions)
    latest_fx = fx_snap[fx_snap["date"] == fx_snap["date"].max()]

    # ── Equity positions ───────────────────────────────────────────────────────
    equity_pos = build_daily_positions(prior, trades, opens)
    equity_wr  = compute_weightings(equity_pos, opens)

    all_dates = pd.DatetimeIndex(sorted(equity_pos["date"].unique()))

    # ── Cash positions (constant quantities, variable FX) ─────────────────────
    cash_pos = build_cash_positions(prior, latest_fx, all_dates)

    # ── Merge equity + cash ───────────────────────────────────────────────────
    eq_cols   = ["date", "symbol", "name", "isin", "currency", "asset_type",
                 "value_eur", "pct_nav", "daily_return", "cumulative_return"]
    cash_cols = ["date", "symbol", "name", "isin", "currency", "asset_type", "value_eur"]

    df_eq   = equity_wr[eq_cols].copy()
    df_cash = cash_pos[cash_cols].copy()

    # Cash daily_return = FX rate change; cumulative_return = FX gain from inception
    df_cash = df_cash.sort_values(["symbol", "date"])
    df_cash["prev_val"] = df_cash.groupby("symbol")["value_eur"].shift(1)
    df_cash["daily_return"] = (
        df_cash["value_eur"] / df_cash["prev_val"] - 1
    ).fillna(0.0)

    # Inception cost prices from fx_snap for cumulative return
    inception_fx = {
        row["fx_currency"]: row["cost_price"]
        for _, row in latest_fx.iterrows()
    }
    qty_map = {
        row["fx_currency"]: row["quantity"]
        for _, row in latest_fx.iterrows()
    }
    for ccy, sym in [("EUR", "CASH_EUR"), ("GBP", "CASH_GBP"), ("USD", "CASH_USD")]:
        inception_rate = inception_fx.get(ccy, 1.0) or 1.0
        inception_val  = qty_map.get(ccy, 0.0) * inception_rate
        mask = df_cash["symbol"] == sym
        if ccy == "EUR":
            df_cash.loc[mask, "cumulative_return"] = 0.0
        else:
            vals = df_cash.loc[mask, "value_eur"]
            df_cash.loc[mask, "cumulative_return"] = (
                (vals / inception_val - 1) if inception_val > 0 else 0.0
            )
    df_cash["cumulative_return"] = df_cash["cumulative_return"].fillna(0.0)
    df_cash.drop(columns=["prev_val"], inplace=True)

    # Add placeholder pct_nav to cash before concat (will be recomputed below)
    df_cash["pct_nav"] = 0.0
    combined = pd.concat([df_eq, df_cash[eq_cols]], ignore_index=True)

    # ── Recompute pct_nav relative to total NAV (equity + cash) ───────────────
    # Prefer rawNav from nav_history as denominator; fall back to summing positions
    nav_hist = _load_nav_history()
    if not nav_hist.empty and "rawNav" in nav_hist.columns:
        nav_lookup = nav_hist[["date", "rawNav"]].dropna().rename(columns={"rawNav": "total_nav"})
        combined = combined.merge(nav_lookup, on="date", how="left")
        daily_sum = combined.groupby("date")["value_eur"].transform("sum")
        combined["total_nav"] = combined["total_nav"].fillna(daily_sum)
    else:
        combined["total_nav"] = combined.groupby("date")["value_eur"].transform("sum")

    combined["pct_nav"] = combined["value_eur"] / combined["total_nav"] * 100

    # ── Category (for theme/basket grouping) ──────────────────────────────────
    def _category(asset_type: str) -> str:
        if asset_type == "ETF":
            return "ETF"
        if asset_type == "CASH":
            return "Cash"
        return "Equities"

    combined["category"] = combined["asset_type"].map(_category)
    combined["ccy"] = combined["currency"]

    out_cols = ["date", "symbol", "name", "isin", "ccy", "category",
                "pct_nav", "cumulative_return", "daily_return"]
    out = combined[out_cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    _write(out, "daily_weightings")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building derived tables…")

    print("\n[portfolio_and_benchmarks]")
    try:
        build_portfolio_and_benchmarks()
    except Exception as e:
        print(f"  skipped: {e}")

    print("\n[daily_weightings]")
    try:
        build_daily_weightings()
    except Exception as e:
        print(f"  failed: {e}")
        raise

    print("\nDone.")


if __name__ == "__main__":
    main()
