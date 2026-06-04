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
    import boto3
    _creds = boto3.Session().get_credentials()
    if _creds:
        _creds = _creds.get_frozen_credentials()
        _con.execute(f"SET s3_access_key_id='{_creds.access_key}';")
        _con.execute(f"SET s3_secret_access_key='{_creds.secret_key}';")
        if _creds.token:
            _con.execute(f"SET s3_session_token='{_creds.token}';")


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
    Build daily_weightings from the hive-partitioned raw CSV files written by the lambda.

    Reads:
      {RAW_PREFIX}/daily_equity_positions/date=YYYY-MM-DD/data.csv
      {RAW_PREFIX}/daily_fx_positions/date=YYYY-MM-DD/data.csv

    Columns out: date, symbol, name, isin, ccy, category, pct_nav, cumulative_return, daily_return
    """
    if _S3_BUCKET:
        eq_glob = f"s3://{_S3_BUCKET}/{_RAW_PREFIX}/daily_equity_positions/**/*.csv"
        fx_glob = f"s3://{_S3_BUCKET}/{_RAW_PREFIX}/daily_fx_positions/**/*.csv"
    else:
        eq_glob = str(pathlib.Path(_RAW_PREFIX) / "daily_equity_positions" / "**" / "*.csv")
        fx_glob = str(pathlib.Path(_RAW_PREFIX) / "daily_fx_positions" / "**" / "*.csv")

    # ── Equity positions ───────────────────────────────────────────────────────
    eq = _con.execute(f"""
        SELECT
            date::DATE           AS date,
            symbol, name, isin, currency, asset_type,
            fx_rate_to_base, mark_price,
            position             AS shares,
            value_eur,
            cost_basis_price
        FROM read_csv_auto('{eq_glob}', hive_partitioning=true)
    """).df()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values(["symbol", "date"])

    eq["prev_price"]        = eq.groupby("symbol")["mark_price"].shift(1).fillna(eq["cost_basis_price"])
    eq["daily_return"]      = (eq["mark_price"] / eq["prev_price"] - 1).fillna(0.0)
    eq["cumulative_return"] = (eq["mark_price"] / eq["cost_basis_price"] - 1).fillna(0.0)
    eq["pct_nav"]           = 0.0  # recomputed below
    eq["category"]          = eq["asset_type"].map(
        lambda x: "ETF" if x == "ETF" else ("Cash" if x == "CASH" else "Equities")
    )

    # ── Cash positions from FX CSV ─────────────────────────────────────────────
    fx = _con.execute(f"""
        SELECT
            date::DATE   AS date,
            fx_currency, quantity, cost_price, value_eur
        FROM read_csv_auto('{fx_glob}', hive_partitioning=true)
    """).df()
    fx["date"] = pd.to_datetime(fx["date"])

    latest_fx = fx[fx["date"] == fx["date"].max()]
    ccy_snap  = {row["fx_currency"]: row for _, row in latest_fx.iterrows()}

    cash_rows = []
    for dt, grp in fx.groupby("date"):
        fx_val = {row["fx_currency"]: row["value_eur"] for _, row in grp.iterrows()}
        for ccy, sym, lbl in [
            ("EUR", "CASH_EUR", "Euro Cash"),
            ("GBP", "CASH_GBP", "British Pound Cash"),
            ("USD", "CASH_USD", "US Dollar Cash"),
        ]:
            val = fx_val.get(ccy, 0.0)
            if ccy in ccy_snap:
                snap         = ccy_snap[ccy]
                qty          = float(snap["quantity"])
                cost         = float(snap["cost_price"])
                inception_val = qty * (cost if cost > 0 else 1.0)
            else:
                inception_val = 0.0
            cash_rows.append({
                "date":              dt,
                "symbol":            sym,
                "name":              lbl,
                "isin":              "",
                "currency":          ccy,
                "asset_type":        "CASH",
                "category":          "Cash",
                "value_eur":         val,
                "pct_nav":           0.0,
                "cumulative_return": (val / inception_val - 1) if inception_val > 0 else 0.0,
                "daily_return":      0.0,  # filled below
            })

    cash_df = pd.DataFrame(cash_rows).sort_values(["symbol", "date"])
    cash_df["prev_val"]    = cash_df.groupby("symbol")["value_eur"].shift(1)
    cash_df["daily_return"] = (cash_df["value_eur"] / cash_df["prev_val"] - 1).fillna(0.0)
    cash_df.drop(columns=["prev_val"], inplace=True)

    # ── Merge equity + cash ───────────────────────────────────────────────────
    shared = ["date", "symbol", "name", "isin", "currency", "asset_type",
              "category", "value_eur", "pct_nav", "daily_return", "cumulative_return"]
    combined = pd.concat([eq[shared], cash_df[shared]], ignore_index=True)

    # ── Recompute pct_nav (prefer rawNav from nav_history as denominator) ─────
    nav_hist = _load_nav_history()
    if not nav_hist.empty and "rawNav" in nav_hist.columns:
        nav_lookup = nav_hist[["date", "rawNav"]].dropna().rename(columns={"rawNav": "total_nav"})
        combined   = combined.merge(nav_lookup, on="date", how="left")
        daily_sum  = combined.groupby("date")["value_eur"].transform("sum")
        combined["total_nav"] = combined["total_nav"].fillna(daily_sum)
    else:
        combined["total_nav"] = combined.groupby("date")["value_eur"].transform("sum")

    combined["pct_nav"] = combined["value_eur"] / combined["total_nav"] * 100
    combined["ccy"]     = combined["currency"]

    out = combined[["date", "symbol", "name", "isin", "ccy", "category",
                    "pct_nav", "cumulative_return", "daily_return"]] \
            .sort_values(["symbol", "date"]).reset_index(drop=True)
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
