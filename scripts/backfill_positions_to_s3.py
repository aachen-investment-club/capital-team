"""
Full portfolio data setup: parse IBKR XMLs → upload to S3 → build derived tables.

Run this once to bootstrap the dashboard, and re-run whenever you want to rebuild
everything from scratch (e.g. after a bug fix or adding new XML files).

XML sources (data/ibkr/):
  prior_positions.xml          — historical daily prices
  trade_log.xml                — all trades since inception
  open_positions/YYYYMMDD.xml  — current open positions + FX snapshot

Writes to S3 (portfolio/):
  portfolio/nav_history.json         ← also copied to history/nav_history.json
  portfolio/equity_positions.json
  portfolio/fx_positions.json
  portfolio/trade_log.json
  portfolio/derived/portfolio_and_benchmarks.json
  portfolio/derived/daily_weightings.json

Usage:
    S3_BUCKET=aic-fund-public-data python scripts/backfill_positions_to_s3.py
"""
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET

import boto3
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent

S3_BUCKET  = os.environ.get("S3_BUCKET", "aic-fund-public-data")
AWS_REGION = os.environ.get("AWS_REGION", "eu-central-1")

# S3 key layout
NAV_KEY         = "history/portfolio/nav_history.json"
NAV_KEY_LEGACY  = "history/nav_history.json"
EQUITY_POS_KEY  = "history/portfolio/equity_positions.json"
FX_POS_KEY      = "history/portfolio/fx_positions.json"
TRADE_LOG_KEY   = "history/portfolio/trade_log.json"
DERIVED_PREFIX  = "history/portfolio/derived"

IBKR_DIR         = _ROOT / "data" / "ibkr"
PRIOR_POS_XML    = IBKR_DIR / "prior_positions.xml"
TRADE_LOG_XML    = IBKR_DIR / "trade_log.xml"
OPEN_POS_XML_DIR = IBKR_DIR / "open_positions"

s3 = boto3.client("s3", region_name=AWS_REGION)


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _put_json(key: str, records: list, log: bool = True) -> None:
    s3.put_object(Bucket=S3_BUCKET, Key=key,
                  Body=json.dumps(records),
                  ContentType="application/json")
    if log:
        print(f"  wrote s3://{S3_BUCKET}/{key}  ({len(records)} records)")


def _put_df_json(key: str, df: pd.DataFrame) -> None:
    body = df.to_json(orient="records", date_format="iso")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    print(f"  wrote s3://{S3_BUCKET}/{key}  ({len(df):,} rows)")


def _load_nav_history() -> list:
    """Read nav_history from S3 (legacy path for backward compat)."""
    from botocore.exceptions import ClientError
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=NAV_KEY_LEGACY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise


# ── IBKR XML parsers ──────────────────────────────────────────────────────────

def parse_trades(xml_str: str) -> pd.DataFrame:
    """Parse Trades XML from IBKR Flex Query.
    FX (CASH) trades with zero commission are excluded.
    """
    root = ET.fromstring(xml_str)
    rows = []
    for t in root.iter("Trade"):
        if t.get("assetCategory") == "CASH" and float(t.get("ibCommission", 0)) == 0:
            continue
        is_fx = t.get("assetCategory") == "CASH"
        rows.append({
            "trade_date": pd.to_datetime(t.get("tradeDate"), format="%Y%m%d"),
            "symbol":     t.get("symbol"),
            "name":       t.get("description"),
            "isin":       t.get("securityID"),
            "currency":   t.get("currency"),
            "asset_type": "FX" if is_fx else t.get("subCategory"),
            "quantity":   float(t.get("quantity", 0)),
            "proceeds":   float(t.get("proceeds", 0)),
            "commission": float(t.get("ibCommission", 0)),
            "buy_sell":   t.get("buySell"),
        })
    return pd.DataFrame(rows)


def parse_prior_positions(xml_str: str) -> pd.DataFrame:
    """Parse PriorPeriodPositions XML from IBKR Flex Query."""
    root = ET.fromstring(xml_str)
    rows = []
    for p in root.iter("PriorPeriodPosition"):
        rows.append({
            "date":            pd.to_datetime(p.get("date"), format="%Y%m%d"),
            "symbol":          p.get("symbol"),
            "name":            p.get("description"),
            "isin":            p.get("securityID"),
            "currency":        p.get("currency"),
            "asset_type":      p.get("subCategory"),
            "fx_rate_to_base": float(p.get("fxRateToBase", 1)),
            "price":           float(p.get("price", 0)),
            "prior_mtm_pnl":   float(p.get("priorMtmPnl", 0)),
        })
    return pd.DataFrame(rows)


def parse_fx_positions(xml_str: str) -> pd.DataFrame:
    """Parse FxPositions XML from IBKR Flex Query.
    cost_price is the EUR rate at which the cash was acquired.
    """
    root = ET.fromstring(xml_str)
    rows = []
    for p in root.iter("FxPosition"):
        rows.append({
            "date":          pd.to_datetime(p.get("reportDate"), format="%Y%m%d"),
            "fx_currency":   p.get("fxCurrency"),
            "quantity":      float(p.get("quantity", 0)),
            "cost_price":    float(p.get("costPrice", 0)),
            "value_eur":     float(p.get("value", 0)),
            "unrealized_pl": float(p.get("unrealizedPL", 0)),
        })
    return pd.DataFrame(rows)


def parse_open_positions(xml_str: str) -> pd.DataFrame:
    """Parse OpenPositions XML from IBKR Flex Query."""
    root = ET.fromstring(xml_str)
    rows = []
    for p in root.iter("OpenPosition"):
        rows.append({
            "date":                pd.to_datetime(p.get("reportDate"), format="%Y%m%d"),
            "symbol":              p.get("symbol"),
            "name":                p.get("description"),
            "isin":                p.get("securityID"),
            "currency":            p.get("currency"),
            "asset_type":          p.get("subCategory"),
            "fx_rate_to_base":     float(p.get("fxRateToBase", 1)),
            "price":               float(p.get("markPrice", 0)),
            "shares":              float(p.get("position", 0)),
            "cost_basis_price":    float(p.get("costBasisPrice", 0)),
            "pct_nav":             float(p.get("percentOfNAV", 0)),
            "fifo_pnl_unrealized": float(p.get("fifoPnlUnrealized", 0)),
        })
    return pd.DataFrame(rows)


# ── Position computation ──────────────────────────────────────────────────────

def _shares_from_trades(trades: pd.DataFrame) -> dict[str, float]:
    """Net share count per symbol (buys positive, sells negative)."""
    df = trades.copy()
    df.loc[df["buy_sell"] == "SELL", "quantity"] *= -1
    return df.groupby("symbol")["quantity"].sum().to_dict()


def build_daily_positions(
    prior_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    open_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a unified daily positions table from IBKR data sources.

    Prior rows are enriched with share counts from the trade log. Open positions
    take precedence for any date they cover. Prices are forward-filled within
    each symbol to cover exchange holidays.
    """
    shares_map = _shares_from_trades(trades_df)

    hist = prior_df.copy()
    hist["shares"] = hist["symbol"].map(shares_map)
    hist = hist.dropna(subset=["shares"])
    hist["value_eur"] = hist["price"] * hist["shares"] * hist["fx_rate_to_base"]

    cur = open_df[["date", "symbol", "name", "isin", "currency", "asset_type",
                   "fx_rate_to_base", "price", "shares"]].copy()
    cur["value_eur"] = cur["price"] * cur["shares"] * cur["fx_rate_to_base"]

    open_dates = set(open_df["date"].dt.normalize().unique())
    hist = hist[~hist["date"].dt.normalize().isin(open_dates)]

    cols = ["date", "symbol", "name", "isin", "currency", "asset_type",
            "fx_rate_to_base", "price", "shares", "value_eur"]
    combined = pd.concat([hist[cols], cur[cols]], ignore_index=True)

    all_dates   = sorted(combined["date"].unique())
    all_symbols = sorted(combined["symbol"].unique())
    grid = pd.MultiIndex.from_product([all_dates, all_symbols], names=["date", "symbol"])
    combined = (
        combined.set_index(["date", "symbol"])
        .reindex(grid)
        .reset_index()
        .sort_values(["symbol", "date"])
    )

    for col in ["name", "isin", "currency", "asset_type", "shares"]:
        combined[col] = combined.groupby("symbol")[col].ffill().bfill()
    combined["price"]           = combined.groupby("symbol")["price"].ffill()
    combined["fx_rate_to_base"] = combined.groupby("symbol")["fx_rate_to_base"].ffill()

    combined = combined.dropna(subset=["price"])
    combined["value_eur"] = combined["price"] * combined["shares"] * combined["fx_rate_to_base"]
    return combined.sort_values(["date", "symbol"]).reset_index(drop=True)


def build_cash_positions(
    prior_df: pd.DataFrame,
    fx_snap: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build daily cash position rows (CASH_EUR, CASH_GBP, CASH_USD) for every date."""
    fx_map: dict[str, dict] = {}
    for _, row in fx_snap.iterrows():
        ccy = row["fx_currency"]
        fx_map[ccy] = {
            "quantity":   row["quantity"],
            "cost_price": row["cost_price"] if row["cost_price"] > 0 else 1.0,
        }

    gbp_rates = (
        prior_df[prior_df["symbol"] == "TRNl"][["date", "fx_rate_to_base"]]
        .rename(columns={"fx_rate_to_base": "fx_gbp"})
        .set_index("date")["fx_gbp"]
    )
    usd_rates = (
        prior_df[prior_df["symbol"] == "URNU"][["date", "fx_rate_to_base"]]
        .rename(columns={"fx_rate_to_base": "fx_usd"})
        .set_index("date")["fx_usd"]
    )

    cash_rows = []
    for dt in all_dates:
        fx_gbp = gbp_rates.get(dt, fx_map.get("GBP", {}).get("cost_price", 1.165320039))
        fx_usd = usd_rates.get(dt, fx_map.get("USD", {}).get("cost_price", 0.853018161))
        for ccy, sym, name_str, fx in [
            ("EUR", "CASH_EUR", "Euro Cash",          1.0),
            ("GBP", "CASH_GBP", "British Pound Cash", fx_gbp),
            ("USD", "CASH_USD", "US Dollar Cash",      fx_usd),
        ]:
            qty = fx_map.get(ccy, {}).get("quantity", 0.0)
            cash_rows.append({
                "date":            dt,
                "symbol":          sym,
                "name":            name_str,
                "isin":            "",
                "currency":        ccy,
                "asset_type":      "CASH",
                "fx_rate_to_base": fx,
                "price":           1.0,
                "shares":          qty,
                "value_eur":       qty * fx,
            })
    return pd.DataFrame(cash_rows)


# ── XML parsing ───────────────────────────────────────────────────────────────

def _parse_xmls() -> tuple:
    prior  = parse_prior_positions(PRIOR_POS_XML.read_text())
    trades = parse_trades(TRADE_LOG_XML.read_text())

    open_dfs, fx_dfs = [], []
    for xml_path in sorted(OPEN_POS_XML_DIR.glob("*.xml")):
        xml_text = xml_path.read_text()
        open_dfs.append(parse_open_positions(xml_text))
        fx_df = parse_fx_positions(xml_text)
        if not fx_df.empty:
            fx_dfs.append(fx_df)

    opens = pd.concat(open_dfs, ignore_index=True) if open_dfs else pd.DataFrame()
    fx    = pd.concat(fx_dfs,   ignore_index=True) if fx_dfs   else pd.DataFrame()
    return prior, trades, opens, fx


# ── Raw position builders ─────────────────────────────────────────────────────

def build_equity_records(prior: pd.DataFrame, trades: pd.DataFrame, opens: pd.DataFrame) -> list:
    eq = build_daily_positions(prior, trades, opens)
    cost_map = opens.sort_values("date").groupby("symbol")["cost_basis_price"].last().to_dict()

    records = []
    for dt, group in eq.groupby("date"):
        date_str = dt.strftime("%Y-%m-%d")
        for _, row in group.iterrows():
            mp  = row["price"]
            pos = row["shares"]
            fx  = row["fx_rate_to_base"]
            cb  = cost_map.get(row["symbol"], 0)
            records.append({
                "date":                date_str,
                "symbol":              row["symbol"],
                "name":                row["name"],
                "isin":                row["isin"],
                "currency":            row["currency"],
                "asset_type":          row["asset_type"],
                "fx_rate_to_base":     round(fx, 8),
                "mark_price":          round(mp, 6),
                "position":            round(pos, 6),
                "value_eur":           round(mp * pos * fx, 4),
                "cost_basis_price":    round(cb, 6),
                "fifo_pnl_unrealized": round((mp - cb) * pos, 4),
            })
    return records


def build_fx_records(prior: pd.DataFrame, fx: pd.DataFrame, nav_by_date: dict) -> list:
    latest_fx   = fx[fx["date"] == fx["date"].max()]
    all_dates   = pd.DatetimeIndex(sorted(prior["date"].unique()))
    cash_pos    = build_cash_positions(prior, latest_fx, all_dates)
    fx_map      = {row["fx_currency"]: row for _, row in latest_fx.iterrows()}

    records = []
    for dt, group in cash_pos.groupby("date"):
        date_str  = dt.strftime("%Y-%m-%d")
        total_nav = nav_by_date.get(date_str, 0.0)
        for _, cash_row in group.iterrows():
            ccy       = cash_row["currency"]
            snap      = fx_map.get(ccy, {})
            value_eur = cash_row["value_eur"]
            records.append({
                "date":             date_str,
                "fx_currency":      ccy,
                "quantity":         round(cash_row["shares"], 6),
                "cost_price":       round(float(snap.get("cost_price", 1.0 if ccy == "EUR" else 0.0)), 8),
                "value_eur":        round(value_eur, 4),
                "unrealized_pl":    round(float(snap.get("unrealized_pl", 0.0)) if ccy != "EUR" else 0.0, 4),
                "percent_of_total": round(value_eur / total_nav * 100, 2) if total_nav else 0.0,
            })
    return records


def build_trade_records(trades: pd.DataFrame) -> list:
    return trades.assign(
        trade_date=trades["trade_date"].dt.strftime("%Y-%m-%d")
    ).to_dict(orient="records")


# ── Derived table builders ────────────────────────────────────────────────────

_BENCHMARK_LABELS = {
    "spxClose":        "SPX",
    "msciWorldClose":  "MSCI_WORLD",
    "msciEuropeClose": "MSCI_EUROPE",
    "sixtyFortyClose": "60_40",
}


def _build_portfolio_and_benchmarks(history: list) -> None:
    nav = pd.DataFrame(history)
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.sort_values("date").reset_index(drop=True)

    rows = []
    port = nav[["date", "fundNav"]].dropna().copy()
    port["prev"] = port["fundNav"].shift(1).fillna(port["fundNav"].iloc[0])
    for _, r in port.iterrows():
        rows.append({
            "date":         r["date"],
            "ticker":       "PORTFOLIO",
            "index_value":  round(r["fundNav"], 6),
            "daily_return": round((r["fundNav"] - r["prev"]) / r["prev"], 6),
        })

    for field, ticker in _BENCHMARK_LABELS.items():
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
    _put_df_json(f"{DERIVED_PREFIX}/portfolio_and_benchmarks.json", out)


def _build_daily_weightings(equity_records: list, fx_records: list, history: list) -> None:
    if not equity_records:
        print("  no equity records — skipped")
        return

    eq = pd.DataFrame(equity_records)
    fx = pd.DataFrame(fx_records) if fx_records else pd.DataFrame()
    eq["date"] = pd.to_datetime(eq["date"])
    if not fx.empty:
        fx["date"] = pd.to_datetime(fx["date"])

    eq = eq.sort_values(["symbol", "date"])
    eq["prev_price"]        = eq.groupby("symbol")["mark_price"].shift(1).fillna(eq["cost_basis_price"])
    eq["daily_return"]      = (eq["mark_price"] / eq["prev_price"] - 1).fillna(0.0)
    eq["cumulative_return"] = (eq["mark_price"] / eq["cost_basis_price"] - 1).fillna(0.0)
    eq["pct_nav"]           = 0.0
    eq["category"]          = eq["asset_type"].map(
        lambda x: "ETF" if x == "ETF" else ("Cash" if x == "CASH" else "Equities")
    )

    latest_fx = fx[fx["date"] == fx["date"].max()] if not fx.empty else pd.DataFrame()
    ccy_snap  = {row["fx_currency"]: row for _, row in latest_fx.iterrows()} if not latest_fx.empty else {}

    cash_rows = []
    for dt, grp in (fx.groupby("date") if not fx.empty else []):
        fx_val = {row["fx_currency"]: row["value_eur"] for _, row in grp.iterrows()}
        for ccy, sym, lbl in [
            ("EUR", "CASH_EUR", "Euro Cash"),
            ("GBP", "CASH_GBP", "British Pound Cash"),
            ("USD", "CASH_USD", "US Dollar Cash"),
        ]:
            val = fx_val.get(ccy, 0.0)
            if ccy in ccy_snap:
                snap          = ccy_snap[ccy]
                qty           = float(snap["quantity"])
                cost          = float(snap["cost_price"])
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
                "daily_return":      0.0,
            })

    cash_df = pd.DataFrame(cash_rows)
    if not cash_df.empty:
        cash_df = cash_df.sort_values(["symbol", "date"])
        cash_df["prev_val"]    = cash_df.groupby("symbol")["value_eur"].shift(1)
        cash_df["daily_return"] = (cash_df["value_eur"] / cash_df["prev_val"] - 1).fillna(0.0)
        cash_df.drop(columns=["prev_val"], inplace=True)

    shared = ["date", "symbol", "name", "isin", "currency", "asset_type",
              "category", "value_eur", "pct_nav", "daily_return", "cumulative_return"]
    parts = [eq[shared]]
    if not cash_df.empty:
        parts.append(cash_df[shared])
    combined = pd.concat(parts, ignore_index=True)

    nav_df = pd.DataFrame(history)[["date", "rawNav"]].dropna()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df = nav_df.rename(columns={"rawNav": "total_nav"})
    combined = combined.merge(nav_df, on="date", how="left")
    daily_sum = combined.groupby("date")["value_eur"].transform("sum")
    combined["total_nav"] = combined["total_nav"].fillna(daily_sum)
    combined["pct_nav"]   = combined["value_eur"] / combined["total_nav"] * 100
    combined["ccy"]       = combined["currency"]

    out = combined[["date", "symbol", "name", "isin", "ccy", "category",
                    "pct_nav", "cumulative_return", "daily_return"]] \
            .sort_values(["symbol", "date"]).reset_index(drop=True)
    _put_df_json(f"{DERIVED_PREFIX}/daily_weightings.json", out)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not S3_BUCKET:
        print("ERROR: S3_BUCKET env var must be set")
        sys.exit(1)

    print(f"Setting up portfolio data in s3://{S3_BUCKET}/portfolio/\n")

    # 1. Parse XML source files
    print("Parsing XML files…")
    prior, trades, opens, fx = _parse_xmls()
    print(f"  prior_positions : {len(prior)} rows")
    print(f"  trades          : {len(trades)} rows")
    print(f"  open_positions  : {len(opens)} rows")
    print(f"  fx_positions    : {len(fx)} rows\n")

    # 2. Load nav history (needed for percent_of_total and derived builds)
    print("Loading nav_history from S3…")
    nav_history = _load_nav_history()
    nav_by_date = {e["date"]: float(e["rawNav"]) for e in nav_history}
    print(f"  {len(nav_history)} entries ({list(nav_by_date.keys())[:3]}…)\n")

    # 3. Build raw position records
    print("[equity_positions]")
    equity_records = build_equity_records(prior, trades, opens)
    print(f"  {len(equity_records)} records\n")

    print("[fx_positions]")
    fx_records = build_fx_records(prior, fx, nav_by_date)
    print(f"  {len(fx_records)} records\n")

    print("[trade_log]")
    trade_records = build_trade_records(trades)
    print(f"  {len(trade_records)} records\n")

    # 4. Upload raw files
    print("Uploading raw files…")
    _put_json(EQUITY_POS_KEY, equity_records)
    _put_json(FX_POS_KEY,     fx_records)
    _put_json(TRADE_LOG_KEY,  trade_records)
    if nav_history:
        _put_json(NAV_KEY, nav_history)
    print()

    # 5. Build and upload derived tables
    print("Building derived tables…")
    try:
        _build_portfolio_and_benchmarks(nav_history)
    except Exception as exc:
        print(f"  portfolio_and_benchmarks failed: {exc}")

    try:
        _build_daily_weightings(equity_records, fx_records, nav_history)
    except Exception as exc:
        print(f"  daily_weightings failed: {exc}")

    print("\nDone — dashboard is ready.")
