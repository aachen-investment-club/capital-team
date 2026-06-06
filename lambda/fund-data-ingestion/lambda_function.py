import os
import json
import math
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from decimal import Decimal

from botocore.exceptions import ClientError

import boto3
import numpy as np
import yfinance as yf

DYNAMODB_TABLE    = "fund-data"
S3_BUCKET         = "aic-fund-public-data"
S3_KEY            = "history/portfolio/nav_history.json"
DEPOSIT_LOG_KEY   = "history/portfolio/deposit_log.json"
TRADE_LOG_KEY     = "history/portfolio/trade_log.json"
EQUITY_POS_KEY    = "history/portfolio/equity_positions.json"
FX_POS_KEY        = "history/portfolio/fx_positions.json"
DERIVED_PREFIX    = "history/portfolio/derived"
MIN_POINTS_RISK   = 30

dynamodb = boto3.resource("dynamodb", region_name="eu-central-1")
s3       = boto3.client("s3",         region_name="eu-central-1")
table    = dynamodb.Table(DYNAMODB_TABLE)


def to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_decimal(i) for i in obj]
    return obj


# ── IBKR Flex Query ──────────────────────────────────────────────────────────

def fetch_flex_query():
    token    = os.environ["IBKR_FLEX_TOKEN"]
    query_id = os.environ["IBKR_QUERY_ID"]
    base     = "https://gdcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

    req_url = f"{base}/SendRequest?t={token}&q={query_id}&v=3"

    for send_attempt in range(5):
        with urllib.request.urlopen(req_url) as r:
            root = ET.fromstring(r.read())
        status     = root.findtext("Status")
        ref_code   = root.findtext("ReferenceCode")
        error_code = root.findtext("ErrorCode")
        if status == "Success":
            break
        if error_code in ("1001", "1004"):
            print(f"IBKR not ready ({error_code}), retrying in 30s… (attempt {send_attempt + 1}/5)")
            time.sleep(30)
        else:
            raise RuntimeError(f"Flex SendRequest failed: {status} / {error_code}: {root.findtext('ErrorMessage')}")
    else:
        raise TimeoutError("IBKR returned 1001 five times — statement not available yet")

    for attempt in range(6):
        time.sleep(5)
        get_url = f"{base}/GetStatement?t={token}&q={ref_code}&v=3"
        with urllib.request.urlopen(get_url) as r:
            xml_bytes = r.read()
        root2 = ET.fromstring(xml_bytes)
        if root2.tag == "FlexQueryResponse":
            return root2
        print(f"Attempt {attempt + 1}: still processing…")

    raise TimeoutError("Flex Query did not complete in time")


def parse_flex(root):
    stmt = root.find(".//FlexStatement")

    equity_rows   = stmt.findall(".//EquitySummaryByReportDateInBase")
    latest_equity = equity_rows[-1]
    report_date   = latest_equity.attrib["reportDate"]  # YYYYMMDD
    total_nav     = float(latest_equity.attrib["total"])

    positions = []
    for op in stmt.findall(".//OpenPosition"):
        fx_rate    = float(op.attrib.get("fxRateToBase", 1))
        mark_price = float(op.attrib.get("markPrice", 0))
        position   = float(op.attrib.get("position", 0))
        positions.append({
            "symbol":            op.attrib["symbol"],
            "description":       op.attrib["description"],
            "currency":          op.attrib["currency"],
            "assetCategory":     op.attrib["assetCategory"],
            "subCategory":       op.attrib["subCategory"],
            "percentOfNAV":      float(op.attrib["percentOfNAV"]),
            "securityID":        op.attrib["securityID"],
            "securityIDType":    op.attrib["securityIDType"],
            "fxRateToBase":      fx_rate,
            "markPrice":         mark_price,
            "position":          position,
            "costBasisPrice":    float(op.attrib.get("costBasisPrice", 0)),
            "fifoPnlUnrealized": float(op.attrib.get("fifoPnlUnrealized", 0)),
        })

    fx_positions = []
    for fp in stmt.findall(".//FxPosition"):
        value = float(fp.attrib["value"])
        fx_positions.append({
            "fxCurrency":    fp.attrib["fxCurrency"],
            "quantity":      float(fp.attrib["quantity"]),
            "value":         value,
            "costPrice":     float(fp.attrib.get("costPrice", 0)),
            "unrealizedPL":  float(fp.attrib["unrealizedPL"]),
            "percentOfTotal": round(value / total_nav * 100, 2),
        })

    new_trades = []
    for t in stmt.findall(".//Trade"):
        if t.get("assetCategory") == "CASH" and float(t.get("ibCommission", 0)) == 0:
            continue
        td = t.get("tradeDate", "")
        new_trades.append({
            "trade_date": f"{td[:4]}-{td[4:6]}-{td[6:]}",
            "symbol":     t.get("symbol"),
            "name":       t.get("description"),
            "isin":       t.get("securityID", ""),
            "currency":   t.get("currency"),
            "asset_type": t.get("subCategory"),
            "buy_sell":   t.get("buySell"),
            "quantity":   float(t.get("quantity", 0)),
            "proceeds":   float(t.get("proceeds", 0)),
            "commission": float(t.get("ibCommission", 0)),
        })

    date_iso = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"
    return date_iso, total_nav, positions, fx_positions, new_trades


# ── Yahoo Finance prices ─────────────────────────────────────────────────────

MARKET_TICKERS = {
    "^GSPC": "spxClose",
    "URTH":  "msciWorldClose",
    "EZU":   "msciEuropeClose",
    "AOR":   "sixtyFortyClose",
}


def fetch_market_prices(report_date_iso: str) -> dict:
    end_date = (date.fromisoformat(report_date_iso) + timedelta(days=1)).isoformat()
    results  = {}
    for symbol, key in MARKET_TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist   = ticker.history(start=report_date_iso, end=end_date, auto_adjust=True)
            if hist.empty:
                hist = ticker.history(period="5d", auto_adjust=True)
            results[key] = round(float(hist["Close"].iloc[-1]), 4)
        except Exception as exc:
            print(f"Warning: could not fetch {symbol}: {exc}")
            results[key] = None
    return results


# ── S3 History ───────────────────────────────────────────────────────────────

def load_history():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise


def save_history(history):
    s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY,
                  Body=json.dumps(history),
                  ContentType="application/json")


def load_deposit_log() -> list:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=DEPOSIT_LOG_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise


def recompute_history(history: list, deposit_log: list) -> list:
    """Replay the full history to compute unit-price-based fundNav.

    Deposits buy units at the unit price of the previous day, so they never
    inflate the reported return.  Running this on every invocation means a
    late-entered deposit automatically corrects all past entries.
    """
    if not history:
        return history

    deposits_by_date = {d["date"]: d["amount"] for d in deposit_log}
    total_units = history[0]["rawNav"]  # bootstrap: first rawNav defines unit count

    for i, entry in enumerate(history):
        if i > 0:
            deposit = deposits_by_date.get(entry["date"], 0)
            if deposit:
                prev_unit_price = history[i - 1]["rawNav"] / total_units
                total_units    += deposit / prev_unit_price
                if total_units <= 0:
                    raise ValueError(
                        f"deposit_log entry on {entry['date']} (amount={deposit}) "
                        f"makes totalUnits non-positive ({total_units:.4f}) — check the amount"
                    )

        entry["totalUnits"] = round(total_units, 6)
        entry["fundNav"]    = round(entry["rawNav"] / total_units, 6)

    return history


# ── S3 position helpers ────────────────────────────────────────────────────────

def _load_positions_json(key: str) -> list:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise


def _save_positions_json(key: str, records: list) -> None:
    s3.put_object(Bucket=S3_BUCKET, Key=key,
                  Body=json.dumps(records),
                  ContentType="application/json")


def _append_position_records(existing: list, new_records: list, report_date: str) -> list:
    """Replace any records for report_date then append new ones (idempotent)."""
    filtered = [r for r in existing if r.get("date") != report_date]
    return filtered + new_records


def _append_positions_to_s3(report_date: str, positions: list, fx_positions: list) -> None:
    try:
        if positions:
            eq_new = [{
                "date":                report_date,
                "symbol":              p["symbol"],
                "name":                p["description"],
                "isin":                p.get("securityID", ""),
                "currency":            p["currency"],
                "asset_type":          p.get("subCategory", ""),
                "fx_rate_to_base":     float(p.get("fxRateToBase", 1)),
                "mark_price":          float(p.get("markPrice", 0)),
                "position":            float(p.get("position", 0)),
                "value_eur":           float(p.get("markPrice", 0)) * float(p.get("position", 0)) * float(p.get("fxRateToBase", 1)),
                "cost_basis_price":    float(p.get("costBasisPrice", 0)),
                "fifo_pnl_unrealized": float(p.get("fifoPnlUnrealized", 0)),
            } for p in positions]
            updated = _append_position_records(_load_positions_json(EQUITY_POS_KEY), eq_new, report_date)
            _save_positions_json(EQUITY_POS_KEY, updated)
            print(f"  equity_positions: {len(eq_new)} records for {report_date}, {len(updated)} total")

        if fx_positions:
            fx_new = [{
                "date":             report_date,
                "fx_currency":      p["fxCurrency"],
                "quantity":         float(p.get("quantity", 0)),
                "cost_price":       float(p.get("costPrice", 0)),
                "value_eur":        float(p.get("value", 0)),
                "unrealized_pl":    float(p.get("unrealizedPL", 0)),
                "percent_of_total": float(p.get("percentOfTotal", 0)),
            } for p in fx_positions]
            updated = _append_position_records(_load_positions_json(FX_POS_KEY), fx_new, report_date)
            _save_positions_json(FX_POS_KEY, updated)
            print(f"  fx_positions: {len(fx_new)} records for {report_date}, {len(updated)} total")

    except Exception as exc:
        print(f"  Warning: could not write positions to S3: {exc}")


# ── Trade log ────────────────────────────────────────────────────────────────

def load_trade_log_s3() -> list:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=TRADE_LOG_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return []
        raise


def append_trades(existing: list, new_trades: list) -> list:
    existing_dates = {t["trade_date"] for t in existing}
    to_add = [t for t in new_trades if t["trade_date"] not in existing_dates]
    if not to_add:
        return existing
    return sorted(existing + to_add, key=lambda t: t["trade_date"])


def save_trade_log_s3(records: list) -> None:
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=TRADE_LOG_KEY,
        Body=json.dumps(records),
        ContentType="application/json",
    )


# ── Build derived ─────────────────────────────────────────────────────────────

import pandas as pd

_BENCHMARK_LABELS = {
    "spxClose":        "SPX",
    "msciWorldClose":  "MSCI_WORLD",
    "msciEuropeClose": "MSCI_EUROPE",
    "sixtyFortyClose": "60_40",
}


def _write_derived(df: pd.DataFrame, table: str) -> None:
    key = f"{DERIVED_PREFIX}/{table}.json"
    body = df.to_json(orient="records", date_format="iso")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
    print(f"  wrote s3://{S3_BUCKET}/{key}  ({len(df):,} rows)")


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
    _write_derived(out,"portfolio_and_benchmarks")


def _build_daily_weightings(history: list) -> None:
    eq_records = _load_positions_json(EQUITY_POS_KEY)
    fx_records = _load_positions_json(FX_POS_KEY)
    print(f"    equity_positions: {len(eq_records)} records loaded")
    print(f"    fx_positions    : {len(fx_records)} records loaded")

    if not eq_records:
        print("    no equity position data found — skipped")
        return

    eq = pd.DataFrame(eq_records)
    fx = pd.DataFrame(fx_records) if fx_records else pd.DataFrame()
    eq["date"] = pd.to_datetime(eq["date"])
    if not fx.empty:
        fx["date"] = pd.to_datetime(fx["date"])

    if eq.empty:
        return

    eq = eq.sort_values(["symbol", "date"])
    eq["prev_price"]        = eq.groupby("symbol")["mark_price"].shift(1).fillna(eq["cost_basis_price"])
    eq["daily_return"]      = (eq["mark_price"] / eq["prev_price"] - 1).fillna(0.0)
    eq["cumulative_return"] = (eq["mark_price"] / eq["cost_basis_price"] - 1).fillna(0.0)
    eq["pct_nav"]           = 0.0
    eq["category"]          = eq["asset_type"].map(
        lambda x: "ETF" if x == "ETF" else ("Cash" if x == "CASH" else "Equities")
    )

    cash_rows = []
    if not fx.empty:
        latest_fx = fx[fx["date"] == fx["date"].max()]
        ccy_snap  = {row["fx_currency"]: row for _, row in latest_fx.iterrows()}
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
    parts = [eq[[c for c in shared if c in eq.columns]]]
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
    dates = out["date"].dt.strftime("%Y-%m-%d")
    print(f"    daily_weightings: {len(out)} rows  {dates.min()} → {dates.max()}"
          f"  symbols={sorted(out['symbol'].unique().tolist())}")
    _write_derived(out, "daily_weightings")


def run_build_derived(history: list) -> None:
    print("  [portfolio_and_benchmarks]")
    try:
        _build_portfolio_and_benchmarks(history)
        print("  [portfolio_and_benchmarks] OK")
    except Exception as exc:
        print(f"  [portfolio_and_benchmarks] FAILED: {exc}")

    print("  [daily_weightings]")
    try:
        _build_daily_weightings(history)
        print("  [daily_weightings] OK")
    except Exception as exc:
        print(f"  [daily_weightings] FAILED: {exc}")


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(history: list, current_nav_multiple: float, current_date: str) -> dict:
    n   = len(history)
    tba = "TBA"

    nav_multiple = round(current_nav_multiple, 4)

    def return_pct(past_nav):
        return round((current_nav_multiple - past_nav) / past_nav * 100, 2) if past_nav else 0.0

    change_yesterday = return_pct(history[-1]["fundNav"]) if n >= 1 else 0.0

    current_dt = datetime.fromisoformat(current_date)

    mtd_nav = next(
        (h["fundNav"] for h in reversed(history)
         if datetime.fromisoformat(h["date"]).month != current_dt.month),
        history[0]["fundNav"] if history else current_nav_multiple,
    )
    change_mtd = return_pct(mtd_nav)

    ytd_nav = next(
        (h["fundNav"] for h in reversed(history)
         if datetime.fromisoformat(h["date"]).year != current_dt.year),
        history[0]["fundNav"] if history else current_nav_multiple,
    )
    change_ytd = return_pct(ytd_nav)

    if n + 1 < MIN_POINTS_RISK:
        return {
            "currentNavMultiple": nav_multiple,
            "changeYesterdayPct": change_yesterday,
            "changeMonthlyPct":   change_mtd,
            "changeYTDPct":       change_ytd,
            "volatilityAnnPct":   tba,
            "sharpeRatio":        tba,
            "correlationSPX":     tba,
            "lastUpdated":        current_date,
        }

    navs = np.array([h["fundNav"] for h in history] + [current_nav_multiple])
    spxs = np.array([h.get("spxClose", 0) for h in history])

    daily_ret_nav = np.diff(navs) / navs[:-1]
    daily_ret_spx = np.diff(spxs) / spxs[:-1] if len(spxs) >= 2 else np.array([])

    vol_ann   = round(float(np.std(daily_ret_nav) * math.sqrt(252) * 100), 2)
    rf_daily  = 0.045 / 252
    excess    = daily_ret_nav - rf_daily
    std_excess = np.std(excess)
    sharpe    = round(float(np.mean(excess) / std_excess * math.sqrt(252)), 2) if std_excess > 0 else 0.0

    correlation = tba
    if len(daily_ret_spx) >= MIN_POINTS_RISK - 1:
        n_common    = min(len(daily_ret_nav), len(daily_ret_spx))
        corr_matrix = np.corrcoef(daily_ret_nav[-n_common:], daily_ret_spx[-n_common:])
        correlation = round(float(corr_matrix[0, 1]), 3)

    return {
        "currentNavMultiple": nav_multiple,
        "changeYesterdayPct": change_yesterday,
        "changeMonthlyPct":   change_mtd,
        "changeYTDPct":       change_ytd,
        "volatilityAnnPct":   vol_ann,
        "sharpeRatio":        sharpe,
        "correlationSPX":     correlation,
        "lastUpdated":        current_date,
    }


# ── Handler ──────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    print("=" * 60)
    print("fund-data-ingestion START")
    print("=" * 60)

    # ── 1. Fetch & parse Flex Query ───────────────────────────────
    print("\n[1/7] Fetching IBKR Flex Query…")
    flex_root = fetch_flex_query()
    report_date, total_nav, positions, fx, new_trades = parse_flex(flex_root)
    print(f"  report_date : {report_date}")
    print(f"  total_nav   : {total_nav:.2f} EUR")
    print(f"  positions   : {len(positions)} equity  ({[p['symbol'] for p in positions]})")
    print(f"  fx_positions: {len(fx)} FX  ({[p['fxCurrency'] for p in fx]})")
    print(f"  new_trades  : {len(new_trades)} trade(s) in query")

    # ── 2. Market prices ─────────────────────────────────────────
    print("\n[2/7] Fetching market prices…")
    prices = fetch_market_prices(report_date)
    for k, v in prices.items():
        print(f"  {k}: {v}")

    # ── 3. NAV history ────────────────────────────────────────────
    print("\n[3/7] Loading & updating NAV history…")
    deposit_log = load_deposit_log()
    print(f"  deposit_log : {len(deposit_log)} entries")
    raw_history = load_history()
    print(f"  nav_history : {len(raw_history)} existing entries")
    history = recompute_history(raw_history, deposit_log)

    if not any(h["date"] == report_date for h in history):
        if history:
            prev_units      = history[-1]["totalUnits"]
            prev_unit_price = history[-1]["rawNav"] / prev_units
        else:
            prev_units      = total_nav
            prev_unit_price = 1.0

        deposit     = sum(d["amount"] for d in deposit_log if d["date"] == report_date)
        total_units = prev_units + (deposit / prev_unit_price if deposit else 0)

        entry = {
            "date":       report_date,
            "fundNav":    round(total_nav / total_units, 6),
            "rawNav":     round(total_nav, 4),
            "totalUnits": round(total_units, 6),
            **{k: v for k, v in prices.items() if v is not None},
        }
        history.append(entry)
        print(f"  new entry   : fundNav={entry['fundNav']}, rawNav={entry['rawNav']}, units={entry['totalUnits']}")
        if deposit:
            print(f"  deposit     : {deposit:.2f} EUR applied")
    else:
        print(f"  entry for {report_date} already present — recomputed only")

    save_history(history)
    print(f"  saved nav_history: {len(history)} entries  ({history[0]['date']} → {history[-1]['date']})")

    # ── 4. Metrics → DynamoDB ─────────────────────────────────────
    print("\n[4/7] Computing metrics & writing DynamoDB…")
    current_nav_multiple = history[-1]["fundNav"]
    metrics = compute_metrics(history[:-1], current_nav_multiple, report_date)
    print(f"  NAV×={metrics['currentNavMultiple']}  day={metrics['changeYesterdayPct']}%"
          f"  MTD={metrics['changeMonthlyPct']}%  YTD={metrics['changeYTDPct']}%")
    print(f"  vol={metrics['volatilityAnnPct']}  sharpe={metrics['sharpeRatio']}"
          f"  corr(SPX)={metrics['correlationSPX']}")

    ddb_positions = [{
        "symbol":        p["symbol"],
        "description":   p["description"],
        "currency":      p["currency"],
        "assetCategory": p["assetCategory"],
        "subCategory":   p["subCategory"],
        "percentOfNAV":  p["percentOfNAV"],
        "securityID":    p["securityID"],
        "securityIDType": p["securityIDType"],
    } for p in positions]

    ddb_fx = [{
        "fxCurrency":    fp["fxCurrency"],
        "percentOfTotal": fp["percentOfTotal"],
    } for fp in fx]

    table.put_item(Item=to_decimal({"pk": "METRICS", **metrics}))
    print("  DynamoDB METRICS: OK")
    table.put_item(Item=to_decimal({
        "pk":          "POSITIONS",
        "positions":   ddb_positions,
        "fxPositions": ddb_fx,
        "lastUpdated": report_date,
    }))
    print(f"  DynamoDB POSITIONS: OK  ({len(ddb_positions)} equity, {len(ddb_fx)} FX)")

    # ── 5. Positions → S3 ────────────────────────────────────────
    print("\n[5/7] Appending positions to S3…")
    _append_positions_to_s3(report_date, positions, fx)

    # ── 6. Trade log → S3 ────────────────────────────────────────
    print("\n[6/7] Updating trade log…")
    trade_log = append_trades(load_trade_log_s3(), new_trades)
    save_trade_log_s3(trade_log)
    print(f"  {len(new_trades)} new trade(s) today, {len(trade_log)} total in log")

    # ── 7. Build derived tables ───────────────────────────────────
    print("\n[7/7] Building derived tables…")
    run_build_derived(history)

    print("\n" + "=" * 60)
    prices_str = "  ".join(f"{k}={v}" for k, v in prices.items() if v is not None)
    print(f"DONE  date={report_date}  NAV={total_nav:.2f}  NAV×={current_nav_multiple:.4f}")
    print(f"      {prices_str}")
    print("=" * 60)
    return {"statusCode": 200, "date": report_date}
