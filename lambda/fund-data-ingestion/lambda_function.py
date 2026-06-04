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

DYNAMODB_TABLE  = "fund-data"
S3_BUCKET       = "aic-fund-public-data"
S3_KEY          = "history/nav_history.json"
DEPOSIT_LOG_KEY = "history/deposit_log.json"
RAW_PREFIX      = "history/raw"
MIN_POINTS_RISK = 30

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
        if error_code == "1001":
            print(f"IBKR not ready (1001), retrying in 30s… (attempt {send_attempt + 1}/5)")
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

    date_iso = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"
    return date_iso, total_nav, positions, fx_positions


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
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_KEY,
        Body=json.dumps(history),
        ContentType="application/json",
    )


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


# ── S3 raw positions writers (dashboard data) ─────────────────────────────────

def _write_positions_csv(report_date: str, positions: list[dict], fx_positions: list[dict]) -> None:
    """
    Write daily equity + FX position snapshots to S3 as CSV (no pyarrow needed).
    Hive-partitioned so precompute can glob all history.

    Paths:
      history/raw/daily_equity_positions/date=YYYY-MM-DD/data.csv
      history/raw/daily_fx_positions/date=YYYY-MM-DD/data.csv
    """
    try:
        import pandas as pd

        date_partition = f"date={report_date}"

        if positions:
            eq_df = pd.DataFrame([{
                "symbol":               p["symbol"],
                "name":                 p["description"],
                "isin":                 p.get("securityID", ""),
                "currency":             p["currency"],
                "asset_type":           p.get("subCategory", ""),
                "fx_rate_to_base":      float(p.get("fxRateToBase", 1)),
                "mark_price":           float(p.get("markPrice", 0)),
                "position":             float(p.get("position", 0)),
                "value_eur":            float(p.get("markPrice", 0)) * float(p.get("position", 0)) * float(p.get("fxRateToBase", 1)),
                "cost_basis_price":     float(p.get("costBasisPrice", 0)),
                "fifo_pnl_unrealized":  float(p.get("fifoPnlUnrealized", 0)),
            } for p in positions])
            key = f"{RAW_PREFIX}/daily_equity_positions/{date_partition}/data.csv"
            s3.put_object(Bucket=S3_BUCKET, Key=key,
                          Body=eq_df.to_csv(index=False).encode(),
                          ContentType="text/csv")
            print(f"  wrote s3://{S3_BUCKET}/{key}")

        if fx_positions:
            fx_df = pd.DataFrame([{
                "fx_currency":      p["fxCurrency"],
                "quantity":         float(p.get("quantity", 0)),
                "cost_price":       float(p.get("costPrice", 0)),
                "value_eur":        float(p.get("value", 0)),
                "unrealized_pl":    float(p.get("unrealizedPL", 0)),
                "percent_of_total": float(p.get("percentOfTotal", 0)),
            } for p in fx_positions])
            key = f"{RAW_PREFIX}/daily_fx_positions/{date_partition}/data.csv"
            s3.put_object(Bucket=S3_BUCKET, Key=key,
                          Body=fx_df.to_csv(index=False).encode(),
                          ContentType="text/csv")
            print(f"  wrote s3://{S3_BUCKET}/{key}")

    except Exception as exc:
        print(f"  Warning: could not write positions to S3: {exc}")


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
    flex_root                             = fetch_flex_query()
    report_date, total_nav, positions, fx = parse_flex(flex_root)
    prices                                = fetch_market_prices(report_date)

    deposit_log = load_deposit_log()
    history     = recompute_history(load_history(), deposit_log)

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

    save_history(history)

    current_nav_multiple = history[-1]["fundNav"]
    metrics = compute_metrics(history[:-1], current_nav_multiple, report_date)

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
    table.put_item(Item=to_decimal({
        "pk":          "POSITIONS",
        "positions":   ddb_positions,
        "fxPositions": ddb_fx,
        "lastUpdated": report_date,
    }))

    # Write raw positions to S3 for the dashboard precompute job
    _write_positions_csv(report_date, positions, fx)

    prices_str = ", ".join(f"{k}={v}" for k, v in prices.items() if v is not None)
    print(f"Done — date={report_date}, NAV={total_nav:.2f}, NAV×={current_nav_multiple:.4f}, {prices_str}")
    return {"statusCode": 200, "date": report_date}
