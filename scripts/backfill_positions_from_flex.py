"""
Backfill daily_positions, position_returns, and instrument_metadata by:
  1. Fetching the trade log from the existing IBKR Flex query
  2. Reconstructing share counts forward from inception
  3. Fetching daily closing prices + FX rates from yFinance
  4. Computing pct_nav = (shares x price_in_EUR) / total_nav_EUR x 100

IBKR Flex query setup (one-time):
  The existing query already covers the last 30 days, which spans the full
  fund history from inception. You just need to add the Trades section to it
  in Account Management -> Reports -> Flex Queries.
  Include fields: Symbol, Buy/Sell, Quantity, Trade Price, Proceeds,
                  IB Commission, Currency, Trade Date, ISIN / Security ID

Usage:
    IBKR_FLEX_TOKEN=<token> IBKR_QUERY_ID=<id> \
        python scripts/backfill_positions_from_flex.py

    # Dry-run: write to data/raw/ instead of S3
    IBKR_FLEX_TOKEN=<token> IBKR_QUERY_ID=<id> \
        python scripts/backfill_positions_from_flex.py --local

After running:
    python -m precompute.build_derived
    aws s3 sync data/derived/ s3://aic-fund-public-data/history/derived/ --region eu-central-1
"""
import argparse
import io
import json
import os
import pathlib
import subprocess
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

S3_BUCKET  = "aic-fund-public-data"
AWS_REGION = "eu-central-1"
RAW_PREFIX = "history/raw"
NAV_KEY    = "history/nav_history.json"

ROOT = pathlib.Path(__file__).resolve().parent.parent

# IBKR symbol -> yFinance ticker (add overrides here as needed)
_CCY_SUFFIX = {"EUR": ".DE", "GBP": ".L", "CHF": ".SW", "USD": ""}
_SYMBOL_OVERRIDES = {
    "SAN":  "SAN.MC",
    "TRNl": "TRN.L",
    "ESIF": "ESIF.L",
    "WDEF": "WDEF.L",
    "URNU": "URNU.L",
    "XLUS": "XLUS.L",
}

# FX pairs: settlement currency -> EUR conversion rate ticker
_FX_PAIRS = {
    "GBP": "GBPEUR=X",
    "USD": "USDEUR=X",
    "CHF": "CHFEUR=X",
}


def _yf_symbol(ibkr_symbol: str, currency: str) -> str:
    if ibkr_symbol in _SYMBOL_OVERRIDES:
        return _SYMBOL_OVERRIDES[ibkr_symbol]
    return ibkr_symbol + _CCY_SUFFIX.get(currency, "")


def _category(asset_category: str, sub_category: str) -> str:
    if sub_category == "ETF":
        return "ETF"
    return {"STK": "Equities", "BOND": "Bonds", "CASH": "Cash"}.get(asset_category, asset_category)


# ── Flex query ────────────────────────────────────────────────────────────────

def fetch_flex_query(token: str, query_id: str):
    base    = "https://gdcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
    req_url = f"{base}/SendRequest?t={token}&q={query_id}&v=3"

    for attempt in range(5):
        with urllib.request.urlopen(req_url) as r:
            root = ET.fromstring(r.read())
        status     = root.findtext("Status")
        ref_code   = root.findtext("ReferenceCode")
        error_code = root.findtext("ErrorCode")
        if status == "Success":
            break
        if error_code == "1001":
            print(f"  IBKR not ready (1001), retrying in 30s... ({attempt + 1}/5)")
            time.sleep(30)
        else:
            raise RuntimeError(
                f"Flex SendRequest failed: {status} / {error_code}: {root.findtext('ErrorMessage')}"
            )
    else:
        raise TimeoutError("IBKR returned 1001 five times")

    for attempt in range(10):
        time.sleep(5)
        with urllib.request.urlopen(f"{base}/GetStatement?t={token}&q={ref_code}&v=3") as r:
            xml_bytes = r.read()
        root2 = ET.fromstring(xml_bytes)
        if root2.tag == "FlexQueryResponse":
            return root2
        print(f"  Waiting for statement... ({attempt + 1}/10)")

    raise TimeoutError("Flex Query did not complete in time")


def parse_trades(root) -> list[dict]:
    """
    Parse trade elements into a flat list.
    Handles both <Trade> (from the "Trades" section) and
    <TradeConfirm> (from "Trade Confirmations") — whichever your query returns.
    """
    trades = []
    for stmt in root.findall(".//FlexStatement"):
        for tag in ("Trade", "TradeConfirm"):
            for tc in stmt.findall(f".//{tag}"):
                # Skip IBKR's internal FX rounding entries — they have assetCategory=CASH
                # and zero commission. Real user-initiated FX conversions always have a fee.
                if tc.attrib.get("assetCategory") == "CASH" and float(tc.attrib.get("ibCommission", 0)) == 0:
                    continue

                # Date: prefer tradeDate, fall back to dateTime (YYYYMMDD;HHMMSS) or reportDate
                raw_date = (
                    tc.attrib.get("tradeDate")
                    or (tc.attrib.get("dateTime") or "")[:8]
                    or tc.attrib.get("reportDate", "")
                )
                if not raw_date or len(raw_date) < 8:
                    continue
                trade_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

                buy_sell = tc.attrib.get("buySell", "")
                qty      = abs(float(tc.attrib.get("quantity", 0)))
                proceeds = abs(float(tc.attrib.get("proceeds", 0)))
                if qty == 0:
                    continue

                # tradePrice is absent in some query configs — derive from proceeds/qty
                raw_price = tc.attrib.get("tradePrice")
                price = float(raw_price) if raw_price else (proceeds / qty if qty else 0.0)

                trades.append({
                    "date":           trade_date,
                    "symbol":         tc.attrib["symbol"],
                    "description":    tc.attrib.get("description", ""),
                    "isin":           tc.attrib.get("isin") or tc.attrib.get("securityID", ""),
                    "currency":       tc.attrib["currency"],
                    "commission_ccy": tc.attrib.get("ibCommissionCurrency", tc.attrib.get("currency", "EUR")),
                    "assetCategory":  tc.attrib.get("assetCategory", "STK"),
                    "subCategory":    tc.attrib.get("subCategory", ""),
                    "action":         "BUY" if buy_sell.upper().startswith("B") else "SELL",
                    "shares":         qty,
                    "price":          round(price, 4),
                    "proceeds":       round(proceeds, 4),
                    "commission":     abs(float(tc.attrib.get("ibCommission", 0))),
                })

    trades.sort(key=lambda t: t["date"])
    return trades


# ── yFinance data ─────────────────────────────────────────────────────────────

def fetch_prices_and_fx(
    symbols_ccy: dict[str, str],
    date_from: str,
    date_to: str,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, str]]:
    """
    Returns:
        prices:           {symbol: Close price series}
        fx_rates:         {currency: CCY/EUR series}
        price_currencies: {symbol: actual currency yFinance uses for that ticker}
    """
    start = (date.fromisoformat(date_from) - timedelta(days=7)).isoformat()
    end   = (date.fromisoformat(date_to)   + timedelta(days=2)).isoformat()

    prices:           dict[str, pd.Series] = {}
    price_currencies: dict[str, str]       = {}   # actual trading currency per symbol

    for sym, ibkr_ccy in symbols_ccy.items():
        yf_sym = _yf_symbol(sym, ibkr_ccy)
        try:
            ticker      = yf.Ticker(yf_sym)
            hist        = ticker.history(start=start, end=end, auto_adjust=True)
            yf_currency = (getattr(ticker.fast_info, "currency", "") or "").strip()

            if not hist.empty:
                hist.index = pd.to_datetime(hist.index).normalize().tz_localize(None)

                if yf_currency == "GBp":
                    # London-listed stocks quoted in pence → convert to pounds
                    hist = hist.copy()
                    hist["Close"] = hist["Close"] / 100
                    effective_ccy = "GBP"
                    note = "(GBp→GBP ÷100)"
                elif yf_currency in ("GBP", "USD", "EUR", "CHF", "SEK", "NOK"):
                    effective_ccy = yf_currency
                    note = f"({yf_currency})" if yf_currency != ibkr_ccy else ""
                else:
                    # Unknown or missing currency — fall back to IBKR's settlement currency
                    effective_ccy = ibkr_ccy
                    note = f"(unknown yF currency '{yf_currency}', using IBKR {ibkr_ccy})"

                prices[sym]           = hist["Close"]
                price_currencies[sym] = effective_ccy
                print(f"    {yf_sym:12}: {len(hist)} days {note}".rstrip())
            else:
                print(f"    {yf_sym:12}: no data -- check symbol override")
        except Exception as e:
            print(f"    {yf_sym:12}: error -- {e}")

    # Fetch FX rates for every currency that actually appears in prices
    needed_ccys = {c for c in price_currencies.values() if c != "EUR"}
    # Also include IBKR currencies (for cash FX tracking)
    needed_ccys |= {c for c in symbols_ccy.values() if c != "EUR"}

    fx_rates: dict[str, pd.Series] = {"EUR": pd.Series(dtype=float)}
    for ccy in needed_ccys:
        if ccy in fx_rates:
            continue
        pair = _FX_PAIRS.get(ccy)
        if not pair:
            print(f"    No FX pair for {ccy} -- will assume 1.0")
            continue
        try:
            hist = yf.Ticker(pair).history(start=start, end=end, auto_adjust=True)
            if not hist.empty:
                hist.index = pd.to_datetime(hist.index).normalize().tz_localize(None)
                fx_rates[ccy] = hist["Close"]
                print(f"    {pair:12}: {len(hist)} days")
            else:
                print(f"    {pair:12}: no data")
        except Exception as e:
            print(f"    {pair:12}: error -- {e}")

    return prices, fx_rates, price_currencies


def _get_price_eur(
    prices: dict,
    fx_rates: dict,
    price_currencies: dict,
    symbol: str,
    ibkr_ccy: str,
    ts: pd.Timestamp,
) -> float | None:
    """Return EUR-equivalent closing price using the actual yFinance currency, not IBKR's."""
    series = prices.get(symbol)
    if series is None or series.empty:
        return None

    idx = series.index.get_indexer([ts], method="pad")[0]
    if idx < 0:
        return None
    price_local = float(series.iloc[idx])

    # Use the currency yFinance actually trades the instrument in (may differ from IBKR)
    actual_ccy = price_currencies.get(symbol, ibkr_ccy)
    if actual_ccy == "EUR":
        return price_local

    fx_series = fx_rates.get(actual_ccy)
    if fx_series is None or fx_series.empty:
        return price_local

    fx_idx = fx_series.index.get_indexer([ts], method="pad")[0]
    if fx_idx < 0:
        return price_local
    return price_local * float(fx_series.iloc[fx_idx])


# ── nav_history ───────────────────────────────────────────────────────────────

def load_nav_history() -> dict[str, dict]:
    raw = subprocess.run(
        ["aws", "--region", AWS_REGION, "s3", "cp", f"s3://{S3_BUCKET}/{NAV_KEY}", "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {entry["date"]: entry for entry in json.loads(raw)}


# ── Reconstruction ────────────────────────────────────────────────────────────

def _cash_to_eur(fx_rates: dict, ccy: str, ts: pd.Timestamp) -> float:
    """Return the EUR conversion rate for a cash currency on a given date."""
    if ccy == "EUR":
        return 1.0
    series = fx_rates.get(ccy)
    if series is None or series.empty:
        return 1.0
    idx = series.index.get_indexer([ts], method="pad")[0]
    return float(series.iloc[idx]) if idx >= 0 else 1.0


def reconstruct_daily_positions(
    trades: list[dict],
    nav_by_date: dict[str, dict],
    prices: dict[str, pd.Series],
    fx_rates: dict[str, pd.Series],
    price_currencies: dict[str, str],
    symbols_meta: dict[str, dict],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Walk forward from inception applying trades to maintain share counts and
    per-currency cash balances. Cash is tracked separately per currency (EUR,
    GBP, USD) using FX conversion trades, rather than as a single EUR residual.
    """
    equity_by_date: dict[str, list] = defaultdict(list)
    fx_by_date: dict[str, list]     = defaultdict(list)
    for t in trades:
        if t["assetCategory"] == "CASH":
            fx_by_date[t["date"]].append(t)
        else:
            equity_by_date[t["date"]].append(t)

    # Pre-compute weighted-average execution price in EUR per symbol (first buy only).
    # Used as the inception-day return anchor so cumulative return starts from the
    # actual purchase price, not the closing price on the first trade day.
    exec_prices_eur: dict[str, float] = {}
    for d, day_trades in equity_by_date.items():
        ts_d = pd.Timestamp(d)
        buys_by_sym: dict[str, list] = defaultdict(list)
        for t in day_trades:
            if t["action"] == "BUY":
                buys_by_sym[t["symbol"]].append(t)
        for sym, buys in buys_by_sym.items():
            if sym not in exec_prices_eur:   # anchor to first buy date only
                total_shares   = sum(t["shares"]   for t in buys)
                total_proceeds = sum(
                    t["proceeds"] * _cash_to_eur(fx_rates, t["currency"], ts_d)
                    for t in buys
                )
                if total_shares > 0:
                    exec_prices_eur[sym] = total_proceeds / total_shares

    # Initialise: full NAV is EUR cash on inception day
    first_date = sorted(nav_by_date.keys())[0]
    cash: dict[str, float] = {"EUR": float(nav_by_date[first_date]["rawNav"])}
    shares: dict[str, float] = defaultdict(float)

    daily_positions_dfs: dict[str, pd.DataFrame]  = {}
    position_returns_dfs: dict[str, pd.DataFrame] = {}
    prev_prices_eur: dict[str, float] = {}

    for d in sorted(nav_by_date.keys()):
        ts            = pd.Timestamp(d)
        total_nav_eur = float(nav_by_date[d]["rawNav"])

        # ── Apply FX conversion trades (EUR.GBP, EUR.USD) ────────────────────
        for t in fx_by_date.get(d, []):
            parts = t["symbol"].split(".")
            if len(parts) != 2:
                continue
            base_ccy, quote_ccy = parts           # e.g. EUR, USD
            if t["action"] == "SELL":
                cash[base_ccy]  = cash.get(base_ccy,  0) - t["shares"]    # EUR sold
                cash[quote_ccy] = cash.get(quote_ccy, 0) + t["proceeds"]  # USD/GBP received
            else:
                cash[base_ccy]  = cash.get(base_ccy,  0) + t["shares"]
                cash[quote_ccy] = cash.get(quote_ccy, 0) - t["proceeds"]
            cash[t["commission_ccy"]] = cash.get(t["commission_ccy"], 0) - t["commission"]

        # ── Apply equity trades ───────────────────────────────────────────────
        for t in equity_by_date.get(d, []):
            sym = t["symbol"]
            if t["action"] == "BUY":
                shares[sym] = shares.get(sym, 0.0) + t["shares"]
                cash[t["currency"]]       = cash.get(t["currency"],       0) - t["proceeds"]
                cash[t["commission_ccy"]] = cash.get(t["commission_ccy"], 0) - t["commission"]
            else:
                shares[sym] = max(0.0, shares.get(sym, 0.0) - t["shares"])
                cash[t["currency"]]       = cash.get(t["currency"],       0) + t["proceeds"]
                cash[t["commission_ccy"]] = cash.get(t["commission_ccy"], 0) - t["commission"]

        if not shares and not any(v > 0 for v in cash.values()):
            print(f"  {d}: no positions yet")
            continue

        rows: list[dict]            = []
        total_mktval_eur            = 0.0
        pos_prices_eur: dict[str, float] = {}

        # ── Equity positions ──────────────────────────────────────────────────
        for sym, qty in shares.items():
            if qty <= 0:
                continue
            meta      = symbols_meta.get(sym, {})
            ccy       = meta.get("currency", "EUR")
            price_eur = _get_price_eur(prices, fx_rates, price_currencies, sym, ccy, ts)
            if price_eur is None:
                print(f"  Warning: no price for {sym} on {d}, skipping")
                continue
            mktval_eur = qty * price_eur
            total_mktval_eur += mktval_eur
            pos_prices_eur[sym] = price_eur
            rows.append({
                "symbol":   sym,
                "name":     meta.get("description", sym),
                "isin":     meta.get("isin", ""),
                "ccy":      ccy,
                "category": _category(meta.get("assetCategory", "STK"), meta.get("subCategory", "")),
                "_mktval":  mktval_eur,
            })

        # ── Cash positions per currency ───────────────────────────────────────
        for ccy, amount in cash.items():
            if amount <= 0:
                continue
            rate       = _cash_to_eur(fx_rates, ccy, ts)
            mktval_eur = amount * rate
            total_mktval_eur += mktval_eur
            rows.append({
                "symbol":   f"CASH_{ccy}",
                "name":     f"Cash ({ccy}) – Base Currency" if ccy == "EUR" else f"Cash ({ccy})",
                "isin":     "",
                "ccy":      ccy,
                "category": "Cash",
                "_mktval":  mktval_eur,
            })

        for row in rows:
            row["pct_nav"] = round(row.pop("_mktval") / total_nav_eur * 100, 4)

        daily_positions_dfs[d] = pd.DataFrame(rows)

        # ── Daily returns: equity + foreign cash FX return ───────────────────
        ret_rows = []
        for sym, price_today in pos_prices_eur.items():
            if sym in prev_prices_eur and prev_prices_eur[sym] > 0:
                # Normal day-over-day return
                ret_rows.append({
                    "symbol":       sym,
                    "daily_return": round((price_today - prev_prices_eur[sym]) / prev_prices_eur[sym], 6),
                })
            elif sym not in prev_prices_eur and sym in exec_prices_eur and exec_prices_eur[sym] > 0:
                # First day this position has a price: return from execution price to today's close.
                # This captures the intraday move from purchase price to closing price on trade day.
                ret_rows.append({
                    "symbol":       sym,
                    "daily_return": round((price_today - exec_prices_eur[sym]) / exec_prices_eur[sym], 6),
                })

        # FX return for foreign cash: daily change in CCY/EUR rate
        for ccy, amount in cash.items():
            if amount <= 0 or ccy == "EUR":
                continue
            fx_series = fx_rates.get(ccy)
            if fx_series is None or fx_series.empty:
                continue
            idx = fx_series.index.get_indexer([ts], method="pad")[0]
            if idx < 1:
                continue
            rate_today = float(fx_series.iloc[idx])
            rate_prev  = float(fx_series.iloc[idx - 1])
            if rate_prev > 0:
                ret_rows.append({
                    "symbol":       f"CASH_{ccy}",
                    "daily_return": round((rate_today - rate_prev) / rate_prev, 6),
                })

        if ret_rows:
            position_returns_dfs[d] = pd.DataFrame(ret_rows)

        prev_prices_eur = pos_prices_eur

    return daily_positions_dfs, position_returns_dfs


# ── Writers ───────────────────────────────────────────────────────────────────

def write_local(df: pd.DataFrame, rel_key: str) -> None:
    # Strip the S3 prefix (history/raw/ or history/derived/) so local writes
    # land in data/raw/ or data/derived/, matching the app's local path convention.
    local_key = rel_key.removeprefix(f"{RAW_PREFIX}/")
    path = ROOT / "data" / "raw" / local_key
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"  -> {path.relative_to(ROOT)}")


def write_s3(df: pd.DataFrame, rel_key: str, tmp_dir: str) -> None:
    local = pathlib.Path(tmp_dir) / rel_key.replace("/", "_")
    df.to_parquet(local, index=False, engine="pyarrow")
    subprocess.run(
        ["aws", "--region", AWS_REGION, "s3", "cp", str(local), f"s3://{S3_BUCKET}/{rel_key}"],
        check=True, capture_output=True,
    )
    print(f"  -> s3://{S3_BUCKET}/{rel_key}")


def write_all(trades, daily_positions_dfs, position_returns_dfs, instrument_metadata, write_fn) -> None:
    if trades:
        trade_dates = sorted({t["date"] for t in trades})
        print(f"\nWriting trade_log ({len(trades)} trades, {len(trade_dates)} days)...")
        for d in trade_dates:
            day_trades = [t for t in trades if t["date"] == d]
            cols = ["symbol", "description", "isin", "currency", "assetCategory", "action", "shares", "price", "proceeds", "commission"]
            df   = pd.DataFrame([{k: t[k] for k in cols if k in t} for t in day_trades])
            df   = df.rename(columns={"description": "name", "currency": "ccy"})
            write_fn(df, f"{RAW_PREFIX}/trade_log/date={d}/data.parquet")

    print(f"\nWriting {len(daily_positions_dfs)} daily_positions partitions...")
    for d, df in sorted(daily_positions_dfs.items()):
        write_fn(df, f"{RAW_PREFIX}/daily_positions/date={d}/data.parquet")

    print(f"\nWriting {len(position_returns_dfs)} position_returns partitions...")
    for d, df in sorted(position_returns_dfs.items()):
        write_fn(df, f"{RAW_PREFIX}/position_returns/date={d}/data.parquet")

    print(f"\nWriting instrument_metadata ({len(instrument_metadata)} symbols)...")
    write_fn(instrument_metadata, f"{RAW_PREFIX}/instrument_metadata.parquet")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--local",  action="store_true", help="Write to data/raw/ instead of S3")
    parser.add_argument("--debug",  action="store_true", help="Print raw Flex XML and exit")
    args = parser.parse_args()

    token    = os.environ["IBKR_FLEX_TOKEN"]
    query_id = os.environ["IBKR_QUERY_ID"]

    print(f"Fetching Flex query (id={query_id})...")
    root = fetch_flex_query(token, query_id)

    if args.debug:
        import xml.dom.minidom
        print(xml.dom.minidom.parseString(ET.tostring(root)).toprettyxml(indent="  "))
        return

    trades = parse_trades(root)

    if not trades:
        print(
            "\nNo trades found. Make sure the Trades section is enabled in your\n"
            "Flex query with Trade Date and Quantity fields included."
        )
        return

    dates = sorted({t["date"] for t in trades})
    print(f"  {len(trades)} trades across {len(dates)} days: {dates[0]} -> {dates[-1]}")

    symbols_meta: dict[str, dict] = {}
    for t in trades:
        symbols_meta[t["symbol"]] = {
            "description":   t["description"],
            "isin":          t["isin"],
            "currency":      t["currency"],
            "assetCategory": t["assetCategory"],
            "subCategory":   t["subCategory"],
        }

    print("\nLoading nav_history.json...")
    nav_by_date = load_nav_history()
    nav_dates   = sorted(nav_by_date.keys())
    print(f"  {len(nav_dates)} dates: {nav_dates[0]} -> {nav_dates[-1]}")

    date_from = min(dates[0], nav_dates[0])
    date_to   = max(dates[-1], nav_dates[-1])

    # Only fetch prices for equity positions — skip FX pair symbols (assetCategory=CASH)
    symbols_ccy = {
        sym: meta["currency"]
        for sym, meta in symbols_meta.items()
        if meta.get("assetCategory") != "CASH"
    }
    print(f"\nFetching prices for {len(symbols_ccy)} symbols + FX rates...")
    prices, fx_rates, price_currencies = fetch_prices_and_fx(symbols_ccy, date_from, date_to)

    print("\nReconstructing daily positions...")
    daily_positions_dfs, position_returns_dfs = reconstruct_daily_positions(
        trades, nav_by_date, prices, fx_rates, price_currencies, symbols_meta
    )
    print(f"  Built {len(daily_positions_dfs)} daily snapshots")

    instrument_metadata = pd.DataFrame([
        {"symbol": sym, "name": m["description"], "isin": m["isin"],
         "ccy": m["currency"], "category": _category(m["assetCategory"], m["subCategory"])}
        for sym, m in symbols_meta.items()
    ])

    if args.local:
        write_all(trades, daily_positions_dfs, position_returns_dfs, instrument_metadata, write_local)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            write_all(
                trades, daily_positions_dfs, position_returns_dfs, instrument_metadata,
                lambda df, key: write_s3(df, key, tmp),
            )

    print("\nDone. Next steps:")
    print("  python -m precompute.build_derived")
    if not args.local:
        print("  aws s3 sync data/derived/ s3://aic-fund-public-data/history/derived/ --region eu-central-1")


if __name__ == "__main__":
    main()
