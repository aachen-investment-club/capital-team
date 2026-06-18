"""
Nightly data ingestion Lambda — runs after market close.

Fetches in a single LSEG session:
  1. EOD OHLCV prices  → s3://{S3_BUCKET}/history/eod_prices/security_id=*/data.parquet
  2. Fundamentals      → s3://{S3_BUCKET}/history/fundamentals/data.parquet
     (P/B, market cap, shares outstanding, GICS sector)
     Universe = security_master.csv UNION barra_universe.csv

Trigger: EventBridge Scheduler — daily after market close (e.g. 22:00 CET).

Environment variables (set in Lambda console):
  S3_BUCKET       — target bucket (e.g. aic-fund-public-data)
  AWS_REGION      — default eu-central-1
  LSEG_APP_KEY    — LSEG Data Platform application key
  LSEG_USERNAME   — RDP username (email)
  LSEG_PASSWORD   — RDP password
  LOOKBACK_DAYS   — days to fetch for EOD + fundamentals; default 5
"""
import io
import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

S3_BUCKET     = os.environ["S3_BUCKET"]
AWS_REGION    = os.environ.get("AWS_REGION", "eu-central-1")
EOD_PREFIX    = "history/eod_prices"
FUND_KEY      = "history/fundamentals/data.parquet"
MASTER_KEY    = "config/security_master.csv"
BARRA_KEY     = "config/barra_universe.csv"
LOOKBACK_DAYS = int(os.environ.get("LOOKBACK_DAYS", "5"))

# External market tickers stored under history/market_data/{ticker}.parquet
MARKET_TICKERS = ["SPY", "IWM", "TLT", "HYG", "QQQ", "GLD"]   # yfinance symbols
VIX_TICKER     = "^VIX"                                          # stored as VIX
FRED_SERIES    = {"BAMLH0A0HYM2": "HY_OAS"}                     # {fred_id: friendly_name}
MARKET_PREFIX  = "history/market_data"

s3 = boto3.client("s3", region_name=AWS_REGION)

# ── Schemas ───────────────────────────────────────────────────────────────────

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

_FUND_SCHEMA = pa.schema([
    pa.field("date",               pa.date32()),
    pa.field("ric",                pa.string()),
    pa.field("ticker",             pa.string()),
    pa.field("gics_sector",        pa.string()),
    pa.field("pb_ratio",           pa.float64()),
    pa.field("market_cap",         pa.float64()),
    pa.field("shares_outstanding", pa.float64()),
])


# ── LSEG session ──────────────────────────────────────────────────────────────

def _open_lseg_session() -> None:
    import lseg.data as ld
    from lseg.data.session import platform as lseg_platform
    sess = lseg_platform.Definition(
        app_key=os.environ["LSEG_APP_KEY"],
        grant=lseg_platform.GrantPassword(
            username=os.environ["LSEG_USERNAME"],
            password=os.environ["LSEG_PASSWORD"],
        ),
        signon_control=True,
    ).get_session()
    sess.open()
    ld.session.set_default(sess)
    log.info("LSEG session opened")


def _close_lseg_session() -> None:
    import lseg.data as ld
    try:
        ld.close_session()
    except Exception:
        pass


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _s3_read_csv(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()), comment="#", dtype=str)


def _s3_read_parquet(key: str) -> pd.DataFrame | None:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def _s3_write_parquet(df: pd.DataFrame, key: str, schema: pa.Schema) -> None:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False),
                   buf, compression="snappy")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


# ── Universe loading ──────────────────────────────────────────────────────────

def _load_universe() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Load security_master + barra_universe from S3.

    Returns (master_df, ric_to_id, ric_to_ticker).
    master_df has all active non-INDEX securities.
    ric_to_ticker covers both files (deduplicated by RIC).
    """
    master = _s3_read_csv(MASTER_KEY)
    master["active"] = master["active"].str.lower().isin(("true", "1", "yes"))
    master = master[master["active"]].reset_index(drop=True)

    ric_to_id     = dict(zip(master["ric"], master["security_id"]))
    ric_to_ticker = dict(zip(master["ric"], master["ticker"]))

    # Barra universe — extra RICs for fundamentals only (no EOD security_id)
    try:
        barra = _s3_read_csv(BARRA_KEY)
        for _, row in barra.iterrows():
            ric = str(row["ric"])
            if ric not in ric_to_ticker:
                ric_to_ticker[ric] = str(row.get("ticker", ric))
    except ClientError:
        log.warning("barra_universe.csv not found on S3 — fundamentals universe = security_master only")

    return master, ric_to_id, ric_to_ticker


# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — EOD PRICES
# ═════════════════════════════════════════════════════════════════════════════

_OHLCV_FIELDS = ["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.CLOSEPRICE", "TR.VOLUME"]
_OHLCV_COLS   = ["open", "high", "low", "close", "volume"]


def _fetch_eod(rics: list[str], start: str, end: str) -> pd.DataFrame:
    import lseg.data as ld
    for attempt in range(1, 4):
        try:
            raw = ld.get_history(universe=rics, fields=_OHLCV_FIELDS,
                                 start=start, end=end, interval="1D")
            return raw if raw is not None else pd.DataFrame()
        except Exception as exc:
            if attempt == 3:
                raise
            wait = 10 * attempt
            log.warning("EOD fetch attempt %d/3 failed: %s — retrying in %ds", attempt, exc, wait)
            time.sleep(wait)


def _normalise_eod(df: pd.DataFrame, ric: str, ric_to_id: dict) -> tuple | None:
    sec_id = ric_to_id.get(str(ric))
    if not sec_id or df.empty:
        return None
    df = df.copy()
    df.columns = _OHLCV_COLS
    df.index.name = "date"
    df = df.reset_index()
    df["date"]        = pd.to_datetime(df["date"]).dt.date
    df["security_id"] = sec_id
    df["adj_close"]   = pd.to_numeric(df["close"], errors="coerce")
    df["volume"]      = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return (sec_id, df[["security_id", "date", "open", "high",
                          "low", "close", "adj_close", "volume"]]) if not df.empty else None


def _ingest_eod(master: pd.DataFrame, ric_to_id: dict, start: str, end: str) -> dict:
    portfolio_rics = master[master["asset_type"] != "INDEX"]["ric"].tolist()
    if not portfolio_rics:
        return {"succeeded": 0, "skipped": 0, "failed": 0}

    log.info("[EOD] Fetching %d RICs %s → %s", len(portfolio_rics), start, end)
    raw = _fetch_eod(portfolio_rics, start, end)

    if raw.empty:
        log.warning("[EOD] LSEG returned no data")
        return {"succeeded": 0, "skipped": len(portfolio_rics), "failed": 0}

    # Reshape MultiIndex or flat response
    per_sec: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ric in raw.columns.get_level_values(0).unique():
            out = _normalise_eod(raw[ric], ric, ric_to_id)
            if out:
                per_sec[out[0]] = out[1]
    elif isinstance(raw.index, pd.MultiIndex):
        for ric in raw.index.get_level_values(0).unique():
            out = _normalise_eod(raw.xs(ric, level=0), ric, ric_to_id)
            if out:
                per_sec[out[0]] = out[1]

    succeeded, skipped, failed = [], [], []
    max_dates: list[str] = []

    for sec_id, new_df in per_sec.items():
        try:
            existing = _s3_read_parquet(f"{EOD_PREFIX}/security_id={sec_id}/data.parquet")
            if existing is not None and not existing.empty:
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["security_id", "date"], keep="last")
            else:
                combined = new_df
            combined = combined.sort_values("date").reset_index(drop=True)
            _s3_write_parquet(combined, f"{EOD_PREFIX}/security_id={sec_id}/data.parquet", _EOD_SCHEMA)
            max_dates.append(str(combined["date"].max()))
            succeeded.append(sec_id)
            log.info("  [EOD] %-8s  %d rows  max=%s", sec_id, len(combined), max_dates[-1])
        except Exception as exc:
            log.error("  [EOD] %-8s  FAILED: %s", sec_id, exc, exc_info=True)
            failed.append(sec_id)

    if max_dates:
        payload = json.dumps({
            "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "max_date":    max(max_dates),
        })
        s3.put_object(Bucket=S3_BUCKET, Key=f"{EOD_PREFIX}/_version.json",
                      Body=payload, ContentType="application/json")
        log.info("[EOD] _version.json → max_date=%s", max(max_dates))

    log.info("[EOD] Done: succeeded=%d skipped=%d failed=%d",
             len(succeeded), len(skipped), len(failed))
    return {"succeeded": len(succeeded), "skipped": len(skipped), "failed": len(failed)}


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — FUNDAMENTALS
# ═════════════════════════════════════════════════════════════════════════════

_FUND_FIELDS = ["TR.PriceToBVPerShare", "TR.CompanyMarketCap", "TR.SharesOutstanding"]
_FUND_COL_MAP = {
    "pricetobvpershare": "pb_ratio",
    "companymarketcap":  "market_cap",
    "sharesoutstanding": "shares_outstanding",
    # verbose names LSEG sometimes returns
    "price to book value per share": "pb_ratio",
    "company market cap":            "market_cap",
    "shares outstanding":            "shares_outstanding",
}


def _normalise_fund_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = col.lower().replace(" ", "").replace("_", "")
        for pat, canonical in _FUND_COL_MAP.items():
            if pat.replace(" ", "") in key:
                rename[col] = canonical
                break
    return df.rename(columns=rename)


def _fetch_gics(rics: list[str]) -> dict[str, str]:
    import lseg.data as ld
    try:
        df = ld.get_data(universe=rics, fields=["TR.GICSSector"])
        if df is None or df.empty:
            return {}
        sector_col = next((c for c in df.columns
                           if "gics" in c.lower() or "sector" in c.lower()), None)
        inst_col   = next((c for c in df.columns
                           if "instrument" in c.lower()), df.columns[0])
        if sector_col is None:
            return {}
        return dict(zip(df[inst_col].str.upper(), df[sector_col].fillna("")))
    except Exception as exc:
        log.warning("[FUND] GICS fetch failed: %s", exc)
        return {}


def _fetch_fund_history(rics: list[str], start: str, end: str,
                        batch_size: int = 10) -> dict[str, pd.DataFrame]:
    import lseg.data as ld
    result: dict[str, pd.DataFrame] = {}
    batches = [rics[i:i + batch_size] for i in range(0, len(rics), batch_size)]

    for idx, batch in enumerate(batches, 1):
        log.info("  [FUND] batch %d/%d (%d RICs)", idx, len(batches), len(batch))
        try:
            raw = ld.get_history(universe=batch, fields=_FUND_FIELDS,
                                 start=start, end=end, interval="1D")
            if raw is None or raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for ric in raw.columns.get_level_values(0).unique():
                    sub = _normalise_fund_cols(raw[ric].copy())
                    sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
                    sub["date"] = pd.to_datetime(sub["date"]).dt.date
                    result[str(ric)] = sub
            else:
                sub = _normalise_fund_cols(raw.copy())
                sub = sub.reset_index().rename(columns={"Date": "date"})
                sub["date"] = pd.to_datetime(sub["date"]).dt.date
                result[str(batch[0])] = sub
        except Exception as exc:
            log.warning("  [FUND] batch %d failed: %s", idx, exc)

    return result


def _ingest_fundamentals(ric_to_ticker: dict, start: str, end: str) -> dict:
    all_rics = list(ric_to_ticker.keys())
    # Exclude pure benchmark indices (no fundamentals)
    all_rics = [r for r in all_rics if not r.startswith(".")]
    log.info("[FUND] Fetching fundamentals for %d RICs %s → %s", len(all_rics), start, end)

    gics_map  = _fetch_gics(all_rics)
    hist_data = _fetch_fund_history(all_rics, start, end)

    if not hist_data:
        log.warning("[FUND] No fundamentals data returned")
        return {"rics_updated": 0}

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

    existing = _s3_read_parquet(FUND_KEY)
    if existing is not None and not existing.empty:
        existing["date"] = pd.to_datetime(existing["date"]).dt.date
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ric"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(["ric", "date"]).reset_index(drop=True)

    # Ensure all schema columns exist
    for col in ("pb_ratio", "market_cap", "shares_outstanding"):
        if col not in combined.columns:
            combined[col] = float("nan")
    for col in ("gics_sector", "ric", "ticker"):
        if col not in combined.columns:
            combined[col] = ""

    _s3_write_parquet(combined, FUND_KEY, _FUND_SCHEMA)
    log.info("[FUND] Done — %d total rows, %d unique RICs",
             len(combined), combined["ric"].nunique())
    return {"rics_updated": new_df["ric"].nunique()}


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — EXTERNAL MARKET DATA (yfinance + FRED)
# ═════════════════════════════════════════════════════════════════════════════

_MARKET_SCHEMA = pa.schema([
    pa.field("date",   pa.date32()),
    pa.field("close",  pa.float64()),
    pa.field("volume", pa.float64()),
])

_FRED_SCHEMA = pa.schema([
    pa.field("date",  pa.date32()),
    pa.field("value", pa.float64()),
])


def _ingest_market_data(start: str, end: str) -> dict:
    import yfinance as yf

    succeeded, failed = [], []

    # All equity/ETF tickers + VIX in one download
    all_tickers = MARKET_TICKERS + [VIX_TICKER]
    log.info("[MARKET] Downloading %d tickers via yfinance %s → %s",
             len(all_tickers), start, end)
    try:
        raw = yf.download(all_tickers, start=start, end=end,
                          progress=False, auto_adjust=True)
        close  = raw["Close"]  if "Close"  in raw.columns else pd.DataFrame()
        volume = raw.get("Volume", pd.DataFrame())

        for yf_ticker in all_tickers:
            store_name = "VIX" if yf_ticker == VIX_TICKER else yf_ticker
            try:
                col = yf_ticker if yf_ticker in close.columns else None
                if col is None:
                    failed.append(store_name)
                    continue

                s_close  = close[col].dropna()
                s_volume = volume[col].dropna() if col in volume.columns else pd.Series(dtype=float)

                df = pd.DataFrame({
                    "date":   s_close.index.normalize().date,
                    "close":  s_close.values.astype(float),
                    "volume": s_volume.reindex(s_close.index).fillna(0).values.astype(float),
                })
                df["date"] = pd.to_datetime(df["date"]).dt.date

                key = f"{MARKET_PREFIX}/{store_name}.parquet"
                existing = _s3_read_parquet(key)
                if existing is not None and not existing.empty:
                    existing["date"] = pd.to_datetime(existing["date"]).dt.date
                    df = pd.concat([existing, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["date"], keep="last")
                df = df.sort_values("date").reset_index(drop=True)
                _s3_write_parquet(df, key, _MARKET_SCHEMA)
                succeeded.append(store_name)
                log.info("  [MARKET] %-6s  %d rows  max=%s", store_name, len(df), df["date"].max())
            except Exception as exc:
                log.error("  [MARKET] %-6s  FAILED: %s", store_name, exc)
                failed.append(store_name)

    except Exception as exc:
        log.error("[MARKET] yfinance download failed: %s", exc)
        return {"succeeded": 0, "failed": len(all_tickers)}

    log.info("[MARKET] Done: succeeded=%d failed=%d", len(succeeded), len(failed))
    return {"succeeded": len(succeeded), "failed": len(failed)}


def _ingest_fred(start: str, end: str) -> dict:
    import urllib.request as _url
    import io as _io

    succeeded, failed = [], []

    for series_id, name in FRED_SERIES.items():
        try:
            log.info("[FRED] Fetching %s (%s)", series_id, name)
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            with _url.urlopen(url, timeout=30) as r:
                raw = pd.read_csv(_io.BytesIO(r.read()))

            raw.columns = ["date", "value"]
            raw["date"]  = pd.to_datetime(raw["date"]).dt.date
            raw["value"] = pd.to_numeric(raw["value"].replace(".", float("nan")),
                                         errors="coerce")
            raw = raw.dropna(subset=["value"])
            raw = raw[raw["date"] >= pd.to_datetime(start).date()]
            raw = raw.sort_values("date").reset_index(drop=True)

            key = f"{MARKET_PREFIX}/FRED_{series_id}.parquet"
            existing = _s3_read_parquet(key)
            if existing is not None and not existing.empty:
                existing["date"] = pd.to_datetime(existing["date"]).dt.date
                raw = pd.concat([existing, raw], ignore_index=True)
                raw = raw.drop_duplicates(subset=["date"], keep="last")
            raw = raw.sort_values("date").reset_index(drop=True)
            _s3_write_parquet(raw, key, _FRED_SCHEMA)
            succeeded.append(series_id)
            log.info("  [FRED] %-20s  %d rows  max=%s", series_id, len(raw), raw["date"].max())
        except Exception as exc:
            log.error("  [FRED] %-20s  FAILED: %s", series_id, exc)
            failed.append(series_id)

    return {"succeeded": len(succeeded), "failed": len(failed)}


def _to_float(val) -> float:
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# HANDLER
# ═════════════════════════════════════════════════════════════════════════════

def lambda_handler(event, context):
    master, ric_to_id, ric_to_ticker = _load_universe()

    if master.empty:
        log.warning("Security master is empty — nothing to fetch")
        return {"status": "skipped", "reason": "empty universe"}

    end   = date.today().isoformat()
    start = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()

    # Market data + FRED run first (no LSEG session needed)
    market_result = _ingest_market_data(start, end)
    fred_result   = _ingest_fred(start, end)

    _open_lseg_session()
    try:
        eod_result  = _ingest_eod(master, ric_to_id, start, end)
        fund_result = _ingest_fundamentals(ric_to_ticker, start, end)
    finally:
        _close_lseg_session()

    summary = {
        "eod":          eod_result,
        "fundamentals": fund_result,
        "market_data":  market_result,
        "fred":         fred_result,
        "ingested_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    log.info("All done: %s", summary)

    if eod_result.get("failed", 0) > 0:
        raise RuntimeError(f"EOD failures: {eod_result['failed']} securities")

    return summary
