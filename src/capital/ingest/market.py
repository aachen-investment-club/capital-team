"""
External market data ingestion → local DuckDB store.

yfinance ETF/index tickers + FRED series (same set as the old EOD Lambda's
part 3, which never made it to S3 — the store is authoritative from now on).
Independent of LSEG, so an LSEG outage doesn't kill market data.
"""
import io
import logging
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd

from capital.data import store
from capital.settings import settings

log = logging.getLogger(__name__)

MARKET_TICKERS = ["SPY", "IWM", "TLT", "HYG", "QQQ", "GLD"]
VIX_TICKER = "^VIX"                       # stored as VIX
FRED_SERIES = {"BAMLH0A0HYM2": "HY_OAS"}  # {fred_id: friendly_name}

# EUR crosses, stored in market_data as FX_<BASE><QUOTE> (quote units per EUR).
# The factor model uses them to translate a multi-currency universe into one
# numeraire; without them it estimates in local currency and says so in the run's
# coverage report (see data.loaders.get_fx_rates).
FX_PAIRS = {
    "EURUSD=X": "FX_EURUSD", "EURGBP=X": "FX_EURGBP", "EURCHF=X": "FX_EURCHF",
    "EURSEK=X": "FX_EURSEK", "EURDKK=X": "FX_EURDKK", "EURNOK=X": "FX_EURNOK",
    "EURJPY=X": "FX_EURJPY", "EURPLN=X": "FX_EURPLN",
}


def run_market(days: int | None = None) -> dict:
    """Download OHLCV for the market tickers and upsert into market_data."""
    import yfinance as yf

    lookback = days if days is not None else max(settings.eod_lookback_days, 5)
    start = (date.today() - timedelta(days=lookback)).isoformat()
    all_tickers = MARKET_TICKERS + [VIX_TICKER] + list(FX_PAIRS)
    log.info("[MARKET] %d tickers via yfinance from %s", len(all_tickers), start)

    try:
        raw = yf.download(all_tickers, start=start, progress=False,
                          auto_adjust=True, group_by="ticker")
    except Exception as exc:
        log.error("[MARKET] yfinance download failed: %s", exc)
        return {"succeeded": 0, "failed": len(all_tickers)}

    succeeded, failed = [], []
    con = store.write_connection()
    try:
        for yf_ticker in all_tickers:
            name = "VIX" if yf_ticker == VIX_TICKER else FX_PAIRS.get(yf_ticker, yf_ticker)
            try:
                sub = raw[yf_ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
                sub = sub.dropna(subset=["Close"])
                if sub.empty:
                    failed.append(name)
                    continue
                df = pd.DataFrame({
                    "ticker": name,
                    "date": pd.to_datetime(sub.index).date,
                    "open": pd.to_numeric(sub.get("Open"), errors="coerce"),
                    "high": pd.to_numeric(sub.get("High"), errors="coerce"),
                    "low": pd.to_numeric(sub.get("Low"), errors="coerce"),
                    "close": pd.to_numeric(sub["Close"], errors="coerce"),
                    "volume": pd.to_numeric(sub.get("Volume"), errors="coerce")
                              .fillna(0).astype("int64"),
                })
                store.upsert(con, "market_data", df)
                succeeded.append(name)
                log.info("[MARKET] %-6s %d rows  max=%s", name, len(df), df["date"].max())
            except Exception as exc:
                log.error("[MARKET] %-6s FAILED: %s", name, exc)
                failed.append(name)
        if succeeded:
            store.bump_data_version(con)
    finally:
        con.close()

    log.info("[MARKET] done: succeeded=%d failed=%d", len(succeeded), len(failed))
    return {"succeeded": len(succeeded), "failed": len(failed)}


def _fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Full history via the official API when FRED_API_KEY is set; otherwise the
    anonymous fredgraph CSV (which FRED caps at ~3 years)."""
    if settings.fred_api_key:
        import json
        url = ("https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={settings.fred_api_key}"
               "&file_type=json&observation_start=2010-01-01")
        with urllib.request.urlopen(url, timeout=30) as r:
            obs = json.loads(r.read())["observations"]
        raw = pd.DataFrame(obs)[["date", "value"]]
    else:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        with urllib.request.urlopen(url, timeout=30) as r:
            raw = pd.read_csv(io.BytesIO(r.read()))
        raw.columns = ["date", "value"]
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    raw["value"] = pd.to_numeric(raw["value"].replace(".", np.nan), errors="coerce")
    return raw.dropna(subset=["value"])


def run_fred() -> dict:
    """Fetch each FRED series (full history — it's tiny) and upsert."""
    succeeded, failed = [], []
    con = store.write_connection()
    try:
        for series_id, name in FRED_SERIES.items():
            try:
                raw = _fetch_fred_series(series_id)
                raw.insert(0, "series_id", series_id)
                store.upsert(con, "fred", raw)
                succeeded.append(series_id)
                log.info("[FRED] %-16s (%s) %d rows  max=%s",
                         series_id, name, len(raw), raw["date"].max())
            except Exception as exc:
                log.error("[FRED] %-16s FAILED: %s", series_id, exc)
                failed.append(series_id)
        if succeeded:
            store.bump_data_version(con)
    finally:
        con.close()
    return {"succeeded": len(succeeded), "failed": len(failed)}
