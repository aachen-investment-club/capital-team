"""
LSEG EOD OHLCV + fundamentals ingestion → local DuckDB store.

Rewrite of the `fund-eod-ingestion` Lambda: same LSEG fields and normalisation,
but upserts into market.duckdb instead of writing per-security parquet to S3.

Universe = config/security_master.csv (local checkout first, S3 copy as
fallback). EOD is fetched for all active non-INDEX securities; fundamentals
for every active RIC that isn't a pure index (leading dot).
"""
import io
import logging
import time
from datetime import date, timedelta

import pandas as pd

from capital.data import store
from capital.settings import settings

log = logging.getLogger(__name__)

_OHLCV_FIELDS = ["TR.OPENPRICE", "TR.HIGHPRICE", "TR.LOWPRICE", "TR.CLOSEPRICE", "TR.VOLUME"]
_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
_EOD_BATCH = 50   # RICs per get_history request
_FUND_BATCH = 10

_FUND_FIELDS = ["TR.PriceToBVPerShare", "TR.CompanyMarketCap", "TR.SharesOutstanding"]
_FUND_COL_MAP = {
    "pricetobvpershare": "pb_ratio",
    "companymarketcap": "market_cap",
    "sharesoutstanding": "shares_outstanding",
    "price to book value per share": "pb_ratio",
    "company market cap": "market_cap",
    "shares outstanding": "shares_outstanding",
}


# ── LSEG session ──────────────────────────────────────────────────────────────

def open_lseg_session() -> None:
    """Open and set the default LSEG session.

    LSEG_SESSION_TYPE selects how we connect:
      - "platform" (default): RDP GrantPassword — headless, for the server.
      - "desktop": attaches to a running LSEG Workspace/Eikon on this machine
        (app key only, no username/password). Use for local ad-hoc ingests and
        testing. The get_history response shape is identical either way.
    """
    import lseg.data as ld
    if settings.lseg_session_type == "desktop":
        from lseg.data.session import desktop
        sess = desktop.Definition(app_key=settings.lseg_app_key).get_session()
    else:
        from lseg.data.session import platform as lseg_platform
        sess = lseg_platform.Definition(
            app_key=settings.lseg_app_key,
            grant=lseg_platform.GrantPassword(
                username=settings.lseg_username,
                password=settings.lseg_password,
            ),
            signon_control=True,
        ).get_session()
    sess.open()
    ld.session.set_default(sess)
    log.info("LSEG %s session opened", settings.lseg_session_type)


def close_lseg_session() -> None:
    import lseg.data as ld
    try:
        ld.close_session()
    except Exception:
        pass


# ── Universe ──────────────────────────────────────────────────────────────────

def load_universe() -> pd.DataFrame:
    """Active rows of security_master.csv (local checkout, else the S3 copy)."""
    csv_path = settings.config_dir / "security_master.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, comment="#", dtype=str)
    elif settings.s3_bucket:
        import boto3
        s3c = boto3.client("s3", region_name=settings.aws_region)
        obj = s3c.get_object(Bucket=settings.s3_bucket, Key="config/security_master.csv")
        df = pd.read_csv(io.BytesIO(obj["Body"].read()), comment="#", dtype=str)
    else:
        raise FileNotFoundError("security_master.csv not found locally and S3_BUCKET unset")
    df["active"] = df["active"].str.lower().isin(("true", "1", "yes"))
    return df[df["active"]].reset_index(drop=True)


def _window(start: str | None, days: int | None) -> tuple[str, str]:
    end = date.today().isoformat()
    if start:
        return start, end
    lookback = days if days is not None else settings.eod_lookback_days
    return (date.today() - timedelta(days=lookback)).isoformat(), end


# ── EOD prices ────────────────────────────────────────────────────────────────

def _fetch_history(rics: list[str], fields: list[str], start: str, end: str) -> pd.DataFrame:
    import lseg.data as ld
    for attempt in range(1, 4):
        try:
            raw = ld.get_history(universe=rics, fields=fields, start=start, end=end, interval="1D")
            return raw if raw is not None else pd.DataFrame()
        except Exception as exc:
            if attempt == 3:
                raise
            wait = 10 * attempt
            log.warning("get_history attempt %d/3 failed: %s — retrying in %ds", attempt, exc, wait)
            time.sleep(wait)


def _normalise_eod(df: pd.DataFrame, ric: str, ric_to_id: dict) -> pd.DataFrame | None:
    """Same normalisation as the Lambda: OHLCV columns, adj_close = close."""
    sec_id = ric_to_id.get(str(ric))
    if not sec_id or df.empty:
        return None
    df = df.copy()
    df.columns = _OHLCV_COLS
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["security_id"] = sec_id
    df["adj_close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        return None
    return df[["security_id", "date", "open", "high", "low", "close", "adj_close", "volume"]]


def _split_response(raw: pd.DataFrame, batch: list[str]) -> dict[str, pd.DataFrame]:
    """LSEG returns MultiIndex columns (multi-RIC) or a flat frame (single RIC)."""
    per_ric: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for ric in raw.columns.get_level_values(0).unique():
            per_ric[str(ric)] = raw[ric]
    elif isinstance(raw.index, pd.MultiIndex):
        for ric in raw.index.get_level_values(0).unique():
            per_ric[str(ric)] = raw.xs(ric, level=0)
    elif len(batch) == 1:
        per_ric[batch[0]] = raw
    return per_ric


def run_eod(start: str | None = None, days: int | None = None,
            universe: pd.DataFrame | None = None) -> dict:
    """Fetch EOD OHLCV for the active non-INDEX universe and upsert the store.

    `start` (ISO date) enables history backfill for newly added securities;
    default window is the configured lookback.
    """
    master = universe if universe is not None else load_universe()
    eod_master = master[master["asset_type"] != "INDEX"]
    ric_to_id = dict(zip(eod_master["ric"], eod_master["security_id"]))
    rics = eod_master["ric"].tolist()
    lo, hi = _window(start, days)
    log.info("[EOD] %d RICs  %s → %s", len(rics), lo, hi)

    succeeded, failed = [], []
    con = store.write_connection()
    try:
        for i in range(0, len(rics), _EOD_BATCH):
            batch = rics[i:i + _EOD_BATCH]
            log.info("[EOD] batch %d/%d (%d RICs)",
                     i // _EOD_BATCH + 1, -(-len(rics) // _EOD_BATCH), len(batch))
            try:
                raw = _fetch_history(batch, _OHLCV_FIELDS, lo, hi)
            except Exception as exc:
                log.error("[EOD] batch failed: %s", exc)
                failed.extend(batch)
                continue
            per_ric = _split_response(raw, batch)
            for ric in batch:
                norm = _normalise_eod(per_ric.get(ric, pd.DataFrame()), ric, ric_to_id)
                if norm is None:
                    failed.append(ric)
                    continue
                store.upsert(con, "eod_prices", norm)
                succeeded.append(ric)
        if succeeded:
            store.bump_data_version(con)
    finally:
        con.close()

    log.info("[EOD] done: succeeded=%d failed=%d %s", len(succeeded), len(failed), failed or "")
    return {"succeeded": len(succeeded), "failed": len(failed), "failed_rics": failed}


# ── Fundamentals ──────────────────────────────────────────────────────────────

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
        sector_col = next((c for c in df.columns if "gics" in c.lower() or "sector" in c.lower()), None)
        inst_col = next((c for c in df.columns if "instrument" in c.lower()), df.columns[0])
        if sector_col is None:
            return {}
        return dict(zip(df[inst_col].str.upper(), df[sector_col].fillna("")))
    except Exception as exc:
        log.warning("[FUND] GICS fetch failed: %s", exc)
        return {}


def _to_float(val) -> float:
    if val is None:
        return float("nan")
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def run_fundamentals(start: str | None = None, days: int | None = None,
                     universe: pd.DataFrame | None = None) -> dict:
    """Fetch P/B, market cap, shares outstanding, GICS for the active universe
    (excluding pure indices) and upsert into the fundamentals table."""
    master = universe if universe is not None else load_universe()
    fund = master[~master["ric"].str.startswith(".")]
    ric_to_ticker = dict(zip(fund["ric"], fund["ticker"]))
    rics = fund["ric"].tolist()
    lo, hi = _window(start, days)
    log.info("[FUND] %d RICs  %s → %s", len(rics), lo, hi)

    gics_map = _fetch_gics(rics)

    rows = []
    for i in range(0, len(rics), _FUND_BATCH):
        batch = rics[i:i + _FUND_BATCH]
        log.info("[FUND] batch %d/%d", i // _FUND_BATCH + 1, -(-len(rics) // _FUND_BATCH))
        try:
            raw = _fetch_history(batch, _FUND_FIELDS, lo, hi)
        except Exception as exc:
            log.warning("[FUND] batch failed: %s", exc)
            continue
        if raw.empty:
            continue
        for ric, sub in _split_response(raw, batch).items():
            sub = _normalise_fund_cols(sub.copy())
            sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
            sub["date"] = pd.to_datetime(sub["date"]).dt.date
            for _, row in sub.iterrows():
                rows.append({
                    "ric": ric,
                    "date": row.get("date"),
                    "ticker": ric_to_ticker.get(ric, ric),
                    "gics_sector": gics_map.get(ric.upper(), ""),
                    "pb_ratio": _to_float(row.get("pb_ratio")),
                    "market_cap": _to_float(row.get("market_cap")),
                    "shares_outstanding": _to_float(row.get("shares_outstanding")),
                })

    if not rows:
        log.warning("[FUND] no fundamentals returned")
        return {"rows": 0}

    new_df = pd.DataFrame(rows).dropna(subset=["date"])
    new_df = new_df.drop_duplicates(subset=["ric", "date"], keep="last")
    con = store.write_connection()
    try:
        n = store.upsert(con, "fundamentals", new_df)
        store.bump_data_version(con)
    finally:
        con.close()
    log.info("[FUND] done: %d rows, %d RICs", n, new_df["ric"].nunique())
    return {"rows": n, "rics": int(new_df["ric"].nunique())}
