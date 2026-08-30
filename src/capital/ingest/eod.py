"""
LSEG EOD OHLCV + fundamentals ingestion → local DuckDB store.

Rewrite of the `fund-eod-ingestion` Lambda: same LSEG fields and normalisation,
but upserts into market.duckdb instead of writing per-security parquet to S3.

Universe = config/security_master.csv (local checkout first, S3 copy as
fallback). EOD is fetched for all active non-INDEX securities; fundamentals
for every active RIC that isn't a pure index (leading dot).
"""
import io
import json
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

# ── Fundamental descriptor fields ─────────────────────────────────────────────
#
# The TR field name that delivers a given quantity depends on the account's
# entitlements and LSEG occasionally renames them, so each store column carries a
# *list of candidates in priority order* rather than one hard-coded name. The
# first candidate that returns data wins, and the resolution is cached in the
# store (meta.lseg_fields) so the nightly run does not re-probe every night.
#
# The defaults below were resolved against this project's own credential with
# `capital-ingest fund --probe`; re-run it if the entitlement changes.
#
# Two names are not what a reader might assume, and the labels say so:
#   roe          -> TR.ROEMean is the analyst *consensus estimate*, not realised
#                   return on equity. The only ROE field entitled here.
#   debt_to_ev   -> debt over enterprise value, not over assets. Total-debt-to-
#                   assets is not entitled; EV gearing is the available proxy.

FUND_CANDIDATES: dict[str, list[str]] = {
    "pb_ratio":           ["TR.PriceToBVPerShare"],
    "market_cap":         ["TR.CompanyMarketCap"],
    "shares_outstanding": ["TR.SharesOutstanding", "TR.CommonSharesOutstanding"],
    "pe_ratio":           ["TR.PE", "TR.PriceToEPSPerShare"],
    "ps_ratio":           ["TR.PriceToSalesPerShare"],
    "pcf_ratio":          ["TR.PriceToCFPerShare"],
    "dividend_yield":     ["TR.DividendYield"],
    "roe":                ["TR.ROEMean", "TR.ROETotalEquityPercent", "TR.ROE"],
    "roa":                ["TR.ROATotalAssetsPercent", "TR.ReturnOnAvgTotAssetsPct"],
    "gross_margin":       ["TR.GrossMargin", "TR.GrossMarginPercent"],
    "debt_to_ev":         ["TR.TotalDebtToEV", "TR.TotalDebtPctofTotalAssets"],
    "revenue_ttm":        ["TR.Revenue", "TR.TotalRevenue"],
    "eps_ttm":            ["TR.BasicEPSInclExtraItems", "TR.EPSInclExtraItems"],
}

#: Columns that must resolve or the ingest is pointless.
FUND_REQUIRED = ("pb_ratio", "market_cap")

#: Store columns the fundamentals table carries, in write order.
FUND_NUMERIC_COLS = list(FUND_CANDIDATES)

_FUND_FIELDS_PER_REQUEST = 4   # LSEG is happier with a few fields at a time
_META_FIELDS_KEY = "lseg_fields"


def _norm_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


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

_last_request_at = 0.0


def _throttle() -> None:
    """Keep a floor between requests so a long backfill cannot look like abuse.

    A full-history run is thousands of consecutive calls on one credential; a
    small enforced gap costs minutes over the whole run and is the difference
    between a steady job and a throttled one.
    """
    global _last_request_at
    gap = settings.lseg_min_request_interval - (time.monotonic() - _last_request_at)
    if gap > 0:
        time.sleep(gap)
    _last_request_at = time.monotonic()


def _get_history_once(rics: list[str], fields: list[str], start: str, end: str) -> pd.DataFrame:
    import lseg.data as ld
    _throttle()
    raw = ld.get_history(universe=rics, fields=fields, start=start, end=end, interval="1D")
    return raw if raw is not None else pd.DataFrame()


def _fetch_history(rics: list[str], fields: list[str], start: str, end: str,
                   _depth: int = 0) -> pd.DataFrame:
    """Fetch with retry, exponential backoff, and adaptive batch splitting.

    Three failure modes have to be told apart on a long backfill:

    - **Transient** (timeout, connection reset, throttling): retry the same
      request after a growing backoff. Most failures are this.
    - **Too big**: a batch of 50 RICs over 10 years can exceed what the server
      will assemble in one response. Halving the batch and recursing turns one
      fatal error into two smaller successes.
    - **One poisoned RIC**: a single delisted or malformed instrument can fail
      the whole batch. Splitting isolates it down to a single-RIC request, which
      fails alone and is reported, leaving its 49 neighbours intact.

    Splitting handles the last two without needing to distinguish them.
    """
    attempts = settings.lseg_retry_attempts
    for attempt in range(1, attempts + 1):
        try:
            return _get_history_once(rics, fields, start, end)
        except Exception as exc:                                      # noqa: BLE001
            transient = attempt < attempts
            if transient:
                wait = min(settings.lseg_backoff_seconds * 2 ** (attempt - 1), 120)
                log.warning("get_history attempt %d/%d for %d RICs failed (%s); "
                            "retrying in %ds", attempt, attempts, len(rics),
                            str(exc)[:120], wait)
                time.sleep(wait)
                continue
            # Out of retries. Split and recurse before giving up.
            if len(rics) > 1 and _depth < 6:
                mid = len(rics) // 2
                log.warning("splitting a failing batch of %d RICs into %d + %d",
                            len(rics), mid, len(rics) - mid)
                left = _fetch_history(rics[:mid], fields, start, end, _depth + 1)
                right = _fetch_history(rics[mid:], fields, start, end, _depth + 1)
                if left.empty:
                    return right
                if right.empty:
                    return left
                return pd.concat([left, right], axis=1)
            log.error("giving up on %s: %s", rics, str(exc)[:200])
            return pd.DataFrame()


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


def _progress(done: int, total: int, started: float, label: str) -> None:
    """Progress with an ETA — a multi-hour backfill needs to say how long is left."""
    if not done:
        return
    elapsed = time.time() - started
    remaining = elapsed / done * (total - done)
    log.info("[%s] %d/%d (%.0f%%)  elapsed %s  eta %s",
             label, done, total, 100 * done / total,
             _hms(elapsed), _hms(remaining))


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 \
        else f"{seconds // 60:d}m{seconds % 60:02d}s"


def _resume_filter(rics: list[str], ric_to_id: dict, start: str | None,
                   table: str, key: str) -> list[str]:
    """Drop securities whose stored history already reaches back to `start`.

    Backfills are long enough that they get interrupted — a laptop sleeping, a
    session kicked, a dropped VPN. Resuming should cost one query, not another
    hour, so this skips anything already covered rather than re-fetching it.
    """
    if not start:
        return rics
    cov = store.coverage(table, key)
    if cov.empty:
        return rics
    lo = dict(zip(cov[key].astype(str), pd.to_datetime(cov["lo"])))
    cutoff = pd.Timestamp(start)
    keep = []
    for ric in rics:
        ident = ric_to_id.get(ric, ric) if key == "security_id" else ric
        have = lo.get(str(ident))
        # A few days of slack: exchanges have holidays and listings start late.
        if have is None or have > cutoff + pd.Timedelta(days=7):
            keep.append(ric)
    log.info("[RESUME] %d of %d already cover %s — %d to fetch",
             len(rics) - len(keep), len(rics), start, len(keep))
    return keep


def run_eod(start: str | None = None, days: int | None = None,
            universe: pd.DataFrame | None = None, resume: bool = False,
            batch_size: int | None = None) -> dict:
    """Fetch EOD OHLCV for the active non-INDEX universe and upsert the store.

    `start` (ISO date) enables history backfill for newly added securities;
    default window is the configured lookback. `resume` skips securities whose
    stored history already reaches `start`.

    Each batch is written in its own short transaction. That matters: DuckDB's
    write lock is file-exclusive, so holding one connection across a 1,400-name
    backfill would keep the dashboard offline for the entire run.
    """
    master = universe if universe is not None else load_universe()
    eod_master = master[master["asset_type"] != "INDEX"]
    ric_to_id = dict(zip(eod_master["ric"], eod_master["security_id"]))
    rics = eod_master["ric"].tolist()
    lo, hi = _window(start, days)
    if resume:
        rics = _resume_filter(rics, ric_to_id, start, "eod_prices", "security_id")
    if not rics:
        log.info("[EOD] nothing to fetch")
        return {"succeeded": 0, "failed": 0, "failed_rics": [], "skipped": True}

    size = batch_size or _EOD_BATCH
    n_batches = -(-len(rics) // size)
    log.info("[EOD] %d RICs  %s -> %s  (%d batches of %d)", len(rics), lo, hi, n_batches, size)

    succeeded, failed = [], []
    started = time.time()
    for i in range(0, len(rics), size):
        batch = rics[i:i + size]
        try:
            raw = _fetch_history(batch, _OHLCV_FIELDS, lo, hi)
        except Exception as exc:                                     # noqa: BLE001
            log.error("[EOD] batch %d failed: %s", i // size + 1, str(exc)[:150])
            failed.extend(batch)
            continue
        per_ric = _split_response(raw, batch)
        frames = []
        for ric in batch:
            norm = _normalise_eod(per_ric.get(ric, pd.DataFrame()), ric, ric_to_id)
            if norm is None:
                failed.append(ric)
                continue
            frames.append(norm)
            succeeded.append(ric)
        rows = store.flush("eod_prices", frames)
        log.info("[EOD] batch %d/%d  %d RICs  %d rows", i // size + 1, n_batches,
                 len(frames), rows)
        _progress(min(i + size, len(rics)), len(rics), started, "EOD")

    if succeeded:
        con = store.write_connection()
        try:
            store.bump_data_version(con)
        finally:
            con.close()

    log.info("[EOD] done in %s: succeeded=%d failed=%d %s", _hms(time.time() - started),
             len(succeeded), len(failed), failed[:20] or "")
    return {"succeeded": len(succeeded), "failed": len(failed), "failed_rics": failed}


def _normalise_fund_cols(df: pd.DataFrame, fields: list[str],
                         field_to_col: dict[str, str]) -> pd.DataFrame:
    """Rename an LSEG response onto store columns.

    Positionally, because LSEG returns columns in the order they were requested
    and *decorates the labels*: "TR.PriceToBVPerShare" comes back as "Price To
    Book Value Per Share (Daily Time Series Ratio)" and "TR.GrossMargin" as
    "Gross Margin, Percent". Matching on those labels is a guessing game that
    silently yields NULL columns; matching on position cannot drift.

    Label matching remains as a fallback for the case where the response has a
    different column count than requested (a field the server dropped).
    """
    wanted = [field_to_col[f] for f in fields if f in field_to_col]
    if len(df.columns) == len(wanted):
        out = df.copy()
        out.columns = wanted
        return out

    log.warning("[FUND] response had %d columns for %d fields — falling back to "
                "label matching (%s)", len(df.columns), len(wanted), list(df.columns))
    keys = sorted(((_norm_key(f.removeprefix("TR.")), field_to_col[f])
                   for f in fields if f in field_to_col),
                  key=lambda kv: len(kv[0]), reverse=True)
    rename, claimed = {}, set()
    for col in df.columns:
        key = _norm_key(col)
        # Exact, then prefix — never a bare substring: "pe" is a substring of
        # "pricetobookvaluepershare" and would claim the wrong column.
        match = next((v for k, v in keys
                      if v not in claimed and (key == k or key.startswith(k))), None)
        if match:
            rename[col] = match
            claimed.add(match)
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


# ── Field resolution ──────────────────────────────────────────────────────────

def _field_returns_data(rics: list[str], field: str) -> bool:
    """Does this TR field actually deliver numbers on this entitlement?

    A field can fail three ways — denied, unresolvable, or resolvable but empty —
    and only the third is silent. Asking for a short recent window and counting
    non-null values catches all three.
    """
    lo, hi = _window(None, 60)
    try:
        raw = ld_get_history(rics, [field], lo, hi)
    except Exception as exc:                                          # noqa: BLE001
        log.debug("[FIELDS] %s: %s", field, str(exc)[:120])
        return False
    if raw is None or raw.empty:
        return False
    values = pd.to_numeric(raw.stack(future_stack=True), errors="coerce")
    return bool(values.notna().sum())


def ld_get_history(rics: list[str], fields: list[str], start: str, end: str) -> pd.DataFrame:
    import lseg.data as ld
    raw = ld.get_history(universe=rics, fields=fields, start=start, end=end, interval="1D")
    return raw if raw is not None else pd.DataFrame()


def resolve_fund_fields(sample_rics: list[str] | None = None,
                        use_cache: bool = True) -> dict[str, str]:
    """Pick the first candidate TR field that delivers data for each store column.

    Cached in meta.lseg_fields so the nightly run does not spend a minute
    re-probing. `capital-ingest fund --probe` refreshes it.
    """
    if use_cache:
        cached = store.get_meta(_META_FIELDS_KEY)
        if cached:
            try:
                resolved = json.loads(cached)
                if isinstance(resolved, dict) and resolved:
                    return resolved
            except json.JSONDecodeError:
                pass

    rics = sample_rics or [r for r in load_universe()["ric"].tolist()
                           if not r.startswith(".")][:3]
    resolved: dict[str, str] = {}
    for col, candidates in FUND_CANDIDATES.items():
        for field in candidates:
            if _field_returns_data(rics, field):
                resolved[col] = field
                log.info("[FIELDS] %-20s -> %s", col, field)
                break
        else:
            log.warning("[FIELDS] %-20s -> unavailable (tried %s)", col, ", ".join(candidates))

    missing = [c for c in FUND_REQUIRED if c not in resolved]
    if missing:
        raise RuntimeError(f"required fundamentals fields unavailable: {missing}. "
                           f"Check the LSEG entitlement before ingesting.")
    con = store.write_connection()
    try:
        store.set_meta(con, _META_FIELDS_KEY, json.dumps(resolved))
    finally:
        con.close()
    return resolved


def probe_fundamental_fields(sample_rics: list[str] | None = None) -> pd.DataFrame:
    """Report which TR field delivers each descriptor, and which have none.

    Field names differ by entitlement and LSEG renames them, so rather than
    assume, ask. Run this before wiring a new descriptor into the factor model —
    an unavailable field is not an error, it just means the model keeps dropping
    that style and says so in its coverage report.
    """
    rics = sample_rics or [r for r in load_universe()["ric"].tolist()
                           if not r.startswith(".")][:3]
    rows = []
    for col, candidates in FUND_CANDIDATES.items():
        winner = next((f for f in candidates if _field_returns_data(rics, f)), None)
        rows.append({"column": col, "required": col in FUND_REQUIRED,
                     "field": winner or "—", "ok": winner is not None,
                     "candidates_tried": len(candidates)})
    out = pd.DataFrame(rows)
    resolved = {r["column"]: r["field"] for _, r in out.iterrows() if r["ok"]}
    con = store.write_connection()
    try:
        store.set_meta(con, _META_FIELDS_KEY, json.dumps(resolved))
    finally:
        con.close()
    log.info("[PROBE] resolved %d/%d columns; cached to meta.%s",
             len(resolved), len(out), _META_FIELDS_KEY)
    return out


# ── Fundamentals ingest ───────────────────────────────────────────────────────

def run_fundamentals(start: str | None = None, days: int | None = None,
                     universe: pd.DataFrame | None = None,
                     resolved: dict[str, str] | None = None,
                     resume: bool = False, batch_size: int | None = None) -> dict:
    """Fetch every resolvable descriptor for the active universe and upsert it.

    Fields are requested a few at a time; a request that fails takes only its own
    columns down, leaving them NULL rather than losing the batch. The factor
    model reads a NULL column as "descriptor unavailable" and drops the style
    with a stated reason, so partial data degrades honestly.
    """
    master = universe if universe is not None else load_universe()
    fund = master[~master["ric"].str.startswith(".")]
    ric_to_ticker = dict(zip(fund["ric"], fund["ticker"]))
    rics = fund["ric"].tolist()
    lo, hi = _window(start, days)
    if resume:
        rics = _resume_filter(rics, {}, start, "fundamentals", "ric")
    if not rics:
        log.info("[FUND] nothing to fetch")
        return {"rows": 0, "skipped": True}

    resolved = resolved or resolve_fund_fields()
    field_to_col = {field: col for col, field in resolved.items()}
    ordered_fields = list(resolved.values())
    field_groups = [ordered_fields[i:i + _FUND_FIELDS_PER_REQUEST]
                    for i in range(0, len(ordered_fields), _FUND_FIELDS_PER_REQUEST)]
    log.info("[FUND] %d RICs  %s -> %s  (%d columns in %d requests per batch)",
             len(rics), lo, hi, len(resolved), len(field_groups))

    gics_map = _fetch_gics(rics)
    size = batch_size or _FUND_BATCH
    n_batches = -(-len(rics) // size)
    failures, total_rows, seen_rics = 0, 0, set()
    filled: dict[str, int] = {c: 0 for c in FUND_NUMERIC_COLS}
    started = time.time()

    for i in range(0, len(rics), size):
        batch = rics[i:i + size]
        frames: list[pd.DataFrame] = []
        merged: dict[str, pd.DataFrame] = {}
        for fields in field_groups:
            try:
                raw = _fetch_history(batch, fields, lo, hi)
            except Exception as exc:                                  # noqa: BLE001
                failures += 1
                log.warning("[FUND] fields %s failed on this batch: %s", fields, str(exc)[:120])
                continue
            if raw is None or raw.empty:
                continue
            for ric, sub in _split_response(raw, batch).items():
                sub = _normalise_fund_cols(sub.copy(), fields, field_to_col)
                sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
                if "date" not in sub.columns:
                    continue
                sub["date"] = pd.to_datetime(sub["date"], errors="coerce").dt.date
                keep = ["date"] + [c for c in sub.columns if c in FUND_NUMERIC_COLS]
                sub = sub[keep].dropna(subset=["date"]).drop_duplicates(subset="date", keep="last")
                merged[ric] = sub if ric not in merged else merged[ric].merge(
                    sub, on="date", how="outer", suffixes=("", "_dup"))

        for ric, df in merged.items():
            df = df.loc[:, ~df.columns.str.endswith("_dup")].copy()
            df.insert(0, "ric", ric)
            df["ticker"] = ric_to_ticker.get(ric, ric)
            df["gics_sector"] = gics_map.get(ric.upper(), "")
            frames.append(df)

        if not frames:
            continue
        # Normalise and write this batch, then let go of the writer. Accumulating
        # the whole universe first would mean gigabytes in memory and one very
        # long lock; neither is acceptable during a backfill.
        batch_df = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
        for col in FUND_NUMERIC_COLS:
            batch_df[col] = pd.to_numeric(batch_df.get(col), errors="coerce")
        batch_df = batch_df.drop_duplicates(subset=["ric", "date"], keep="last")
        ordered = ["ric", "date", "ticker", "gics_sector", *FUND_NUMERIC_COLS]
        batch_df = batch_df[[c for c in ordered if c in batch_df.columns]]

        total_rows += store.flush("fundamentals", [batch_df])
        seen_rics.update(batch_df["ric"].unique())
        for col in FUND_NUMERIC_COLS:
            if col in batch_df:
                filled[col] += int(batch_df[col].notna().sum())
        log.info("[FUND] batch %d/%d  %d rows", i // size + 1, n_batches, len(batch_df))
        _progress(min(i + size, len(rics)), len(rics), started, "FUND")

    if not total_rows:
        log.warning("[FUND] no fundamentals returned")
        return {"rows": 0, "request_failures": failures}

    con = store.write_connection()
    try:
        store.bump_data_version(con)
    finally:
        con.close()
    log.info("[FUND] done in %s: %d rows, %d RICs, request failures=%d, non-null: %s",
             _hms(time.time() - started), total_rows, len(seen_rics), failures, filled)
    return {"rows": total_rows, "rics": len(seen_rics),
            "request_failures": failures, "non_null": filled, "fields": resolved}
