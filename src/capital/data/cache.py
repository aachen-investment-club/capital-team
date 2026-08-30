"""
Version-keyed loader cache (replaces st.cache_data).

Uses cachelib's FileSystemCache directly — framework-free, so the same
decorator works in Dash callbacks, ingest scripts, and tests, and the cache
directory is shared by all gunicorn workers. Every cached value is keyed on
`data_version()`, so bumping it invalidates everything at once. That version
string folds together two independent writers: the DuckDB store's own
`meta.data_version` (bumped by `capital-ingest nightly`) and an ETag
fingerprint of the S3 `history/portfolio/*` JSONs (written by the separate
`fund-data-ingestion` Lambda) — otherwise pages that only ever call the
portfolio S3 loaders would never see a Lambda-side correction, since nothing
would bump the DuckDB version for them.
"""
import functools
import hashlib
import pickle
import time

from cachelib import FileSystemCache

from capital.data import store
from capital.settings import settings

_TTL = 6 * 3600
_cache = None



def _backend() -> FileSystemCache:
    global _cache
    if _cache is None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _cache = FileSystemCache(str(settings.cache_dir / "loaders"), default_timeout=_TTL)
    return _cache


_PORTFOLIO_TABLES = ("portfolio_and_benchmarks", "daily_weightings", "trade_log")


def _portfolio_version() -> str:
    """ETag/mtime fingerprint of the S3 website JSONs (see loaders._portfolio_json)."""
    parts = []
    if settings.s3_bucket:
        import boto3
        s3c = boto3.client("s3", region_name=settings.aws_region)
        for table in _PORTFOLIO_TABLES:
            prefix = settings.derived_prefix if table in ("portfolio_and_benchmarks", "daily_weightings") \
                else settings.portfolio_prefix
            try:
                parts.append(s3c.head_object(Bucket=settings.s3_bucket, Key=f"{prefix}/{table}.json")["ETag"])
            except Exception:
                parts.append("missing")
    else:
        for table in _PORTFOLIO_TABLES:
            derived = table in ("portfolio_and_benchmarks", "daily_weightings")
            path = (settings.root / "data" / "derived" / f"{table}.json") if derived \
                else (settings.root / "data" / f"{table}.json")
            parts.append(str(path.stat().st_mtime) if path.exists() else "missing")
    return "|".join(parts)


# Hits the DB file and (optionally) S3; memoize for 60s like the old _eod_data_version.
_version_state = {"at": 0.0, "value": ""}


def data_version() -> str:
    now = time.monotonic()
    if now - _version_state["at"] > 60:
        _version_state["value"] = f"{store.data_version()}|{_portfolio_version()}"
        _version_state["at"] = now
    return _version_state["value"]


def cached_by_version(fn):
    """Memoize `fn` on (its qualname, current data_version, args, kwargs)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        raw = pickle.dumps((fn.__module__, fn.__qualname__, data_version(), args,
                            tuple(sorted(kwargs.items()))))
        key = hashlib.sha256(raw).hexdigest()
        hit = _backend().get(key)
        if hit is not None:
            return hit
        result = fn(*args, **kwargs)
        _backend().set(key, result)
        return result

    return wrapper


def clear() -> None:
    _backend().clear()
