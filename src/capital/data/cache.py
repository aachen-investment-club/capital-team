"""
Version-keyed loader cache (replaces st.cache_data).

Uses cachelib's FileSystemCache directly — framework-free, so the same
decorator works in Dash callbacks, ingest scripts, and tests, and the cache
directory is shared by all gunicorn workers. Every cached value is keyed on
the store's `data_version`, so the nightly ingest invalidates everything at
once by bumping it.
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


# data_version() hits the DB file; memoize it for 60s like the old _eod_data_version.
_version_state = {"at": 0.0, "value": ""}


def data_version() -> str:
    now = time.monotonic()
    if now - _version_state["at"] > 60:
        _version_state["value"] = store.data_version()
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
