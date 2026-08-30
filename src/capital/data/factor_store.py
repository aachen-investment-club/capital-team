"""
Persistence for factor-model runs.

A run is an immutable directory of parquet files plus a JSON manifest:

    CAPITAL_CACHE/factor_models/<run_id>/
        manifest.json           spec, summary, coverage report, factor groups
        factor_returns.parquet  date x factor
        specific_returns.parquet
        factor_cov.parquet      factor x factor (annualised)
        specific_risk.parquet   security_id -> idiosyncratic vol
        exposure_matrix.parquet security_id x factor, latest cross-section
        exposures.parquet       long panel: date, security_id, factor, value
        security_meta.parquet
        diag_*.parquet          one per diagnostic table

Why files and not the DuckDB store: the store has a single-writer contract and
the nightly ingest owns that writer. A dashboard-triggered job must never
contend for it. Parquet directories are append-only, readable by every process
at once, and a half-written run is discarded rather than corrupting anything.

Runs are immutable once written, so reads are memoised with a plain LRU rather
than the version-keyed loader cache — a nightly ingest does not change what a
finished run computed, it only makes it *older*, which the UI shows from the
manifest's `data_version`.
"""
import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd

from capital.settings import settings

RUNS_DIR = settings.cache_dir / "factor_models"

_FRAMES = ("factor_returns", "factor_std_errors", "specific_returns", "fit_stats",
           "factor_cov", "factor_corr", "specific_risk", "exposure_matrix",
           "security_meta")


def _dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


# ── Write ─────────────────────────────────────────────────────────────────────

#: Index name each frame is written under, so reads can set it back without
#: guessing at pandas' "index"/"level_0" fallbacks.
_INDEX_NAMES = {
    "factor_returns": "date", "factor_std_errors": "date", "specific_returns": "date",
    "fit_stats": "date", "factor_cov": "factor", "factor_corr": "factor",
    "specific_risk": "security_id", "exposure_matrix": "security_id",
}


def _write(frame: pd.DataFrame, path: Path, index_name: str | None = None) -> None:
    if frame is None or len(frame) == 0:
        return
    out = frame
    if not isinstance(frame.index, pd.RangeIndex):
        out = frame.rename_axis(index_name or frame.index.name or "index").reset_index()
    out = out.copy()
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(path, index=False)


def save_run(result, run_id: str | None = None) -> str:
    """Persist a FactorModelResult and return its run id.

    Written to a `.partial` directory first and renamed on success, so a crashed
    or cancelled job never leaves a half-run that the dashboard would try to load.
    """
    from capital.data.cache import data_version

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = run_id or f"{stamp}-{abs(hash(json.dumps(result.spec.to_dict(), sort_keys=True, default=str))) % (16 ** 8):08x}"
    final = _dir(run_id)
    staging = final.with_suffix(".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for name in _FRAMES:
        _write(getattr(result, name, None), staging / f"{name}.parquet",
               _INDEX_NAMES.get(name))

    # Long exposure panel: float32 and sorted by factor so per-factor and
    # per-security reads prune row groups instead of scanning the file.
    if result.style_panels:
        frames = []
        for factor, panel in result.style_panels.items():
            long = panel.stack(future_stack=True).rename("value").reset_index()
            long.columns = ["date", "security_id", "value"]
            long["factor"] = factor
            frames.append(long)
        exposures = pd.concat(frames, ignore_index=True)
        exposures["value"] = exposures["value"].astype("float32")
        exposures = exposures.dropna(subset=["value"]).sort_values(["factor", "security_id", "date"])
        exposures.to_parquet(staging / "exposures.parquet", index=False, row_group_size=200_000)

    for name, frame in (result.diagnostics or {}).items():
        if isinstance(frame, pd.DataFrame) and len(frame):
            _write(frame, staging / f"diag_{name}.parquet")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": data_version(),
        "spec": result.spec.to_dict(),
        "summary": result.summary,
        "coverage": result.coverage,
        "factor_groups": result.factor_groups,
        "factors": list(result.factor_returns.columns),
        "styles": list(result.style_panels),
        "diagnostics": sorted(n for n, f in (result.diagnostics or {}).items()
                              if isinstance(f, pd.DataFrame) and len(f)),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    if final.exists():
        shutil.rmtree(final)
    staging.rename(final)
    _load_frame.cache_clear()
    return run_id


# ── Read ──────────────────────────────────────────────────────────────────────

def list_runs(limit: int = 50) -> list[dict]:
    """Manifests of complete runs, newest first."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for path in RUNS_DIR.iterdir():
        if not path.is_dir() or path.suffix == ".partial":
            continue
        manifest = path / "manifest.json"
        if not manifest.exists():
            continue
        try:
            out.append(json.loads(manifest.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out[:limit]


def load_manifest(run_id: str) -> dict | None:
    path = _dir(run_id) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def latest_run_id() -> str | None:
    runs = list_runs(limit=1)
    return runs[0]["run_id"] if runs else None


@lru_cache(maxsize=96)
def _load_frame(run_id: str, name: str) -> pd.DataFrame:
    path = _dir(run_id) / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_frame(run_id: str, name: str, index: str | None = None) -> pd.DataFrame:
    """One of the run's tables. Treat the result as read-only — it is memoised."""
    df = _load_frame(run_id, name)
    if df.empty or index is None or index not in df.columns:
        return df
    out = df.set_index(index)
    if index == "date":
        out.index = pd.to_datetime(out.index)
    return out


def load_factor_returns(run_id: str) -> pd.DataFrame:
    return load_frame(run_id, "factor_returns", index="date").sort_index()


def load_exposure_matrix(run_id: str) -> pd.DataFrame:
    return load_frame(run_id, "exposure_matrix", index="security_id")


def load_specific_risk(run_id: str) -> pd.DataFrame:
    return load_frame(run_id, "specific_risk", index="security_id")


def load_security_meta(run_id: str) -> pd.DataFrame:
    return _load_frame(run_id, "security_meta")


def load_covariance(run_id: str) -> pd.DataFrame:
    return load_frame(run_id, "factor_cov", index="factor").rename_axis(None)


def load_correlation(run_id: str) -> pd.DataFrame:
    return load_frame(run_id, "factor_corr", index="factor").rename_axis(None)


def load_diagnostic(run_id: str, name: str) -> pd.DataFrame:
    return _load_frame(run_id, f"diag_{name}")


def load_exposures(run_id: str, factors: tuple[str, ...] | None = None,
                   security_ids: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The long exposure panel, filtered at the parquet level.

    Filtering here rather than after the read is what keeps the security explorer
    responsive: a per-security query touches a couple of row groups instead of
    the whole 10M-row panel.
    """
    path = _dir(run_id) / "exposures.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["date", "security_id", "factor", "value"])
    filters = []
    if factors:
        filters.append(("factor", "in", list(factors)))
    if security_ids:
        filters.append(("security_id", "in", list(security_ids)))
    df = pd.read_parquet(path, filters=filters or None)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def exposure_snapshot(run_id: str, date=None) -> pd.DataFrame:
    """Wide security_id x style-factor exposures on one date (default: latest)."""
    long = load_exposures(run_id)
    if long.empty:
        return pd.DataFrame()
    when = pd.Timestamp(date) if date is not None else long["date"].max()
    snap = long[long["date"] == when]
    if snap.empty:
        when = long.loc[long["date"] <= when, "date"].max()
        snap = long[long["date"] == when]
    return snap.pivot(index="security_id", columns="factor", values="value")


def delete_run(run_id: str) -> bool:
    path = _dir(run_id)
    if not path.exists():
        return False
    shutil.rmtree(path)
    _load_frame.cache_clear()
    return True
