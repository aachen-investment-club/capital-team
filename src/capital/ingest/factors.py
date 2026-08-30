"""
Nightly factor-model precompute.

The dashboard can queue a run at any time, but the common case — "what is the
portfolio exposed to this morning" — should not make the first person in wait
for an estimation. So the nightly pipeline builds one standard run after the
data lands, and the Factor Screen opens on a finished result.

The nightly spec deliberately asks for *every* style. Styles whose descriptors
are not in the store are dropped and reported in the run's coverage report, so
the day a new fundamentals column is back-filled, the next night's run picks the
factor up with no code change.
"""
import logging

from capital.analytics.factors.model import run_factor_model
from capital.analytics.factors.spec import ALL_STYLES, ModelSpec
from capital.data import factor_store

log = logging.getLogger(__name__)

NIGHTLY_NAME = "Nightly"
KEEP_NIGHTLY = 10          # ~2 weeks of history; older nightly runs are pruned


def default_spec(years: int = 3, frequency: str = "W-FRI") -> ModelSpec:
    import pandas as pd
    return ModelSpec(
        name=NIGHTLY_NAME,
        start=(pd.Timestamp.today() - pd.DateOffset(years=years)).date().isoformat(),
        frequency=frequency,
        styles=ALL_STYLES,
    )


def prune(keep: int = KEEP_NIGHTLY) -> int:
    """Drop all but the newest `keep` nightly runs. Manual runs are never touched
    — someone queued those deliberately and may still be looking at one."""
    nightly = [r for r in factor_store.list_runs(limit=500)
               if (r.get("spec") or {}).get("name") == NIGHTLY_NAME]
    removed = 0
    for manifest in nightly[keep:]:
        if factor_store.delete_run(manifest["run_id"]):
            removed += 1
    return removed


def run_factors(years: int = 3, frequency: str = "W-FRI", keep: int = KEEP_NIGHTLY) -> dict:
    spec = default_spec(years=years, frequency=frequency)
    log.info("[FACTORS] estimating %s: %s -> latest, %s", spec.name, spec.start, frequency)

    def progress(frac: float, message: str) -> None:
        log.info("[FACTORS] %3.0f%% %s", 100 * frac, message)

    result = run_factor_model(spec, progress=progress)
    run_id = factor_store.save_run(result)
    removed = prune(keep)
    dropped = sorted(result.coverage.get("dropped_styles", {}))
    log.info("[FACTORS] done: run=%s securities=%d periods=%d styles=%s dropped=%s pruned=%d",
             run_id, result.summary["n_securities"], result.summary["n_periods"],
             result.coverage.get("styles_estimated"), dropped or "none", removed)
    return {"run_id": run_id, "pruned": removed,
            "styles": result.coverage.get("styles_estimated"),
            "dropped_styles": dropped, **result.summary}
