"""
Job-kind registry.

A handler is ``fn(params: dict, progress: Callable[[float, str], None]) -> dict``
and must return a small JSON-serialisable summary (ids, counts, timings) — never
a DataFrame. Bulk output belongs in a store the dashboard can read back, e.g.
``capital.data.factor_store``.

Imported by the runner subprocess, so keep the module-level imports light: pull
heavy analytics inside the handler body.
"""
from collections.abc import Callable

HANDLERS: dict[str, Callable[[dict, Callable[[float, str], None]], dict]] = {}


def register(kind: str):
    def deco(fn):
        HANDLERS[kind] = fn
        return fn
    return deco


def get_handler(kind: str):
    if kind not in HANDLERS:
        raise KeyError(f"no handler registered for job kind {kind!r} "
                       f"(known: {sorted(HANDLERS)})")
    return HANDLERS[kind]


# ── Registrations ─────────────────────────────────────────────────────────────

@register("factor_model")
def _factor_model(params: dict, progress) -> dict:
    """Estimate a cross-sectional factor model and persist the run."""
    from capital.analytics.factors.model import run_factor_model
    from capital.analytics.factors.spec import ModelSpec
    from capital.data import factor_store

    spec = ModelSpec.from_dict(params)
    result = run_factor_model(spec, progress=progress)
    progress(0.95, "Writing results")
    run_id = factor_store.save_run(result)
    return {
        "run_id": run_id,
        "securities": int(result.summary["n_securities"]),
        "periods": int(result.summary["n_periods"]),
        "factors": int(result.summary["n_factors"]),
        "mean_r2": result.summary["mean_r2"],
    }
