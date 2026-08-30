"""
File-backed job queue for long dashboard computations.

Why jobs and not a Dash background callback
-------------------------------------------
`pages/_template.py` already documents that `background=True` is unusable here:
dash-mantine-components 2.8 props are not picklable by dill, so DiskcacheManager
crashes on any callback returning a dmc component. Beyond that, a factor-model
run over 1–2k securities must survive a page reload, be visible to *every* user
(not just the browser tab that started it), and never occupy a gunicorn thread.
That is a job, not a callback.

Design
------
- One JSON file per job at ``CAPITAL_CACHE/jobs/<job_id>.json``, rewritten
  atomically (tmp + ``os.replace``). The directory *is* the queue: no broker, no
  extra database, and every process (gunicorn worker, CLI, a second dev server)
  sees the same state.
- Work runs in a **subprocess** — ``python -m capital.jobs.runner <job_id>``.
  Threads would share the GIL with the dashboard and make it sluggish under a
  pandas-heavy run; a subprocess cannot, and a segfault in numpy cannot take the
  app down with it.
- A daemon *pump* thread starts queued jobs while fewer than ``max_concurrent``
  are running, and reaps jobs whose runner died (stale heartbeat + dead pid).
  The UI's poll interval also calls ``pump()``, so the queue advances even if the
  thread was never started.
- Claiming is atomic via ``os.mkdir`` on a per-job claim directory, so two
  gunicorn workers pumping at the same instant cannot double-start a job.
- Cancellation is cooperative: ``cancel()`` writes a flag the runner sees at its
  next ``progress()`` checkpoint, with SIGTERM as the hard fallback.

Nothing here is keyed on ``data_version`` — job artifacts are durable outputs,
not a cache of the store. The *data_version at submit time* is recorded on the
job so the UI can flag a result as stale after a nightly ingest.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from capital.settings import settings

# ── Layout / tunables ─────────────────────────────────────────────────────────

JOBS_DIR = settings.cache_dir / "jobs"
CLAIMS_DIR = JOBS_DIR / ".claims"
LOGS_DIR = JOBS_DIR / "logs"

MAX_CONCURRENT = int(os.getenv("CAPITAL_JOB_WORKERS", "1"))
HEARTBEAT_TIMEOUT = float(os.getenv("CAPITAL_JOB_HEARTBEAT_TIMEOUT", "300"))
PUMP_INTERVAL = 2.0
KEEP_JOBS = 200          # newest N job files are kept; older terminal ones pruned

QUEUED, RUNNING, DONE, FAILED, CANCELLED = "queued", "running", "done", "failed", "cancelled"
TERMINAL = (DONE, FAILED, CANCELLED)


class JobCancelled(Exception):
    """Raised inside a runner when the job has been cancelled."""


# ── Low-level file helpers ────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_dirs() -> None:
    for d in (JOBS_DIR, CLAIMS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def log_path(job_id: str) -> Path:
    return LOGS_DIR / f"{job_id}.log"


def _read(path: Path) -> dict | None:
    """Tolerant read — a job file caught mid-write simply reads as absent."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write(job: dict) -> None:
    """Atomic rewrite so a concurrent reader never sees a half-written file."""
    _ensure_dirs()
    final = _path(job["id"])
    tmp = final.with_suffix(f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(job, fh)
    os.replace(tmp, final)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True          # exists, owned by someone else
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def submit(kind: str, params: dict, label: str = "", dedupe_key: str | None = None) -> str:
    """Queue a job and return its id.

    `dedupe_key` collapses identical requests: if a queued/running job carries
    the same key, its id is returned instead of starting a second run.
    """
    _ensure_dirs()
    if dedupe_key:
        for job in list_jobs(kind=kind, limit=KEEP_JOBS):
            if job.get("dedupe_key") == dedupe_key and job["status"] in (QUEUED, RUNNING):
                return job["id"]

    # Record the store version so the UI can mark a result stale after an ingest.
    try:
        from capital.data.cache import data_version
        version = data_version()
    except Exception:
        version = ""

    job = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "label": label or kind,
        "params": params,
        "dedupe_key": dedupe_key,
        "status": QUEUED,
        "progress": 0.0,
        "message": "Queued",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "heartbeat": None,
        "pid": None,
        "error": None,
        "result": None,
        "data_version": version,
        "cancel_requested": False,
    }
    _write(job)
    pump()
    return job["id"]


def get(job_id: str) -> dict | None:
    return _read(_path(job_id))


def list_jobs(kind: str | None = None, limit: int = 50) -> list[dict]:
    """Jobs newest-first. Cheap: a few hundred small JSON reads at most."""
    _ensure_dirs()
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        job = _read(path)
        if job and (kind is None or job.get("kind") == kind):
            jobs.append(job)
    jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return jobs[:limit]


def cancel(job_id: str) -> bool:
    """Request cancellation; SIGTERM the runner if it is already going."""
    job = get(job_id)
    if not job or job["status"] in TERMINAL:
        return False
    job["cancel_requested"] = True
    if job["status"] == QUEUED:
        job.update(status=CANCELLED, message="Cancelled before start", finished_at=_now())
    _write(job)
    if job["status"] == RUNNING and _pid_alive(job.get("pid")):
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except OSError:
            pass
    return True


def delete(job_id: str) -> bool:
    """Remove a terminal job's record and log (its artifacts are not touched)."""
    job = get(job_id)
    if not job or job["status"] not in TERMINAL:
        return False
    _path(job_id).unlink(missing_ok=True)
    log_path(job_id).unlink(missing_ok=True)
    return True


def tail_log(job_id: str, lines: int = 60) -> str:
    path = log_path(job_id)
    if not path.exists():
        return ""
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-lines:])
    except OSError:
        return ""


# ── Runner-side helpers (imported by capital.jobs.runner) ─────────────────────

def mark_running(job_id: str, pid: int) -> dict | None:
    job = get(job_id)
    if not job:
        return None
    job.update(status=RUNNING, pid=pid, started_at=_now(),
               heartbeat=time.time(), progress=0.0, message="Starting")
    _write(job)
    return job


def report(job_id: str, progress: float, message: str) -> None:
    """Persist progress and refresh the heartbeat; raises if cancel was asked."""
    job = get(job_id)
    if job is None:
        return
    if job.get("cancel_requested"):
        raise JobCancelled(message)
    job.update(progress=max(0.0, min(1.0, float(progress))), message=message,
               heartbeat=time.time())
    _write(job)


def finish(job_id: str, status: str, result: dict | None = None, error: str | None = None,
           message: str = "") -> None:
    job = get(job_id) or {"id": job_id}
    job.update(status=status, result=result, error=error, finished_at=_now(),
               heartbeat=time.time(),
               progress=1.0 if status == DONE else job.get("progress", 0.0),
               message=message or status.capitalize())
    _write(job)


# ── Pump: start queued work, reap dead runners ────────────────────────────────

def _spawn(job: dict) -> None:
    """Claim the job (atomic mkdir) and launch its runner subprocess."""
    claim = CLAIMS_DIR / job["id"]
    try:
        claim.mkdir()                      # atomic: fails if another pump won
    except FileExistsError:
        return
    _ensure_dirs()
    logfile = log_path(job["id"]).open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "capital.jobs.runner", job["id"]],
            cwd=str(settings.root), stdout=logfile, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    except Exception as exc:                       # noqa: BLE001 — surface to the UI
        logfile.close()
        finish(job["id"], FAILED, error=f"could not start runner: {exc}")
        return
    # The runner rewrites the record itself; this is only so a job that dies
    # before its first write is still attributable to a pid.
    job.update(status=RUNNING, pid=proc.pid, started_at=_now(), heartbeat=time.time(),
               message="Starting")
    _write(job)


def _reap(jobs: list[dict]) -> None:
    """Fail jobs whose runner vanished (crash, OOM kill, server restart)."""
    stale_before = time.time() - HEARTBEAT_TIMEOUT
    for job in jobs:
        if job["status"] != RUNNING:
            continue
        beat = job.get("heartbeat") or 0
        if beat > stale_before or _pid_alive(job.get("pid")):
            continue
        finish(job["id"], FAILED,
               error="Runner process disappeared (crash, restart or out-of-memory).",
               message="Interrupted")


def _prune(jobs: list[dict]) -> None:
    for job in jobs[KEEP_JOBS:]:
        if job["status"] in TERMINAL:
            _path(job["id"]).unlink(missing_ok=True)
            log_path(job["id"]).unlink(missing_ok=True)
            claim = CLAIMS_DIR / job["id"]
            if claim.exists():
                claim.rmdir()


_pump_lock = threading.Lock()


def pump() -> None:
    """Advance the queue once. Idempotent, safe to call from any thread/process."""
    if not _pump_lock.acquire(blocking=False):
        return
    try:
        jobs = list_jobs(limit=KEEP_JOBS + 50)
        _reap(jobs)
        jobs = list_jobs(limit=KEEP_JOBS + 50)
        running = sum(1 for j in jobs if j["status"] == RUNNING)
        queued = [j for j in jobs if j["status"] == QUEUED]
        queued.sort(key=lambda j: j.get("created_at") or "")     # FIFO
        for job in queued[: max(0, MAX_CONCURRENT - running)]:
            _spawn(job)
        _prune(jobs)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[jobs] pump error: {exc}")
    finally:
        _pump_lock.release()


_pump_thread: threading.Thread | None = None


def start_pump() -> None:
    """Start the background pump once per process (called by the Dash app)."""
    global _pump_thread
    if _pump_thread is not None and _pump_thread.is_alive():
        return

    def _loop():
        while True:
            pump()
            time.sleep(PUMP_INTERVAL)

    _pump_thread = threading.Thread(target=_loop, name="capital-job-pump", daemon=True)
    _pump_thread.start()


# ── Misc ──────────────────────────────────────────────────────────────────────

def params_fingerprint(params: dict) -> str:
    """Stable short hash of a params dict — used for dedupe keys and run ids."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
