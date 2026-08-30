"""
Job runner subprocess: ``python -m capital.jobs.runner <job_id>``.

Spawned by ``capital.jobs.queue.pump``. Runs one job to completion, streaming
progress into the job's JSON record and stdout/stderr into its log file, then
exits. Isolation is the point — a heavy pandas run must not share the
dashboard's GIL, and a crash here must not take the app down.
"""
import logging
import signal
import sys
import traceback

from capital.jobs import queue
from capital.jobs.handlers import get_handler


def _install_sigterm() -> None:
    """queue.cancel() sends SIGTERM — turn it into the cooperative exception."""
    def _handler(signum, frame):                     # noqa: ARG001
        raise queue.JobCancelled("Cancelled by user")
    signal.signal(signal.SIGTERM, _handler)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python -m capital.jobs.runner <job_id>", file=sys.stderr)
        return 2
    job_id = argv[0]

    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("capital.jobs.runner")

    job = queue.mark_running(job_id, pid=__import__("os").getpid())
    if job is None:
        print(f"job {job_id} not found", file=sys.stderr)
        return 1

    _install_sigterm()
    log.info("job %s (%s) starting — params=%s", job_id, job["kind"], job["params"])

    def progress(frac: float, message: str) -> None:
        log.info("[%3.0f%%] %s", 100 * frac, message)
        queue.report(job_id, frac, message)

    try:
        result = get_handler(job["kind"])(job["params"], progress)
    except queue.JobCancelled as exc:
        log.warning("job %s cancelled: %s", job_id, exc)
        queue.finish(job_id, queue.CANCELLED, message="Cancelled")
        return 0
    except Exception:                                        # noqa: BLE001
        tb = traceback.format_exc()
        log.error("job %s failed\n%s", job_id, tb)
        queue.finish(job_id, queue.FAILED, error=tb, message="Failed")
        return 1

    log.info("job %s done: %s", job_id, result)
    queue.finish(job_id, queue.DONE, result=result, message="Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
