"""
Background job queue for long dashboard computations.

Public surface is deliberately tiny — pages submit work and poll status:

    from capital.jobs import queue
    job_id = queue.submit("factor_model", params={...}, label="Full · 3y · weekly")
    job    = queue.get(job_id)          # dict with status / progress / message
    jobs   = queue.list_jobs("factor_model")

See queue.py for the design rationale.
"""
from capital.jobs import queue  # noqa: F401
