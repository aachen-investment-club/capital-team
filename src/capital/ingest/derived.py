"""
Nightly precompute of derived tables inside the DuckDB store.

Everything input-independent that pages would otherwise compute per request
lives here, so dashboard callbacks are pure reads. Full rebuild each night —
at a few million rows this is seconds, and simpler than incremental append.

Factor-model outputs deliberately do not live here: a run is parameterised (its
window, frequency and factor set are user choices), so it is persisted per run by
`capital.data.factor_store` and estimated by `capital-ingest factors`, which runs
immediately after this step in the nightly pipeline.
"""
import logging

from capital.data import store

log = logging.getLogger(__name__)

_DAILY_RETURNS_SQL = """
INSERT OR REPLACE INTO daily_returns
SELECT security_id, date,
       adj_close / lag(adj_close) OVER w - 1 AS ret,
       ln(adj_close / lag(adj_close) OVER w) AS log_ret
FROM eod_prices
WINDOW w AS (PARTITION BY security_id ORDER BY date)
QUALIFY ret IS NOT NULL
"""

_ROLLING_STATS_SQL = """
INSERT OR REPLACE INTO rolling_stats
WITH j AS (
    SELECT p.security_id, p.date, p.adj_close, p.close, p.volume, r.ret,
           row_number() OVER (PARTITION BY p.security_id ORDER BY p.date) AS rn
    FROM eod_prices p
    LEFT JOIN daily_returns r USING (security_id, date)
),
base AS (
    SELECT security_id, date, rn,
           CASE WHEN rn >= 20
                THEN stddev_samp(ret) OVER w20 * sqrt(252) END AS vol_20d,
           CASE WHEN rn >= 60
                THEN stddev_samp(ret) OVER w60 * sqrt(252) END AS vol_60d,
           lag(adj_close, 21)  OVER wo / lag(adj_close, 252) OVER wo - 1 AS mom_12_1,
           adj_close / max(adj_close) OVER w252 - 1 AS dd,
           CASE WHEN rn >= 20 THEN avg(close * volume) OVER w20 END AS adv_20d
    FROM j
    WINDOW wo   AS (PARTITION BY security_id ORDER BY date),
           w20  AS (PARTITION BY security_id ORDER BY date ROWS BETWEEN 19  PRECEDING AND CURRENT ROW),
           w60  AS (PARTITION BY security_id ORDER BY date ROWS BETWEEN 59  PRECEDING AND CURRENT ROW),
           w252 AS (PARTITION BY security_id ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
)
SELECT security_id, date, vol_20d, vol_60d, mom_12_1,
       min(dd) OVER (PARTITION BY security_id ORDER BY date
                     ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS max_dd_1y,
       adv_20d
FROM base
"""


def run_derived() -> dict:
    con = store.write_connection()
    try:
        log.info("[DERIVED] daily_returns ...")
        con.execute(_DAILY_RETURNS_SQL)
        log.info("[DERIVED] rolling_stats ...")
        con.execute(_ROLLING_STATS_SQL)
        counts = {
            t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("daily_returns", "rolling_stats")
        }
        store.bump_data_version(con)
    finally:
        con.close()
    log.info("[DERIVED] done: %s", counts)
    return counts
