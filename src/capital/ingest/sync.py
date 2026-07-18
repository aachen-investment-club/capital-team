"""
S3 backup of the local DuckDB store.

Nightly: upload market.duckdb to s3://{bucket}/backup/market.duckdb.
Sundays: also keep a dated copy under backup/weekly/, pruning to the last 8.

This (plus the derived prefix owned by the IBKR Lambda) is the only S3 write
in the project — the dashboard itself never writes.
"""
import logging
from datetime import date

import boto3

from capital.settings import settings

log = logging.getLogger(__name__)

_KEEP_WEEKLY = 8


def run_sync() -> dict:
    if not settings.s3_bucket:
        log.warning("[SYNC] S3_BUCKET unset — skipping backup")
        return {"uploaded": False}
    if not settings.db_path.exists():
        log.warning("[SYNC] %s missing — nothing to back up", settings.db_path)
        return {"uploaded": False}

    s3 = boto3.client("s3", region_name=settings.aws_region)
    key = f"{settings.backup_prefix}/market.duckdb"
    s3.upload_file(str(settings.db_path), settings.s3_bucket, key)
    size_mb = settings.db_path.stat().st_size / 1e6
    log.info("[SYNC] uploaded s3://%s/%s (%.1f MB)", settings.s3_bucket, key, size_mb)

    weekly = None
    if date.today().isoweekday() == 7:  # Sunday
        weekly = f"{settings.backup_prefix}/weekly/market-{date.today():%Y%m%d}.duckdb"
        s3.copy_object(Bucket=settings.s3_bucket, Key=weekly,
                       CopySource={"Bucket": settings.s3_bucket, "Key": key})
        log.info("[SYNC] weekly copy → %s", weekly)
        # prune old weeklies
        resp = s3.list_objects_v2(Bucket=settings.s3_bucket,
                                  Prefix=f"{settings.backup_prefix}/weekly/")
        keys = sorted(o["Key"] for o in resp.get("Contents", []))
        for old in keys[:-_KEEP_WEEKLY]:
            s3.delete_object(Bucket=settings.s3_bucket, Key=old)
            log.info("[SYNC] pruned %s", old)

    return {"uploaded": True, "size_mb": round(size_mb, 1), "weekly": weekly}
