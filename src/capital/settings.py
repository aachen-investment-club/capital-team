"""
Central configuration — the only module that reads environment variables.

Everything (dashboard, ingestion, scripts) imports `settings` from here so the
whole system is configured from a single .env / EnvironmentFile.
"""
import os
import pathlib
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Repo root: src/capital/settings.py -> parents[2] (editable install keeps files in place).
# Override with CAPITAL_ROOT when running from an installed wheel.
ROOT = pathlib.Path(os.getenv("CAPITAL_ROOT", pathlib.Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")


def _path_env(name: str, default: pathlib.Path) -> pathlib.Path:
    v = os.getenv(name, "")
    return pathlib.Path(v) if v else default


@dataclass(frozen=True)
class Settings:
    root: pathlib.Path = ROOT

    # AWS
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    aws_region: str = os.getenv("AWS_REGION", "eu-central-1")
    ddb_baskets_table: str = os.getenv("DDB_TABLE", "")            # themes (fund-baskets)
    ddb_fund_table: str = os.getenv("DDB_FUND_TABLE", "fund-data")  # website metrics/positions

    # Local DuckDB store + caches
    db_path: pathlib.Path = _path_env("CAPITAL_DB", ROOT / "data" / "market.duckdb")
    cache_dir: pathlib.Path = _path_env("CAPITAL_CACHE", ROOT / "data" / "cache")

    # Ingestion credentials
    ibkr_flex_token: str = os.getenv("IBKR_FLEX_TOKEN", "")
    ibkr_query_id: str = os.getenv("IBKR_QUERY_ID", "")
    fred_api_key: str = os.getenv("FRED_API_KEY", "")  # free key; without it FRED history caps at ~3y
    # LSEG session: "platform" (RDP GrantPassword, headless — the server) or
    # "desktop" (connects to a running LSEG Workspace on this machine — local use).
    lseg_session_type: str = os.getenv("LSEG_SESSION_TYPE", "platform")
    lseg_app_key: str = os.getenv("LSEG_APP_KEY", "")
    lseg_username: str = os.getenv("LSEG_USERNAME", "")
    lseg_password: str = os.getenv("LSEG_PASSWORD", "")
    eod_lookback_days: int = int(os.getenv("LOOKBACK_DAYS", "5"))

    # Ops
    healthcheck_url: str = os.getenv("HEALTHCHECK_URL", "")

    # Stable S3 keys consumed by the club website (written by the external IBKR
    # Lambda) — read-only here, do not change.
    portfolio_prefix: str = "history/portfolio"
    derived_prefix: str = "history/portfolio/derived"
    # DuckDB store backup (the only S3 write this project makes).
    backup_prefix: str = "backup"

    config_dir: pathlib.Path = field(default_factory=lambda: ROOT / "config")


settings = Settings()
