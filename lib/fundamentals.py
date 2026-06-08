"""
Fundamentals data loader — split out of lib.data to avoid module-cache issues.
"""
import pandas as pd
import streamlit as st

# Import from lib.data at module level so load_dotenv runs before we read env vars
from lib.data import _con, _S3_BUCKET, _ROOT


@st.cache_data(ttl=3600)
def get_fundamentals() -> pd.DataFrame:
    """Daily fundamentals snapshot for the Barra universe.
    Columns: date, ric, ticker, gics_sector, pb_ratio, market_cap, shares_outstanding
    Populated by scripts/ingest_fundamentals.py.
    Returns empty DataFrame if not yet ingested.
    """
    if _S3_BUCKET:
        path = f"s3://{_S3_BUCKET}/history/fundamentals/data.parquet"
    else:
        path = str(_ROOT / "data" / "fundamentals" / "data.parquet")
    try:
        df = _con().execute(
            f"SELECT * FROM read_parquet('{path}') ORDER BY ric, date"
        ).df()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        raise RuntimeError(f"[fundamentals] read failed from {path}: {e}") from e
