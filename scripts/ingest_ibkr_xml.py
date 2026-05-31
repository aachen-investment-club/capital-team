"""
One-time (and repeatable) ingest script: parse IBKR Flex Query XML files
and write/update the Parquet files consumed by the dashboard.

Usage:
    python scripts/ingest_ibkr_xml.py

Daily workflow (add today's open positions):
    1. Run the OpenPositions flex query in IBKR, save XML to:
           data/ibkr/open_positions/YYYYMMDD.xml
    2. Re-run this script — it will append only new dates.
"""
import sys
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd
from lib.ibkr import (
    parse_trades,
    parse_prior_positions,
    parse_open_positions,
    parse_fx_positions,
    append_open_positions,
    append_fx_positions,
    IBKR_DIR,
)

_TRADE_LOG_XML    = IBKR_DIR / "trade_log.xml"
_PRIOR_POS_XML    = IBKR_DIR / "prior_positions.xml"
_OPEN_POS_XML_DIR = IBKR_DIR / "open_positions"

_TRADE_LOG_PARQUET = IBKR_DIR / "trade_log.parquet"
_PRIOR_POS_PARQUET = IBKR_DIR / "prior_positions.parquet"


def ingest_trade_log() -> None:
    print("[trade_log]")
    df = parse_trades(_TRADE_LOG_XML.read_text())
    df.to_parquet(_TRADE_LOG_PARQUET, index=False)
    print(f"  wrote {_TRADE_LOG_PARQUET}  ({len(df)} rows)")


def ingest_prior_positions() -> None:
    print("[prior_positions]")
    df = parse_prior_positions(_PRIOR_POS_XML.read_text())
    df.to_parquet(_PRIOR_POS_PARQUET, index=False)
    print(f"  wrote {_PRIOR_POS_PARQUET}  ({len(df)} rows)")


def ingest_open_positions() -> None:
    print("[open_positions + fx_positions]")
    xml_files = sorted(_OPEN_POS_XML_DIR.glob("*.xml"))
    if not xml_files:
        print("  no XML files found in", _OPEN_POS_XML_DIR)
        return
    for xml_path in xml_files:
        xml_text = xml_path.read_text()
        eq_df = parse_open_positions(xml_text)
        fx_df = parse_fx_positions(xml_text)
        print(f"  {xml_path.name}: {len(eq_df)} equity, {len(fx_df)} FX positions")
        append_open_positions(eq_df)
        if not fx_df.empty:
            append_fx_positions(fx_df)


if __name__ == "__main__":
    print("Ingesting IBKR Flex Query XML files…\n")
    ingest_trade_log()
    print()
    ingest_prior_positions()
    print()
    ingest_open_positions()
    print("\nDone.")
