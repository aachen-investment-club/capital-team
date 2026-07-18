"""Nightly ingestion: LSEG EOD + fundamentals, yfinance/FRED market data,
derived-table precompute, and S3 backup sync — all into the local DuckDB store.

NOTE: the IBKR `fund-data-ingestion` Lambda is a separate project (it feeds the
club website) and is intentionally NOT part of this package. The dashboard only
reads its S3 outputs.
"""
