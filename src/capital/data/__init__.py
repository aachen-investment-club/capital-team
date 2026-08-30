"""
THE DATA CONTRACT — all data access goes through here.
Pages and analytics must never import boto3 or duckdb, and must never
reference S3 paths or the DB file directly.
"""
from capital.data.loaders import (  # noqa: F401
    data_version,
    get_close_matrix,
    get_data_coverage,
    get_daily_weightings_history,
    get_eod_matrix,
    get_eod_prices,
    get_fred_series,
    get_fx_rates,
    get_fundamentals,
    get_fundamentals_asof,
    get_fundamentals_columns,
    get_fundamentals_matrix,
    get_market_ohlcv,
    get_market_prices,
    get_portfolio_and_benchmarks,
    get_returns_matrix,
    get_security_master,
    get_theme_mappings,
    get_trade_log,
    get_universe_snapshot,
)
