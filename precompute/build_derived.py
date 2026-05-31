"""
Nightly precompute job: reads raw Parquet, writes derived Parquet.
This is the ONLY component that writes files.

Run via:  python -m precompute.build_derived
"""
import os
import pathlib

import duckdb
import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RAW_DIR = pathlib.Path(os.getenv("RAW_PREFIX", str(_ROOT / "data" / "raw")))
_DERIVED_DIR = pathlib.Path(os.getenv("DERIVED_PREFIX", str(_ROOT / "data" / "derived")))


def _read(table: str) -> pd.DataFrame:
    return duckdb.execute(
        f"SELECT * FROM read_parquet('{_RAW_DIR / f'{table}.parquet'}')"
    ).df()


def _write(df: pd.DataFrame, table: str) -> None:
    _DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    path = _DERIVED_DIR / f"{table}.parquet"
    df.to_parquet(path, index=False)
    print(f"  wrote {path}  ({len(df):,} rows)")


def build_cumulative_returns(returns: pd.DataFrame) -> None:
    wide = returns.pivot(index="date", columns="ticker", values="daily_return")
    cum = (1 + wide).cumprod() - 1
    long = (
        cum.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="cumulative_return")
    )
    _write(long, "cumulative_returns")


def build_rolling_vol(returns: pd.DataFrame, window: int = 21) -> None:
    wide = returns.pivot(index="date", columns="ticker", values="daily_return")
    # annualise: daily σ × √252
    roll = wide.rolling(window).std() * (252 ** 0.5)
    long = (
        roll.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="rolling_vol_21d")
        .dropna()
    )
    _write(long, "rolling_vol")


def build_daily_weightings(
    daily_positions: pd.DataFrame,
    position_returns: pd.DataFrame,
    instrument_metadata: pd.DataFrame,
) -> None:
    """Join daily positions with metadata and compute per-symbol since-inception cumulative returns."""
    wide = position_returns.pivot(index="date", columns="symbol", values="daily_return")
    cum = ((1 + wide).cumprod() - 1).reset_index().melt(
        id_vars="date", var_name="symbol", value_name="cumulative_return"
    )
    df = (
        daily_positions
        .merge(cum, on=["date", "symbol"], how="left")
        .merge(
            instrument_metadata[["symbol", "name", "isin", "ccy", "category"]],
            on="symbol", how="left",
        )
    )
    _write(
        df[["date", "symbol", "name", "isin", "ccy", "category", "pct_nav", "cumulative_return"]],
        "daily_weightings",
    )


def build_factor_beta_history(betas: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Placeholder: walks current betas through history with small random noise."""
    factor_cols = ["market_beta", "value_beta", "momentum_beta", "quality_beta"]
    dates = pd.DataFrame({"date": sorted(returns["date"].unique()), "_k": 1})
    cross = dates.merge(betas.assign(_k=1), on="_k").drop(columns="_k")
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, size=(len(cross), len(factor_cols)))
    cross[factor_cols] = cross[factor_cols].values + noise
    _write(cross[["date", "ticker"] + factor_cols], "factor_beta_history")


def main() -> None:
    print("Loading raw data …")
    returns = _read("returns")
    betas = _read("factor_betas")
    daily_positions = _read("daily_positions")
    position_returns = _read("position_returns")
    instrument_metadata = _read("instrument_metadata")

    print("Computing derived tables:")
    build_cumulative_returns(returns)
    build_rolling_vol(returns)
    build_factor_beta_history(betas, returns)
    build_daily_weightings(daily_positions, position_returns, instrument_metadata)
    print("Done.")


if __name__ == "__main__":
    main()
