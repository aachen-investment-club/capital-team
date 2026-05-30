# Capital Team — Portfolio Analytics

Read-only Streamlit dashboard for our live portfolio: returns, factor models, risk analytics, and volatility forecasting and economic indicators.

## Quick start

```bash
# Install dependencies
uv sync

# Generate placeholder data (first time only)
uv run python scripts/generate_placeholder_data.py

# Build derived tables
uv run python -m precompute.build_derived

# Launch the dashboard
uv run streamlit run app/Home.py
```

Then open http://localhost:8501.

## Project layout

```
capital-team/
├── app/
│   ├── Home.py                        # Landing page — summary metrics
│   └── pages/
│       └── 01_portfolio_visualizer.py # Weights · returns · factor betas · trade log
│
├── lib/
│   ├── data.py                        # THE DATA CONTRACT — all DuckDB loaders live here
│   ├── metrics.py                     # Shared analytics (cumulative returns, VaR, …)
│   └── theme.py                       # Plotly "capital" template + PNG export config
│
├── precompute/
│   └── build_derived.py               # Nightly job: reads raw → writes derived Parquet
│
├── scripts/
│   └── generate_placeholder_data.py   # One-time: creates ./data/raw/ with sample data
│
├── data/                              # gitignored — generated locally or pulled from S3
│   ├── raw/                           # returns, positions, factor_betas, trade_log
│   └── derived/                       # cumulative_returns, rolling_vol, factor_beta_history
│
├── pyproject.toml
├── uv.lock
└── .env.example
```

## Architecture

The dashboard is read-only and does no heavy computation at request time.

- A nightly precompute job (`precompute/build_derived.py`) computes input-independent results and writes them as Parquet.
- The app reads those tables via DuckDB and caches results in memory.
- Only input-dependent work (what-if scenarios, Monte Carlo) runs live, always cached on its parameters.

**Golden rule:** pages import from `lib/` only. They never touch DuckDB, boto3, or file paths directly.

## Data contract

All data access goes through `lib/data.py`:

| Loader | Returns |
|---|---|
| `get_returns()` | Daily returns per ticker |
| `get_positions()` | Current holdings and weights |
| `get_factor_betas()` | Current factor exposures |
| `get_trade_log()` | Recent trade history |
| `get_cumulative_returns()` | Precomputed cumulative returns |
| `get_rolling_vol()` | Precomputed 21-day rolling volatility |

Add new loaders here, never inside a page.

## Adding a page

1. Create `app/pages/NN_name.py`
2. Import data via `from lib.data import …` and analytics from `lib.metrics`
3. Build charts with `lib.theme.PNG_CONFIG` for consistent styling and one-click PNG export
4. Wrap any expensive computation in `@st.cache_data`

Streamlit auto-registers the file in the sidebar — no routing or config needed.

## Configuration

Copy `.env.example` to `.env` and fill in values to connect to AWS.
Leave `S3_BUCKET` empty to run fully offline against `./data/`.

| Variable | Default | Purpose |
|---|---|---|
| `S3_BUCKET` | _(empty)_ | S3 bucket name; empty = use local `./data/` |
| `RAW_PREFIX` | `data/raw` | Path/prefix for raw Parquet |
| `DERIVED_PREFIX` | `data/derived` | Path/prefix for derived Parquet |
| `DDB_TABLE` | `portfolio-positions` | DynamoDB table for positions |
| `AWS_REGION` | `eu-central-1` | AWS region |

## Commands

| Command | What it does |
|---|---|
| `uv run streamlit run app/Home.py` | Launch dashboard |
| `uv run python -m precompute.build_derived` | Rebuild derived Parquet tables |
| `uv run python scripts/generate_placeholder_data.py` | Regenerate sample data |
