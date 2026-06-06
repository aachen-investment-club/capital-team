# Portfolio Analytics Dashboard

Read-only Streamlit dashboard for a live-money portfolio: returns, factor models,
risk analytics, and (later) volatility forecasting and economic indicators.
Teammates extend it by adding pages in plain Python.

## Golden rule: READ-ONLY

This app never writes to S3 or DynamoDB. No `put_object`, no `put_item`, no writes
of any kind to the production stores. The instance IAM role grants read access only —
assume any write will fail and must never be attempted. The single exception is the
precompute job, which writes derived tables to the derived S3 prefix and nothing else.

## Stack

- Streamlit (multipage) — UI
- DuckDB over Parquet on S3 — query engine; never load full datasets into pandas
- DynamoDB — positions / metadata, read-only
- Plotly — all charts, with one-click PNG export

## Architecture

The dashboard does no heavy computation at request time.

- A nightly precompute job (`precompute/build_derived.py`) computes input-independent
  results (cumulative returns, rolling stats, factor-beta history) and writes them as
  derived Parquet to S3.
- The app reads those precomputed tables via DuckDB and caches them.
- Only input-dependent work (what-if scenarios, Monte Carlo, on-the-fly factor fits)
  runs live — and is always cached on its parameters.

## Data contract (important)

All data access goes through `lib/data.py`. Pages and analytics MUST NOT touch boto3,
DuckDB, or S3 paths directly. Use the cached loaders, which return ready DataFrames:

- `get_returns()`, `get_positions()`, `get_factor_betas()`

Add new data loaders here, never inside a page.

## Caching

- `@st.cache_resource` for the DuckDB connection and reference data (once per server).
- `@st.cache_data` on every expensive computation, keyed on its parameters.
- Loaders use a TTL or key on the S3 object ETag so they refresh when new data lands.

## Charts

Use Plotly with the shared template in `lib/theme.py`. Apply it via the theme helper
and use its PNG export config — do not restyle per chart. The template is what keeps
output consistent and report-ready.

## Adding a page (extension pattern)

1. Create `app/pages/NN_name.py`.
2. Import data via `from lib.data import ...` and helpers from `lib.metrics`.
3. Build Plotly charts using `lib.theme`.
4. Wrap any expensive computation in `@st.cache_data`.

Streamlit auto-registers the page — no routing or config needed.

## Layout

```
app/Home.py
app/pages/            # one file per analysis tab
lib/data.py           # the data contract — cached DuckDB loaders
lib/theme.py          # Plotly template + PNG export config
lib/metrics.py        # shared analytics
precompute/build_derived.py
```

## Commands

- Run app: `streamlit run app/Home.py`
- Precompute: `python -m precompute.build_derived`
- Config via env vars: `S3_BUCKET`, `RAW_PREFIX`, `DERIVED_PREFIX`, `DDB_TABLE`, `AWS_REGION`