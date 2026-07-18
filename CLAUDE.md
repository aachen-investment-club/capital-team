# Portfolio Analytics Dashboard

Dash (Plotly) dashboard for the AIC live-money portfolio: returns, factor models,
risk analytics, volatility forecasting and market monitors. Teammates extend it by
adding pages in plain Python. Live at https://portfolio.aachen-investment-club.de.

## Golden rules

1. **The dashboard never writes to S3 or DynamoDB.** The only writers are the
   nightly `capital-ingest` job (local DuckDB store + the S3 `backup/` prefix)
   and the external IBKR Lambda (below).
2. **The `fund-data-ingestion` (IBKR) Lambda is a separate project — never touch
   it.** It feeds another club website via S3 `history/portfolio/*` JSONs and
   DynamoDB `fund-data`. This repo only *reads* those outputs.

## Stack

- Dash + dash-mantine-components — UI (gunicorn-served, multipage via Dash Pages)
- DuckDB — one local database file (`data/market.duckdb`) holding EOD prices,
  fundamentals, market data, FRED series and derived tables
- Plotly — all charts, via the `capital` template in `capital/theme.py`
- uv — Python env and lockfile

## Architecture

- `capital-ingest nightly` (systemd timer, 03:15 Berlin, Tue–Sat) fetches LSEG
  EOD + fundamentals, yfinance market data and FRED series into the DuckDB store,
  rebuilds derived tables (`daily_returns`, `rolling_stats`), bumps
  `meta.data_version`, and backs the DB file up to S3.
- The dashboard reads the store through cached loaders. Loader caches are keyed
  on `data_version`, so the nightly ingest invalidates everything at once.
- DuckDB concurrency contract: many read-only connections OR one writer.
  Loaders open short-lived read-only connections per query (`capital/data/store.py`);
  the ingest is the sole writer.

## Data contract (important)

All data access goes through `capital.data` (implemented in
`capital/data/loaders.py`). Pages and analytics MUST NOT import boto3 or duckdb,
or reference S3 paths / the DB file directly:

- Store loaders: `get_eod_prices(security_id)`, `get_fundamentals()`,
  `get_market_prices(ticker)`, `get_market_ohlcv(ticker)`, `get_fred_series(id)`
- Bulk loaders (use these for multi-security work — never loop `get_eod_prices`):
  `get_close_matrix()`, `get_returns_matrix()`, `get_universe_snapshot()`
- S3/DynamoDB reads (the IBKR Lambda's stable outputs):
  `get_portfolio_and_benchmarks()`, `get_daily_weightings_history()`,
  `get_trade_log()`, `get_theme_mappings()`
- `get_security_master()` — config/security_master.csv (S3 copy when deployed)

Add new loaders in `capital/data/loaders.py`, decorated with
`@cached_by_version`, never inside a page.

## Adding a page (extension pattern)

1. Copy `src/capital/dashboard/pages/_template.py` to `pages/<name>.py`
   (files starting with `_` are not registered).
2. Set `dash.register_page(__name__, path=…, name=…, order=…, description=…)` —
   the navbar and home page pick it up automatically.
3. `layout()` returns dash-mantine components; use the helpers in
   `capital/dashboard/components.py` (`page_title`, `section`, `graph`,
   `df_table`, `export_button`).
4. Callbacks take small inputs (ticker, dates) and return figures — never move
   DataFrames through the browser (no DataFrames in `dcc.Store`).
5. Heavy interactive math (optimiser, GARCH fits): `background=True` +
   `running=[(Output(btn, "loading"), True, False)]` — see `pages/barra.py`.
6. Charts pick up the `capital` Plotly template automatically.

Shared math lives in `capital/analytics/` — reuse before writing new code.

## Layout

```
src/capital/
  settings.py          # the only reader of env vars
  theme.py             # palette, plotly template, Mantine theme
  data/                # THE DATA CONTRACT: store.py, loaders.py, cache.py
  analytics/           # barra, weighting, volatility, correlation, trend, ...
  ingest/              # capital-ingest CLI: eod, fund, market, fred, derived, sync, nightly
  dashboard/           # app.py (gunicorn target), shell.py, components.py, pages/, assets/
config/security_master.csv   # the ingest universe (barra_universe column = Barra estimation set)
deploy/                      # systemd units, nginx conf, server-setup.sh
```

## Commands

- Run app (dev): `uv run python -m capital.dashboard.app` (http://127.0.0.1:8050)
- Serve (prod): `gunicorn -w 1 --threads 4 -b 127.0.0.1:8050 capital.dashboard.app:server`
- Ingest: `uv run capital-ingest {eod,fund,market,fred,derived,sync,nightly}`
  (`eod --start 2016-01-01` backfills history for newly added securities, or
  rebuilds the whole store from LSEG from scratch)
- Env vars (`.env`): `S3_BUCKET`, `AWS_REGION`, `DDB_TABLE`, `CAPITAL_DB`,
  `CAPITAL_CACHE`, `LSEG_APP_KEY/USERNAME/PASSWORD`, `FRED_API_KEY` (optional),
  `HEALTHCHECK_URL` (optional)

## Ops runbook

- **Add a security**: add a row to `config/security_master.csv` (set
  `barra_universe=true` to include it in the Barra estimation set), push to main
  (CSV syncs to S3), then `capital-ingest eod --start <date>` for history.
- **Rerun a failed night**: `sudo systemctl start capital-ingest` (idempotent —
  PK upserts). Check `journalctl -u capital-ingest`.
- **LSEG session**: one concurrent platform session per credential;
  `signon_control=True` kicks other sessions. Don't run manual LSEG scripts
  while the nightly job is due.
- **Restore the store**: `aws s3 cp s3://<bucket>/backup/market.duckdb data/`
  (weekly dated copies under `backup/weekly/`). The DuckDB backup is the only
  copy of full history — if it is ever lost too, rebuild from LSEG with
  `capital-ingest eod --start <inception>` then `capital-ingest fund/market/fred/derived`.
