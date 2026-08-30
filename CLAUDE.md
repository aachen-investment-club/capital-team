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
  rebuilds derived tables (`daily_returns`, `rolling_stats`), estimates the
  nightly factor model, bumps `meta.data_version`, and backs the DB file up to S3.
- The dashboard reads the store through cached loaders. Loader caches are keyed
  on `data_version`, so the nightly ingest invalidates everything at once.
- DuckDB concurrency contract: many read-only connections OR one writer, and the
  write lock is **file-exclusive** — while it is held, no other process can open
  the database even read-only, so the dashboard goes dark. Loaders open
  short-lived read-only connections per query; the ingest is the sole writer and
  commits **per batch** via `store.flush()` rather than holding one connection
  across a run. That is what makes a multi-hour backfill survivable.
- **Background jobs** (`capital/jobs/`): long computations are queued as jobs, not
  run in callbacks. A job is a JSON file under `CAPITAL_CACHE/jobs/`; a pump
  thread in the app starts each one as a *subprocess* (`python -m
  capital.jobs.runner <id>`), so heavy numpy/pandas work never contends for the
  dashboard's GIL and a crash cannot take the app down. Results are written to a
  durable store, so a run survives reloads and is visible to every user.

## Data contract (important)

All data access goes through `capital.data` (implemented in
`capital/data/loaders.py`). Pages and analytics MUST NOT import boto3 or duckdb,
or reference S3 paths / the DB file directly:

- Store loaders: `get_eod_prices(security_id)`, `get_fundamentals()`,
  `get_market_prices(ticker)`, `get_market_ohlcv(ticker)`, `get_fred_series(id)`
- Bulk loaders (use these for multi-security work — never loop `get_eod_prices`):
  `get_close_matrix()`, `get_returns_matrix()`, `get_universe_snapshot()`,
  `get_eod_matrix(field)`, `get_fundamentals_matrix(field)`,
  `get_fundamentals_asof(dates, columns)` (point-in-time ASOF join),
  `get_fundamentals_columns()` (which descriptor columns the store actually has),
  `get_fx_rates(base)`
- S3/DynamoDB reads (the IBKR Lambda's stable outputs):
  `get_portfolio_and_benchmarks()`, `get_daily_weightings_history()`,
  `get_trade_log()`, `get_theme_mappings()`
- `get_security_master()` — config/security_master.csv (git-tracked; read from
  the local checkout, not S3)
- Factor-model runs: `capital.data.factor_store` (`list_runs`, `load_manifest`,
  `load_factor_returns`, `load_exposures`, `exposure_snapshot`, …). Runs are
  immutable parquet directories under `CAPITAL_CACHE/factor_models/`, written by
  the job runner and never by the DuckDB writer.

Add new loaders in `capital/data/loaders.py`, decorated with
`@cached_by_version`, never inside a page.

## Factor model

`capital/analytics/factors/` is a Barra USE4-style cross-sectional model built to
run over 1–2k securities. Layers, in pipeline order:

- `spec.py`: `ModelSpec` (one run's configuration) plus the descriptor and
  style registries. **Data-driven, and deliberately without presets**: a run
  asks for every style and the whole security master, then descriptors whose
  store column is missing or has no cross-sectional variation are dropped, the
  style's remaining descriptor weights are renormalised, and a style with
  nothing left is dropped and reported. Growing `config/security_master.csv`
  or back-filling a new fundamentals column therefore grows the model with no
  code or spec edit. `max_securities=0` means no cap.
- `descriptors.py` — raw descriptors from whole wide matrices. Nothing here loops
  over securities; 2,000 names cost the same matrix ops as 20.
- `exposures.py` — robust scaling, cap-weighted standardisation (so the market
  portfolio has exposure 0 to every style), industry-mean fill, orthogonalisation.
- `regression.py` — constrained WLS per period, industry/country returns pinned
  cap-weighted zero-sum via a null-space reparameterisation, Huber IRLS.
  Also returns-based exposures for ETFs, which are deliberately excluded from the
  estimation universe.
- `risk.py` — two-half-life Newey-West factor covariance, Bayesian-shrunk specific
  risk, portfolio/active risk decomposition, what-if trades.
- `diagnostics.py` — IC and decay, quantile spreads, persistence, VIF, per-security fit.
- `model.py` — `run_factor_model(spec, progress)`, the single entry point.

Runs are queued from `/factor-screen` (job kind `factor_model`, see
`capital/jobs/handlers.py`) or built nightly by `capital-ingest factors`. Roughly
25s for 1,500 securities × 250 weekly cross-sections; ~33 MB per stored run.
The maths is documented for users in the screen's **Methodology** tab, with copy
shared from `capital/dashboard/factor_text.py`.

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
   `running=[(Output(btn, "loading"), True, False)]`, or queue a job
   (`capital.jobs`) — see `pages/factor_screen.py`.
6. Charts pick up the `capital` Plotly template automatically.

Shared math lives in `capital/analytics/` — reuse before writing new code.

## Layout

```
src/capital/
  settings.py          # the only reader of env vars
  theme.py             # palette, plotly template, Mantine theme
  data/                # THE DATA CONTRACT: store.py, loaders.py, cache.py
  analytics/           # weighting, volatility, correlation, trend, ...
  analytics/factors/   # cross-sectional factor model (spec, descriptors, exposures,
                       #   regression, risk, diagnostics, model)
  jobs/                # background job queue: queue.py, runner.py, handlers.py
  ingest/              # capital-ingest CLI: eod, fund, market, fred, derived,
                       #   factors, sync, nightly
  dashboard/           # app.py (gunicorn target), shell.py, components.py,
                       #   factor_text.py, pages/, assets/
config/security_master.csv   # the ingest universe (asset_type selects the
                             #   factor model's estimation set)
deploy/                      # systemd units, nginx conf, server-setup.sh
```

## Commands

- Run app (dev): `uv run python -m capital.dashboard.app` (http://127.0.0.1:8050)
- Serve (prod): `gunicorn -w 1 --threads 4 -b 127.0.0.1:8050 capital.dashboard.app:server`
- Ingest: `uv run capital-ingest {eod,fund,market,fred,derived,factors,sync,nightly}`
  (`eod --start 2016-01-01` backfills history for newly added securities, or
  rebuilds the whole store from LSEG from scratch)
- Which LSEG descriptor fields this account can actually fetch:
  `uv run capital-ingest fund --probe` (writes nothing)
- Estimate the factor model outside the nightly run:
  `uv run capital-ingest factors --years 5 --frequency W-FRI`
- Grow the universe: `uv run python scripts/expand_universe.py --chains europe us`
  (dry run; add `--write`), then back-fill with `capital-ingest eod --start …`
- Env vars (`.env`): `S3_BUCKET`, `AWS_REGION`, `DDB_TABLE`, `CAPITAL_DB`,
  `CAPITAL_CACHE`, `LSEG_APP_KEY/USERNAME/PASSWORD`, `FRED_API_KEY` (optional),
  `HEALTHCHECK_URL` (optional)

## LSEG entitlements (verified, not assumed)

Probed against this project's own credential — the results shape the ingest, so
re-check with `capital-ingest fund --probe` if the entitlement changes.

**Not entitled** (and therefore not used anywhere):
- Real-time *streaming*: `ld.discovery.Chain` and `TR.IndexConstituentRIC` both
  fail with `NotEntitled: PE(3134)`. Index chains are expanded through the
  **Data** service instead: `ld.get_data(universe=["0#.STOXX"], fields=[...])`
  returns every constituent *and* its reference data in one request.
- `TR.ISIN` and `TR.PriceCurrency` — access denied. New master rows therefore
  carry a blank ISIN (the RIC is the join key everywhere that matters) and derive
  currency from the RIC's exchange suffix.

**Chain sizes**: `0#.STOXX` 600 · `0#.SPX` 503 · `0#.NDX` 102 · `0#.FTSE` 100 ·
`0#.MDAXI` 50 · `0#.GDAXI` 40 — about 1,300 unique names, which is the target
range for the factor model.

**Descriptor fields**: all 13 fundamentals columns resolve. Because TR names vary
by entitlement and get renamed, `ingest/eod.py` holds a *candidate list* per store
column and resolves the first that returns data, caching the answer in
`meta.lseg_fields`. Two resolved names are not what they look like, and the
descriptor labels say so:
- `roe` -> `TR.ROEMean` is the **analyst consensus estimate**, not realised ROE.
- `debt_to_ev` -> `TR.TotalDebtToEV` is debt over **enterprise value**, not over
  assets (`TR.TotalDebtToTotalAssets` is not entitled).

**Response labels are decorated** — `TR.PriceToBVPerShare` comes back as "Price To
Book Value Per Share (Daily Time Series Ratio)". Columns are therefore mapped
**by position** (LSEG returns them in the order requested), with label matching
only as a logged fallback.

## Ops runbook

- **Add a security**: add a row to `config/security_master.csv`, push to main
  (deploys via git pull), then `capital-ingest eod --start <date>` for history.
  `asset_type` decides how the factor model treats it: `COMMON` goes into the
  cross-sectional estimation universe, `ETF` is priced off the finished model by
  time-series regression, `INDEX` is excluded entirely.
- **Grow the screening universe** (see "LSEG entitlements" below for what works):

  ```
  python scripts/expand_universe.py --chains world --limit 1500        # dry run
  python scripts/expand_universe.py --chains world --limit 1500 --write
  capital-ingest eod  --start 2016-01-01 --resume
  capital-ingest fund --start 2016-01-01 --resume
  capital-ingest derived && capital-ingest factors
  ```

  Budget roughly **3 seconds per security per decade** of EOD history: ~70
  minutes for 1,400 names over 10 years, plus a similar amount for fundamentals.
  Both steps are resumable (`--resume` skips securities already covered back to
  `--start`), so an interruption costs one query, not another hour. Run it
  off-hours and never while the nightly job is due — one concurrent LSEG platform
  session per credential, and `signon_control=True` kicks the other one.
- **A factor is missing from the model**: check the run's *Run details* tab. It
  names each dropped style and why. Almost always a fundamentals column that has
  not been back-filled yet — confirm the field resolves with `capital-ingest fund
  --probe`, then `capital-ingest fund --start <date> --resume`.
- **A factor job is stuck**: jobs live in `CAPITAL_CACHE/jobs/*.json` with logs
  under `jobs/logs/`. A job whose runner died is failed automatically once its
  heartbeat goes stale. Cancel from the page, or delete the JSON file.
- **Rerun a failed night**: `sudo systemctl start capital-ingest` (idempotent —
  PK upserts). Check `journalctl -u capital-ingest`.
- **LSEG session**: one concurrent platform session per credential;
  `signon_control=True` kicks other sessions. Don't run manual LSEG scripts
  while the nightly job is due.
- **Restore the store**: `aws s3 cp s3://<bucket>/backup/market.duckdb data/`
  (weekly dated copies under `backup/weekly/`). The DuckDB backup is the only
  copy of full history — if it is ever lost too, rebuild from LSEG with
  `capital-ingest eod --start <inception>` then `capital-ingest fund/market/fred/derived`.
