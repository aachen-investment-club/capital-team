# Capital Team — Portfolio Analytics Dashboard

Dash (Plotly) dashboard for the AIC live-money portfolio: NAV vs benchmarks,
position weights, trade log, a full cross-sectional factor model, portfolio
optimisation, volatility forecasting, correlation/credit/liquidity monitors, and
regime detection — built on a local DuckDB store of daily OHLCV for the whole
universe.

**Live URL:** https://portfolio.aachen-investment-club.de

## Quick start

```bash
uv sync --extra ingest
uv run python -m capital.dashboard.app     # http://127.0.0.1:8050
```

Seed the local data store first — copy the nightly DuckDB backup:

```bash
aws s3 cp s3://aic-fund-public-data/backup/market.duckdb data/market.duckdb
```

(Or, to rebuild from scratch off LSEG: `capital-ingest eod --start <inception>`
followed by `fund`, `market`, `fred`, `derived`.)

## Project layout

```
capital-team/
├── src/capital/
│   ├── settings.py            # the only reader of env vars
│   ├── theme.py               # brand palette, plotly "capital" template, Mantine theme
│   ├── data/                  # THE DATA CONTRACT
│   │   ├── store.py           #   DuckDB file access (read-only conns / sole-writer ingest)
│   │   ├── loaders.py         #   get_* loaders, all cached
│   │   └── cache.py           #   @cached_by_version (invalidated by each ingest)
│   │   └── factor_store.py    #   persisted factor-model runs (parquet, immutable)
│   ├── analytics/             # weighting, volatility, correlation, trend, ...
│   │   └── factors/           #   the cross-sectional factor model (see below)
│   ├── jobs/                  # background job queue (subprocess runners)
│   ├── ingest/                # capital-ingest CLI (see below)
│   └── dashboard/
│       ├── app.py             # Dash app factory — gunicorn target: capital.dashboard.app:server
│       ├── shell.py           # AppShell: header + navbar from the page registry
│       ├── components.py      # shared page building blocks
│       ├── factor_text.py     # explanatory copy shared by the factor pages
│       ├── pages/             # one file per page; _template.py = copy-paste pattern
│       └── assets/            # capital.css, logos
├── config/security_master.csv # the universe (asset_type selects the estimation set)
├── scripts/                   # lookup_rics.py, expand_universe.py, IBKR backfills
├── deploy/                    # systemd units, nginx conf, server-setup.sh, deploy.sh
└── lambda/fund-data-ingestion # EXTERNAL project (feeds the club website) — do not touch
```

## Architecture

```
LSEG (EOD + fundamentals)  yfinance (SPY/QQQ/...)  FRED (HY OAS)
        └──────────────┬──────────────┴──────────────┘
                       ▼
        capital-ingest nightly  (systemd timer, 03:15 Berlin, Tue–Sat)
                       │  upserts + derived tables + data_version bump
                       ▼
        data/market.duckdb  ──backup──▶  s3://…/backup/market.duckdb
                       │
                       ▼
        capital.data loaders (version-keyed cache)  ──▶  Dash pages

IBKR Lambda (external)  ──▶  s3://…/history/portfolio/*.json + DynamoDB
                       └──▶  read-only by the dashboard (NAV, weights, trades)
```

- The dashboard **never writes** to S3/DynamoDB; the ingest job is the sole
  writer of the local store and the S3 backup prefix.
- Everything input-independent is precomputed nightly. Interactive math that
  takes seconds runs synchronously in the callback; anything heavier is queued
  as a **background job** (below).

## Background jobs

Estimating the factor model over 1–2k securities takes tens of seconds — too long
for a callback, which would freeze the dashboard for everyone. Those runs are
**jobs**:

```
page submits ──▶ CAPITAL_CACHE/jobs/<id>.json ──▶ pump thread
                                                      │  spawns
                                                      ▼
                             python -m capital.jobs.runner <id>   (subprocess)
                                                      │  writes
                                                      ▼
                          CAPITAL_CACHE/factor_models/<run>/  (parquet + manifest)
```

A subprocess rather than a thread, so a pandas-heavy run cannot compete with the
dashboard for the GIL and a crash cannot take the app down. The queue is just a
directory of JSON files, so every gunicorn worker and the CLI see the same state,
a run survives a page reload, and finished runs are visible to every user.

## The ingest CLI

```bash
uv run capital-ingest nightly              # full pipeline with healthcheck pings
uv run capital-ingest eod --start 2016-01-01   # backfill new securities
uv run capital-ingest fund | market | fred | derived | factors | sync

uv run capital-ingest fund --probe         # resolve + cache which TR fields work here
uv run capital-ingest eod --start 2016-01-01 --resume   # resumable backfill
uv run capital-ingest factors --years 5    # estimate the factor model on demand
```

Idempotent (primary-key upserts) — safe to rerun any night. `factors` runs after
`derived` and before `sync`, so the nightly backup carries the same store the
model was estimated from, and the Factor Screen opens on a finished run each
morning.

## Adding a page

1. Copy `src/capital/dashboard/pages/_template.py` to `pages/<name>.py`.
2. Adjust `dash.register_page(...)` — navbar + home card appear automatically.
3. Load data only via `capital.data`, math via `capital.analytics`,
   UI helpers from `capital.dashboard.components`.
4. Heavy math: seconds → run it in the callback; tens of
   seconds or more → queue a job (see `pages/factor_screen.py`). Do not use
   `background=True` — dmc 2.8 props are not dill-picklable.

## Adding a security

1. `uv run python scripts/lookup_rics.py` to find the LSEG RIC.
2. Add a row to `config/security_master.csv` (unique `security_id`, never reuse).
   `asset_type` decides its role in the factor model: `COMMON` is estimated in
   the cross-section, `ETF` is priced off the finished model, `INDEX` is excluded.
3. Push to main (deploys via git pull), then `capital-ingest eod --start <date>`.

## Growing the universe

The factor model needs breadth. `scripts/expand_universe.py` expands LSEG index
chain RICs into `config/security_master.csv`:

```bash
uv run python scripts/expand_universe.py --list                     # presets
uv run python scripts/expand_universe.py --chains world --limit 1500        # dry run
uv run python scripts/expand_universe.py --chains world --limit 1500 --write
uv run python scripts/expand_universe.py --rics-file etfs.txt --write       # ETFs
```

Chain sizes verified against our credential: `0#.STOXX` 600, `0#.SPX` 503,
`0#.NDX` 102, `0#.FTSE` 100, `0#.MDAXI` 50, `0#.GDAXI` 40 — about 1,300 unique
names. Existing rows are never modified and a `.bak` is kept.

Then back-fill history:

```bash
uv run capital-ingest eod  --start 2016-01-01 --resume
uv run capital-ingest fund --start 2016-01-01 --resume
uv run capital-ingest derived && uv run capital-ingest factors
```

Budget roughly **3 seconds per security per decade**: ~70 minutes for 1,400 names
over 10 years of EOD, plus a similar amount for fundamentals. Both steps are
resumable — `--resume` skips securities whose stored history already reaches
`--start`, so an interruption costs one query rather than another hour — and both
commit per batch, so the dashboard stays up throughout. Run it off-hours and never
while the nightly job is due: LSEG allows one concurrent platform session per
credential and will kick the other one.

### What LSEG actually gives us

Probed rather than assumed; `capital-ingest fund --probe` re-checks it.

| | |
|---|---|
| Chain expansion | `ld.get_data(["0#.STOXX"], [...])` — the **Data** service. Streaming (`ld.discovery.Chain`, `TR.IndexConstituentRIC`) is not entitled (`PE(3134)`). |
| Reference data | `TR.RIC`, `TR.CommonName`, `TR.ExchangeCountryCode`, `TR.CompanyMarketCap`, `TR.GICSSector`, `TR.AssetCategory`. **`TR.ISIN` and `TR.PriceCurrency` are denied** — new rows carry a blank ISIN, and currency is derived from the RIC's exchange suffix. |
| Descriptors | All 13 fundamentals columns resolve. Each store column has a *candidate list* of TR names; the first that returns data wins and the choice is cached in `meta.lseg_fields`. |

Two resolved fields are not quite what their name suggests, and the factor
descriptions in the app say so: `roe` is `TR.ROEMean`, the **analyst consensus
estimate** rather than realised ROE, and `debt_to_ev` is debt over **enterprise
value** because debt-to-assets is not entitled.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `S3_BUCKET` | _(empty)_ | bucket for portfolio JSONs + backups (empty = fully local) |
| `AWS_REGION` | `eu-central-1` | AWS region |
| `DDB_TABLE` | `fund-baskets` | DynamoDB table for theme mappings |
| `CAPITAL_DB` | `./data/market.duckdb` | local DuckDB store |
| `CAPITAL_CACHE` | `./data/cache` | loader cache, job queue, factor-model runs |
| `CAPITAL_JOB_WORKERS` | `1` | concurrent background jobs per process |
| `LSEG_APP_KEY/USERNAME/PASSWORD` | — | LSEG RDP platform session (ingest only) |
| `FRED_API_KEY` | _(empty)_ | free key; without it FRED history caps at ~3y |
| `HEALTHCHECK_URL` | _(empty)_ | healthchecks.io ping URL for the nightly job |

## Production deployment

Single EC2 instance (t3.small, Ubuntu, eu-central-1) running gunicorn behind
nginx with Let's Encrypt TLS. `deploy/server-setup.sh` bootstraps a fresh box
(uv, swapfile, systemd units for dashboard + nightly ingest timer, nginx).

- **Auto-deploy:** push to `main` → GitHub Actions runs an import smoke test,
  syncs `config/*.csv` to S3, then SSHes in: `git pull`, `uv sync --frozen`,
  restart.
- **Manual deploy:** `cd /opt/capital-dashboard && bash deploy/deploy.sh`
- **Ops:** `journalctl -u capital-ingest` for the nightly job;
  `sudo systemctl start capital-ingest` to rerun a failed night.
