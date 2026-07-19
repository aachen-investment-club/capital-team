# Capital Team — Portfolio Analytics Dashboard

Dash (Plotly) dashboard for the AIC live-money portfolio: NAV vs benchmarks,
position weights, trade log, factor models (Barra), portfolio optimisation,
volatility forecasting, correlation/credit/liquidity monitors, and regime
detection — built on a local DuckDB store of daily OHLCV for the whole universe.

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
│   ├── analytics/             # barra, weighting, volatility, correlation, trend, ...
│   ├── ingest/                # capital-ingest CLI (see below)
│   └── dashboard/
│       ├── app.py             # Dash app factory — gunicorn target: capital.dashboard.app:server
│       ├── shell.py           # AppShell: header + navbar from the page registry
│       ├── components.py      # shared page building blocks
│       ├── pages/             # one file per page; _template.py = copy-paste pattern
│       └── assets/            # capital.css, logos
├── config/security_master.csv # the universe (barra_universe column = Barra estimation set)
├── scripts/                   # lookup_rics.py, IBKR position backfills
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
- Heavy interactive math (optimiser, GARCH/BOCPD fits) runs as Dash
  **background callbacks**; everything input-independent is precomputed nightly.

## The ingest CLI

```bash
uv run capital-ingest nightly              # full pipeline with healthcheck pings
uv run capital-ingest eod --start 2016-01-01   # backfill new securities
uv run capital-ingest fund | market | fred | derived | sync
```

Idempotent (primary-key upserts) — safe to rerun any night.

## Adding a page

1. Copy `src/capital/dashboard/pages/_template.py` to `pages/<name>.py`.
2. Adjust `dash.register_page(...)` — navbar + home card appear automatically.
3. Load data only via `capital.data`, math via `capital.analytics`,
   UI helpers from `capital.dashboard.components`.
4. Heavy math → `background=True` callback (see `pages/barra.py`).

## Adding a security

1. `uv run python scripts/lookup_rics.py` to find the LSEG RIC.
2. Add a row to `config/security_master.csv` (unique `security_id`, never reuse;
   `barra_universe=true` to include it in the Barra estimation set).
3. Push to main (deploys via git pull), then `capital-ingest eod --start <date>`.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `S3_BUCKET` | _(empty)_ | bucket for portfolio JSONs + backups (empty = fully local) |
| `AWS_REGION` | `eu-central-1` | AWS region |
| `DDB_TABLE` | `fund-baskets` | DynamoDB table for theme mappings |
| `CAPITAL_DB` | `./data/market.duckdb` | local DuckDB store |
| `CAPITAL_CACHE` | `./data/cache` | loader + background-callback cache dir |
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
