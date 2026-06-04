# Capital Team — Portfolio Analytics Dashboard

Read-only Streamlit dashboard for a live-money portfolio: NAV vs benchmarks, position weights, individual position returns, trade log, and EOD price history per security.

## Quick start

```bash
uv sync
uv run streamlit run app/Home.py
```

Open http://localhost:8501. Leave `S3_BUCKET` empty in `.env` to run fully offline against local data.

---

## Project layout

```
capital-team/
├── app/
│   ├── Home.py                         # Landing page
│   └── pages/
│       ├── 01_Performance.py           # NAV · benchmarks · weights · positions · trade log
│       └── 02_Equities.py              # Per-security EOD candlestick chart + metrics
│
├── lib/
│   ├── data.py                         # THE DATA CONTRACT — all loaders live here
│   └── theme.py                        # Plotly "capital" template + brand CSS
│
├── lambda/
│   ├── fund-data-ingestion/            # Nightly: IBKR flex query → nav_history.json + S3
│   └── fund-eod-ingestion/             # Nightly: LSEG batch EOD fetch → Parquet on S3
│
├── scripts/
│   ├── backfill_positions_to_s3.py     # Parse IBKR XML → build + upload portfolio JSON to S3
│   ├── ingest_eod.py                   # Backfill + incremental EOD prices via LSEG
│   └── lookup_rics.py                  # Find valid LSEG RICs from ISINs in security_master.csv
│
├── config/
│   └── security_master.csv             # EOD universe — add rows to extend, no code changes
│
├── data/                               # gitignored
│   └── ibkr/                           # Source XML files (irreplaceable)
│       ├── prior_positions.xml
│       ├── trade_log.xml
│       └── open_positions/YYYYMMDD.xml
│
├── pyproject.toml
└── .env.example
```

---

## Architecture

The dashboard is read-only and does no computation at request time.

```
IBKR Flex Query XML (data/ibkr/)
        │
        ▼
scripts/backfill_positions_to_s3.py          ← run once to rebuild from scratch
        │  writes to S3: history/portfolio/*.json + derived/*.json
        ▼
lambda/fund-data-ingestion  (nightly)
        │  appends to nav_history.json and other files on S3
        ▼
lib/data.py loaders  →  Streamlit pages

config/security_master.csv
        │
        ├──▶ lambda/fund-eod-ingestion  (nightly, automated)
        │           │  single LSEG batch call for all securities
        │           │  appends to history/eod_prices/security_id=.../data.parquet
        │
        └──▶ scripts/ingest_eod.py  (manual / backfill)
                    │  same Parquet layout, per-security incremental
        ▼
lib/data.py  get_eod_prices()  →  02_Equities.py
```

**Golden rule:** pages import from `lib/` only — never touch DuckDB, boto3, or file paths directly.

---

## S3 layout

```
s3://aic-fund-public-data/
└── history/
    ├── portfolio/
    │   ├── nav_history.json
    │   ├── equity_positions.json
    │   ├── fx_positions.json
    │   ├── trade_log.json
    │   └── derived/
    │       ├── portfolio_and_benchmarks.json
    │       └── daily_weightings.json
    └── raw/
        └── eod_prices/
            ├── security_id=SEC_001/data.parquet
            ├── security_id=SEC_002/data.parquet
            └── _version.json               ← written after each ingest run
```

---

## Data contract

All data access goes through `lib/data.py`:

| Loader | Source | Returns |
|---|---|---|
| `get_portfolio_and_benchmarks()` | `portfolio/derived/portfolio_and_benchmarks.json` | NAV + benchmark index values since inception |
| `get_daily_weightings_history()` | `portfolio/derived/daily_weightings.json` | Daily weights + returns for all positions |
| `get_trade_log()` | `portfolio/trade_log.json` | Trade history with same-day fills merged |
| `get_theme_mappings()` | DynamoDB `fund-baskets` | Basket/theme per symbol (optional) |
| `get_eod_prices(security_id, cache_version)` | `raw/eod_prices/security_id=.../data.parquet` | Daily OHLCV for one security |
| `get_security_master()` | `config/security_master.csv` | Active EOD universe |

Add new loaders to `lib/data.py`, never inside a page.

---

## Configuration

Copy `.env.example` to `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `S3_BUCKET` | _(empty)_ | S3 bucket — empty = use local `./data/` |
| `AWS_REGION` | `eu-central-1` | AWS region |
| `DDB_TABLE` | `fund-baskets` | DynamoDB table for theme/basket mappings |
| `LSEG_APP_KEY` | — | App key for LSEG Data Library (EOD ingestion only) |
| `LSEG_SESSION_TYPE` | `platform` | `platform` (RDP) or `desktop` (Eikon/Workspace) |

---

## Commands

| Command | What it does |
|---|---|
| `uv run streamlit run app/Home.py` | Launch dashboard |
| `S3_BUCKET=aic-fund-public-data uv run python scripts/backfill_positions_to_s3.py` | Rebuild all portfolio JSON from IBKR XML and push to S3 |
| `uv run python scripts/ingest_eod.py` | Full backfill / incremental EOD price update |
| `uv run python scripts/ingest_eod.py --dry-run` | Preview what would be fetched |
| `uv run python scripts/ingest_eod.py --retry-failures` | Retry securities from last failed run |
| `uv run python scripts/lookup_rics.py` | Find correct LSEG RICs from ISINs in security_master.csv |

---

## Deploying fund-eod-ingestion Lambda

```bash
# 1. Upload the security master to S3 (Lambda reads it from there)
aws s3 cp config/security_master.csv s3://aic-fund-public-data/config/security_master.csv

# 2. Build and push the container image
cd lambda/fund-eod-ingestion
aws ecr create-repository --repository-name fund-eod-ingestion --region eu-central-1  # first time only
ECR=<account-id>.dkr.ecr.eu-central-1.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin $ECR
docker build -t fund-eod-ingestion .
docker tag fund-eod-ingestion:latest $ECR/fund-eod-ingestion:latest
docker push $ECR/fund-eod-ingestion:latest

# 3. Set Lambda env vars in the AWS console:
#    S3_BUCKET, AWS_REGION, LSEG_APP_KEY, LSEG_USERNAME, LSEG_PASSWORD
#    (see .env.example for details)

# 4. Schedule with EventBridge — daily after market close, e.g.:
#    cron(0 21 ? * MON-FRI *)   ← 21:00 UTC / 22:00 CET on weekdays
```

When `config/security_master.csv` changes (new security added), re-upload to S3 — no container rebuild needed.

---

## EOD universe

Securities are defined in `config/security_master.csv`. To add a new security:

1. Run `scripts/lookup_rics.py` to find the correct LSEG RIC for the ISIN
2. Add a row to `config/security_master.csv` — `security_id` must be unique and never reused
3. Run `scripts/ingest_eod.py` — only the new security is fetched (existing ones are skipped)

---

## Rebuild from scratch

If S3 data is lost or you need to rebuild everything from the source XML files:

```bash
# 1. Rebuild portfolio JSON from IBKR XML and push to S3
S3_BUCKET=aic-fund-public-data uv run python scripts/backfill_positions_to_s3.py

# 2. Re-fetch EOD prices
uv run python scripts/ingest_eod.py
```

The three XML files under `data/ibkr/` are the only irreplaceable source data.

---

## Adding a page

1. Create `app/pages/NN_name.py`
2. Import data via `from lib.data import …`
3. Build charts with `template="capital"` (set globally by `inject_css()`)
4. Wrap expensive computation in `@st.cache_data`

Streamlit auto-registers the file in the sidebar — no routing or config needed.
