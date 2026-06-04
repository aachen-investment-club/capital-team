# Capital Team — Portfolio Analytics Dashboard

Read-only Streamlit dashboard for a live-money portfolio.
Displays NAV performance vs benchmarks, position weights, individual position returns, and the trade log — all sourced purely from IBKR Flex Query data with no external price feeds.

## Quick start (local)

```bash
# Install dependencies
uv sync

# Parse IBKR XML source files into local Parquet
uv run python scripts/ingest_ibkr_xml.py

# Build derived tables (daily_weightings + portfolio_and_benchmarks)
uv run python -m precompute.build_derived

# Launch the dashboard
uv run streamlit run app/Home.py
```

Then open http://localhost:8501.

---

## Project layout

```
capital-team/
├── app/
│   ├── Home.py                        # Landing page
│   └── pages/
│       └── 01_Performance.py          # NAV · benchmarks · weights · positions · trade log
│
├── lib/
│   ├── data.py                        # THE DATA CONTRACT — all loaders live here
│   ├── ibkr.py                        # IBKR XML parsers + position computation
│   ├── metrics.py                     # Shared analytics helpers
│   └── theme.py                       # Plotly "capital" template + PNG export config
│
├── precompute/
│   └── build_derived.py               # Nightly job: builds portfolio_and_benchmarks
│                                      #              and daily_weightings from IBKR data
│
├── lambda/
│   └── fund-data-ingestion/           # Fetches IBKR flex query → DynamoDB + S3
│
├── scripts/
│   ├── ingest_ibkr_xml.py             # Parse local XML → data/ibkr/*.parquet
│   └── backfill_positions_to_s3.py    # One-time: push historical positions to S3
│
├── data/                              # gitignored — generated locally or synced from S3
│   ├── ibkr/                          # Source: XML files + parsed Parquet
│   │   ├── trade_log.xml              # IBKR trade log flex query export
│   │   ├── prior_positions.xml        # IBKR historical positions flex query export
│   │   ├── open_positions/
│   │   │   └── YYYYMMDD.xml          # Daily open positions + FX positions flex query
│   │   ├── trade_log.parquet
│   │   ├── prior_positions.parquet
│   │   ├── open_positions.parquet
│   │   └── fx_positions.parquet
│   ├── derived/                       # Built by precompute/build_derived.py
│   │   ├── daily_weightings.parquet
│   │   └── portfolio_and_benchmarks.parquet
│   └── nav_history.json               # Downloaded from S3 for local dev
│
├── pyproject.toml
├── uv.lock
└── .env.example
```

---

## Architecture

The dashboard is read-only and does no heavy computation at request time.

```
IBKR Flex Query XML
        │
        ▼
scripts/ingest_ibkr_xml.py          ← run when new XML is available
        │  writes data/ibkr/*.parquet
        ▼
precompute/build_derived.py         ← run nightly (or manually)
        │  reads ibkr Parquet + nav_history.json
        │  writes daily_weightings.parquet
        │          portfolio_and_benchmarks.parquet
        ▼
lib/data.py loaders  →  Streamlit pages
```

In production, the Lambda runs nightly and a manual precompute step keeps the derived tables fresh:

```
fund-data-ingestion lambda  (nightly, IBKR flex query)
    → nav_history.json (S3)
    → DynamoDB POSITIONS / METRICS
    → daily_equity_positions/date=.../data.csv  (S3)
    → daily_fx_positions/date=.../data.csv      (S3)

precompute/build_derived.py  (run manually after new XML is ingested)
    → history/derived/daily_weightings.parquet  (S3)
    → history/derived/portfolio_and_benchmarks.parquet  (S3)
```

> **Note:** A dedicated Lambda to trigger `build_derived` automatically is planned but not yet set up.

**Golden rule:** pages import from `lib/` only — never touch DuckDB, boto3, or file paths directly.

---

## Data contract

All data access goes through `lib/data.py`:

| Loader | Returns |
|---|---|
| `get_daily_weightings_history()` | Daily weights + returns for all positions including cash |
| `get_portfolio_and_benchmarks()` | NAV multiple + benchmark index values since inception |
| `get_theme_mappings()` | Basket/theme per symbol (DynamoDB or category fallback) |
| `get_trade_log()` | Trade history with same-day fills merged per position |

Add new loaders here, never inside a page.

---

## Daily workflow

### Adding today's open positions (manual)

1. Run the IBKR **open-positions** flex query, save the XML to:
   ```
   data/ibkr/open_positions/YYYYMMDD.xml
   ```
2. Ingest and rebuild:
   ```bash
   uv run python scripts/ingest_ibkr_xml.py
   uv run python -m precompute.build_derived
   ```

### Production (automated via Lambda)

```bash
# Invoke the ingestion lambda manually
aws lambda invoke \
  --function-name fund-data-ingestion \
  --region eu-central-1 \
  --log-type Tail \
  /tmp/response.json \
  --query 'LogResult' \
  --output text | base64 -d
```

---

## Configuration

Copy `.env.example` to `.env`. Leave `S3_BUCKET` empty to run fully offline.

| Variable | Default | Purpose |
|---|---|---|
| `S3_BUCKET` | _(empty)_ | S3 bucket; empty = use local `./data/` |
| `RAW_PREFIX` | `history/raw` | S3 prefix for raw Parquet/CSV |
| `DERIVED_PREFIX` | `history/derived` | S3 prefix for derived Parquet |
| `DDB_TABLE` | `fund-baskets` | DynamoDB table for theme/basket mappings |
| `AWS_REGION` | `eu-central-1` | AWS region |

---

## Commands reference

| Command | What it does |
|---|---|
| `uv run streamlit run app/Home.py` | Launch dashboard |
| `uv run python scripts/ingest_ibkr_xml.py` | Parse XML → local IBKR Parquet files |
| `uv run python -m precompute.build_derived` | Rebuild derived tables (local) |
| `S3_BUCKET=aic-fund-public-data uv run python -m precompute.build_derived` | Rebuild and push to S3 |
| `S3_BUCKET=aic-fund-public-data uv run python scripts/backfill_positions_to_s3.py` | Backfill historical positions to S3 |
| `aws s3 cp data/nav_history.json . ← s3://aic-fund-public-data/history/nav_history.json` | Download nav history for local dev |
| `aws s3 sync data/derived/ s3://aic-fund-public-data/history/derived/` | Push derived tables to S3 |

---

## Backfill / recovery from scratch

If S3 data is lost or you need to rebuild from the source XML files:

```bash
# 1. Rebuild local Parquet from XML
uv run python scripts/ingest_ibkr_xml.py

# 2. Push historical positions to S3
S3_BUCKET=aic-fund-public-data uv run python scripts/backfill_positions_to_s3.py

# 3. Rebuild and push derived tables
S3_BUCKET=aic-fund-public-data \
RAW_PREFIX=history/raw \
DERIVED_PREFIX=history/derived \
  uv run python -m precompute.build_derived
```

The three XML files under `data/ibkr/` are the only irreplaceable source data — everything else can be regenerated from them.

## Adding a page

1. Create `app/pages/NN_name.py`
2. Import data via `from lib.data import …` and helpers from `lib.metrics`
3. Build charts using `lib.theme` — `template="capital"` and `PNG_CONFIG` for export
4. Wrap expensive computation in `@st.cache_data`

Streamlit auto-registers the file in the sidebar — no routing or config needed.
