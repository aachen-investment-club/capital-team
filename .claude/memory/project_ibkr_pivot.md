---
name: project-ibkr-pivot
description: Complete pivot to IBKR-only data infrastructure for the Performance page — no yFinance, pure Flex Query XML
metadata:
  type: project
---

Portfolio data infrastructure was rebuilt from scratch to use only IBKR Flex Query XML exports.

**Why:** IBKR reporting tools are the authoritative source; yFinance was unreliable for this portfolio's instruments.

**Data flow:**
- `data/ibkr/trade_log.xml` → `trade_log.parquet` (parsed once, updated when new trades occur)
- `data/ibkr/prior_positions.xml` → `prior_positions.parquet` (historical backfill May 8–29 2026)
- `data/ibkr/open_positions/YYYYMMDD.xml` → `open_positions.parquet` (appended daily)
- Run `scripts/ingest_ibkr_xml.py` after adding a new XML file

**Key module:** `lib/ibkr.py` — XML parsers + `build_daily_positions()` + `compute_weightings()`
- `build_daily_positions()`: merges prior_positions (with shares from trade log) + open_positions, forward-fills gaps (exchange holidays)
- `compute_weightings()`: derives pct_nav, daily_return, cumulative_return (from inception, vs IBKR costBasisPrice)

**Data contract:** `lib/data.py` exposes `get_daily_weightings()`, `get_open_positions()`, `get_trade_log()`

**Performance page sections (01_Performance.py):**
1. Portfolio Weightings — pie chart (by position / by asset class) + holdings table grouped by COMMON/ETF
2. Single Positions Performance — rebased period return line chart + summary table
3. Trade Log — monthly equity trades only (CASH/FX trades excluded)

**Removed:** benchmark comparison section (no external data source), DynamoDB theme mappings, factor beta tables, all legacy precompute functions

**Daily extension:** drop new `YYYYMMDD.xml` in `data/ibkr/open_positions/`, re-run ingest script.

**How to apply:** When user asks about adding new position data or extending the history, refer to this workflow.
