"""
capital-ingest — nightly data pipeline CLI (runs on the dashboard server).

Subcommands:
    eod       LSEG EOD OHLCV → store          (--start / --days for backfill)
    fund      LSEG fundamentals → store        (--start / --days)
    market    yfinance market tickers → store  (--days)
    fred      FRED series → store
    derived   rebuild derived tables (daily_returns, rolling_stats)
    sync      back up market.duckdb to S3
    nightly   eod → fund → market → fred → derived → sync, with healthcheck pings

The IBKR fund-data ingestion is a separate, independently maintained Lambda —
deliberately not part of this CLI.
"""
import argparse
import logging
import sys
import urllib.request

from capital.settings import settings

log = logging.getLogger("capital.ingest")


def _ping(suffix: str = "") -> None:
    """healthchecks.io ping — /start at begin, bare on success, /fail on failure."""
    if not settings.healthcheck_url:
        return
    try:
        urllib.request.urlopen(settings.healthcheck_url + suffix, timeout=10)
    except Exception as exc:
        log.warning("healthcheck ping%s failed: %s", suffix or " (success)", exc)


def _run_nightly(args) -> int:
    from capital.ingest import derived, eod, market, sync

    _ping("/start")
    failures: list[str] = []

    def step(name, fn, *fn_args, **fn_kwargs):
        log.info("── %s ──", name)
        try:
            result = fn(*fn_args, **fn_kwargs)
            log.info("%s: %s", name, result)
            return result
        except Exception as exc:
            log.error("%s FAILED: %s", name, exc, exc_info=True)
            failures.append(name)
            return None

    eod.open_lseg_session()
    try:
        universe = eod.load_universe()
        eod_result = step("eod", eod.run_eod, universe=universe)
        step("fundamentals", eod.run_fundamentals, universe=universe)
    finally:
        eod.close_lseg_session()

    step("market", market.run_market)
    step("fred", market.run_fred)
    step("derived", derived.run_derived)
    step("sync", sync.run_sync)

    # EOD partial failures count: tolerate a few unresolved RICs, not a wipeout
    if eod_result and eod_result.get("failed", 0) > eod_result.get("succeeded", 0):
        failures.append(f"eod ({eod_result['failed']} RICs failed)")

    if failures:
        log.error("nightly finished with failures: %s", failures)
        _ping("/fail")
        return 1
    log.info("nightly OK")
    _ping()
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(prog="capital-ingest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_eod = sub.add_parser("eod", help="LSEG EOD OHLCV → store")
    p_eod.add_argument("--start", help="ISO date — backfill from this date")
    p_eod.add_argument("--days", type=int, help="lookback days (default from env)")

    p_fund = sub.add_parser("fund", help="LSEG fundamentals → store")
    p_fund.add_argument("--start")
    p_fund.add_argument("--days", type=int)

    p_market = sub.add_parser("market", help="yfinance market tickers → store")
    p_market.add_argument("--days", type=int)

    sub.add_parser("fred", help="FRED series → store")
    sub.add_parser("derived", help="rebuild derived tables")
    sub.add_parser("sync", help="back up market.duckdb to S3")
    sub.add_parser("nightly", help="full pipeline with healthcheck pings")

    args = ap.parse_args(argv)

    if args.cmd == "nightly":
        return _run_nightly(args)

    if args.cmd in ("eod", "fund"):
        from capital.ingest import eod as eod_mod
        eod_mod.open_lseg_session()
        try:
            if args.cmd == "eod":
                result = eod_mod.run_eod(start=args.start, days=args.days)
            else:
                result = eod_mod.run_fundamentals(start=args.start, days=args.days)
        finally:
            eod_mod.close_lseg_session()
    elif args.cmd == "market":
        from capital.ingest import market as market_mod
        result = market_mod.run_market(days=args.days)
    elif args.cmd == "fred":
        from capital.ingest import market as market_mod
        result = market_mod.run_fred()
    elif args.cmd == "derived":
        from capital.ingest import derived as derived_mod
        result = derived_mod.run_derived()
    elif args.cmd == "sync":
        from capital.ingest import sync as sync_mod
        result = sync_mod.run_sync()

    log.info("%s: %s", args.cmd, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
