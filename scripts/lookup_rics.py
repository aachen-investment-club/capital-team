#!/usr/bin/env python3
"""
RIC lookup helper — find valid LSEG RICs for every security in config/security_master.csv.

Queries the LSEG content.search API by ISIN (which is stable and accurate) and returns
candidate instruments/quote listings for each security, ranked by exchange/currency match.

Review the output, pick the RIC for the exchange you want, and update config/security_master.csv.
You do NOT need to re-run this after the CSV is correct — it is a one-time discovery tool.

Usage:
  python scripts/lookup_rics.py                        # all active securities
  python scripts/lookup_rics.py --isin IE00BMW42306    # one specific ISIN
  python scripts/lookup_rics.py --top 10               # show more candidates per security
  python scripts/lookup_rics.py --inactive             # include inactive securities too
"""
import argparse
import os
import pathlib
import sys

import pandas as pd
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


def _search_isin(isin: str, asset_type: str, currency: str, top: int) -> pd.DataFrame:
    """Return candidate instruments/listings for the given ISIN."""
    from lseg.data.content import search

    # Use EQUITY_QUOTES for stocks; FUND_QUOTES for ETFs.
    # Fall back to SEARCH_ALL if those return nothing.
    view_primary   = search.Views.FUND_QUOTES if asset_type.upper() == "ETF" else search.Views.EQUITY_QUOTES
    views_fallback = [search.Views.SEARCH_ALL]

    select_fields = "RIC,CommonName,ExchangeCode,Currency,RCSAssetCategoryLeaf"

    for view in [view_primary] + views_fallback:
        try:
            resp = search.Definition(
                query=isin,
                view=view,
                filter=f"IssuerISIN eq '{isin}'",
                select=select_fields,
                top=top,
            ).get_data()
        except Exception:
            # Some views reject certain filter fields — try next
            try:
                resp = search.Definition(
                    query=isin,
                    view=view,
                    select=select_fields,
                    top=top,
                ).get_data()
            except Exception:
                continue

        if resp is not None and resp.data is not None:
            df = resp.data.df
            if df is not None and not df.empty:
                # Surface rows whose Currency matches the security's home currency first
                cols = [c for c in ("RIC", "CommonName", "ExchangeCode", "Currency",
                                    "RCSAssetCategoryLeaf") if c in df.columns]
                df = df[cols].drop_duplicates(subset=["RIC"] if "RIC" in cols else None)
                if "Currency" in df.columns:
                    df = df.sort_values(
                        "Currency",
                        key=lambda s: s.apply(lambda v: 0 if str(v) == currency else 1),
                    )
                return df

    return pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--isin",     default="",  help="look up a single ISIN only")
    p.add_argument("--top",      type=int, default=6, help="max candidates per security (default: 6)")
    p.add_argument("--inactive", action="store_true", help="include inactive (active=false) securities")
    args = p.parse_args()

    csv_path = _ROOT / "config" / "security_master.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    master = pd.read_csv(csv_path, dtype=str)
    master["_active"] = master["active"].str.lower().isin(("true", "1", "yes"))

    if args.isin:
        master = master[master["isin"] == args.isin]
        if master.empty:
            print(f"ISIN {args.isin!r} not found in security_master.csv")
            sys.exit(1)
    elif not args.inactive:
        master = master[master["_active"]]

    if master.empty:
        print("No securities to look up.")
        sys.exit(0)

    import lseg.data as ld
    ld.open_session()
    print(f"LSEG session opened — searching {len(master)} security(s)\n")

    try:
        for _, row in master.iterrows():
            sec_id  = row["security_id"]
            ticker  = row["ticker"]
            isin    = row["isin"]
            ccy     = row["currency"]
            current = row["ric"]
            atype   = row.get("asset_type", "")

            bar = "─" * 72
            print(bar)
            print(f"  {sec_id}  {ticker:8s}  ISIN: {isin}  ccy: {ccy}  asset_type: {atype}")
            print(f"  Current RIC: {current}")
            print()

            candidates = _search_isin(isin, atype, ccy, args.top)
            if candidates.empty:
                print("  ⚠  No results — try searching manually in LSEG Workspace.")
            else:
                # indent the table
                lines = candidates.to_string(index=False).splitlines()
                for line in lines:
                    print("  " + line)
            print()

    finally:
        try:
            ld.close_session()
        except Exception:
            pass

    print("─" * 72)
    print("Next step: copy the correct RIC into the 'ric' column of config/security_master.csv.")
    print("Rows whose Currency matches the security's home currency are listed first.")


if __name__ == "__main__":
    main()
