#!/usr/bin/env python3
"""
Universe expansion — grow config/security_master.csv to a screening-sized universe.

Why this exists
---------------
A cross-sectional factor model needs *breadth*. With ~100 securities, ~30 factors
are fitted to ~3 observations each: the style exposures are still informative but
the industry and country returns are noise (the Factor Screen shows this as the
"observations per factor" figure, and warns below 10). At 1,000-2,000 names the
same model is properly identified.

How it expands
--------------
`ld.get_data(universe=["0#.STOXX"], fields=[...])` expands an index chain RIC
server-side and returns reference data for every constituent in one request.

This is the *Data* service, not the streaming one. `ld.discovery.Chain` — the
obvious API for this — needs real-time entitlements this account does not have
(`NotEntitled: PE(3134)`), and so does `TR.IndexConstituentRIC`. Verified counts:

    0#.STOXX   600      0#.SPX    503      0#.NDX    102
    0#.FTSE    100      0#.MDAXI   50      0#.GDAXI   40

Two reference fields are *also* not entitled and are therefore not requested:
`TR.ISIN` (new rows get a blank ISIN — the RIC is the join key everywhere that
matters) and `TR.PriceCurrency` (currency is derived from the RIC's exchange
suffix instead, which is deterministic).

Usage
-----
  python scripts/expand_universe.py --list                 # show the chain presets
  python scripts/expand_universe.py --chains europe        # dry run, prints a diff
  python scripts/expand_universe.py --chains europe --write
  python scripts/expand_universe.py --chains world --limit 1500 --write
  python scripts/expand_universe.py --rics-file etfs.txt --write   # ETFs / one-offs

Existing rows are never modified or deactivated — this only ever adds securities,
and keeps a .bak of the previous file.

After writing, back-fill history. This is the slow part and a lot of LSEG
requests, so run it deliberately and off-hours:

  uv run capital-ingest eod  --start 2016-01-01 --resume
  uv run capital-ingest fund --start 2016-01-01 --resume
  uv run capital-ingest derived
  uv run capital-ingest factors

Do not run it while the nightly job is due: LSEG allows one concurrent platform
session per credential and `signon_control=True` will kick the other one.
"""
import argparse
import pathlib
import sys

import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from capital.ingest.eod import close_lseg_session, open_lseg_session  # noqa: E402
from capital.settings import settings  # noqa: E402

#: Chain RICs by preset, with constituent counts verified against this account.
CHAIN_PRESETS: dict[str, list[str]] = {
    "europe":  ["0#.STOXX"],                        # STOXX Europe 600      -> 600
    "germany": ["0#.GDAXI", "0#.MDAXI"],            # DAX + MDAX            ->  90
    "uk":      ["0#.FTSE"],                         # FTSE 100              -> 100
    "us":      ["0#.SPX", "0#.NDX"],                # S&P 500 + Nasdaq 100  -> 555
    "world":   ["0#.STOXX", "0#.SPX", "0#.NDX", "0#.FTSE", "0#.GDAXI", "0#.MDAXI"],
}

#: Reference fields, all verified as entitled. TR.ISIN and TR.PriceCurrency are
#: deliberately absent — both return "access denied" on this account.
FIELDS = ["TR.RIC", "TR.CommonName", "TR.ExchangeCountryCode",
          "TR.CompanyMarketCap", "TR.GICSSector", "TR.AssetCategory"]

COLUMNS = ["security_id", "ric", "ticker", "isin", "name", "currency",
           "asset_type", "country", "active"]

#: RIC exchange suffix -> trading currency. Deterministic, and needed because
#: TR.PriceCurrency is not entitled. London (.L) quotes in pence; the model works
#: on returns, which are scale-invariant, so that does not matter here.
SUFFIX_CCY = {
    "DE": "EUR", "F": "EUR", "PA": "EUR", "AS": "EUR", "BR": "EUR", "MI": "EUR",
    "MC": "EUR", "LS": "EUR", "HE": "EUR", "VI": "EUR", "IR": "EUR", "AT": "EUR",
    "L": "GBP", "S": "CHF", "ST": "SEK", "CO": "DKK", "OL": "NOK", "WA": "PLN",
    "PR": "CZK", "BUD": "HUF", "IS": "TRY", "TA": "ILS",
    "N": "USD", "O": "USD", "OQ": "USD", "K": "USD", "A": "USD", "P": "USD",
    "TO": "CAD", "T": "JPY", "AX": "AUD", "HK": "HKD", "SI": "SGD",
}
COUNTRY_CCY = {"DE": "EUR", "FR": "EUR", "NL": "EUR", "BE": "EUR", "IT": "EUR",
               "ES": "EUR", "PT": "EUR", "FI": "EUR", "IE": "EUR", "AT": "EUR",
               "LU": "EUR", "GR": "EUR", "GB": "GBP", "CH": "CHF", "SE": "SEK",
               "DK": "DKK", "NO": "NOK", "US": "USD", "PL": "PLN", "CA": "CAD",
               "JP": "JPY", "AU": "AUD"}


def currency_for(ric: str, country: str) -> str:
    """Trading currency from the RIC's exchange suffix, country as the fallback."""
    suffix = ric.rsplit(".", 1)[-1].upper() if "." in ric else ""
    return SUFFIX_CCY.get(suffix) or COUNTRY_CCY.get(str(country).upper(), "USD")


def asset_type_for(category: str) -> str:
    """Map TR.AssetCategory onto the master's asset_type vocabulary."""
    cat = str(category or "").lower()
    if "fund" in cat or "etf" in cat or "trust" in cat:
        return "ETF"
    if "index" in cat:
        return "INDEX"
    return "COMMON"


def fetch_chain(chain: str) -> pd.DataFrame:
    """Constituents of an index chain RIC, with their reference data.

    One request: the Data service expands the chain server-side, so there is no
    separate "list constituents, then describe them" round trip.
    """
    import lseg.data as ld
    try:
        df = ld.get_data(universe=[chain], fields=FIELDS)
    except Exception as exc:                                        # noqa: BLE001
        print(f"  ! {chain}: {exc}")
        return pd.DataFrame()
    if df is None or df.empty:
        print(f"  ! {chain}: no constituents returned")
        return pd.DataFrame()
    print(f"  · {chain}: {len(df)} constituents")
    return df


def fetch_rics(rics: list[str], batch: int = 100) -> pd.DataFrame:
    """Reference data for an explicit RIC list (ETFs, one-off additions)."""
    import lseg.data as ld
    frames = []
    for i in range(0, len(rics), batch):
        chunk = rics[i:i + batch]
        print(f"  · reference data {i + 1}-{i + len(chunk)} of {len(rics)}")
        try:
            df = ld.get_data(universe=chunk, fields=FIELDS)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as exc:                                    # noqa: BLE001
            print(f"    ! batch failed: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pick(df: pd.DataFrame, *needles: str) -> pd.Series:
    for col in df.columns:
        key = col.lower().replace(" ", "")
        if all(n in key for n in needles):
            return df[col]
    return pd.Series([None] * len(df), index=df.index)


def build_rows(ref: pd.DataFrame, existing: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Turn LSEG reference data into new security_master rows.

    Ranked by market cap so a `--limit` keeps the most liquid, most modellable
    names rather than an arbitrary slice. Deduplication is on RIC alone: ISIN is
    not entitled here, and the RIC is the key the ingest and the model join on.
    """
    if ref.empty:
        return pd.DataFrame(columns=COLUMNS)
    ric = _pick(ref, "ric")
    if ric.isna().all():
        ric = _pick(ref, "instrument")
    out = pd.DataFrame({
        "ric": ric.astype(str).str.strip(),
        "name": _pick(ref, "commonname").astype(str).str.strip(),
        "country": _pick(ref, "country").astype(str).str.strip(),
        "sector": _pick(ref, "gics").astype(str),
        "category": _pick(ref, "assetcategory").astype(str),
        "market_cap": pd.to_numeric(_pick(ref, "marketcap"), errors="coerce"),
    })
    out = out[out["ric"].str.len() > 0]
    out = out[~out["ric"].str.lower().isin({"nan", "none", "<na>"})]
    out = out[~out["ric"].isin(set(existing["ric"]))]
    out = out.drop_duplicates(subset="ric")
    out = out.sort_values("market_cap", ascending=False, na_position="last")
    if limit:
        out = out.head(limit)

    out["currency"] = [currency_for(r, c) for r, c in zip(out["ric"], out["country"])]
    out["asset_type"] = [asset_type_for(c) for c in out["category"]]
    out["isin"] = ""                       # TR.ISIN is not entitled on this account

    # Ticker: the RIC root, uppercased. It only has to be unique and readable —
    # the model keys on security_id, and portfolio positions match on ticker, so
    # a new row must never collide with a curated one.
    taken = set(existing["ticker"])
    tickers = []
    for r in out["ric"]:
        base = "".join(ch for ch in r.split(".")[0].upper() if ch.isalnum()) or "SEC"
        candidate, n = base, 1
        while candidate in taken:
            n += 1
            candidate = f"{base}{n}"
        taken.add(candidate)
        tickers.append(candidate)
    out["ticker"] = tickers

    first = _next_id(existing)
    out["security_id"] = [f"SEC_{first + i:04d}" for i in range(len(out))]
    out["active"] = "true"
    return out[COLUMNS]


def _next_id(existing: pd.DataFrame) -> int:
    nums = [int(s.split("_")[-1]) for s in existing["security_id"]
            if s.rsplit("_", 1)[-1].isdigit()]
    return (max(nums) + 1) if nums else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", nargs="*", default=[],
                    help="preset names or raw chain RICs (e.g. europe 0#.SPX)")
    ap.add_argument("--rics-file", help="file of explicit RICs, one per line "
                                        "(for ETFs and one-off additions)")
    ap.add_argument("--limit", type=int, help="keep only the N largest new names")
    ap.add_argument("--write", action="store_true",
                    help="write the merged CSV (default is a dry run)")
    ap.add_argument("--list", action="store_true", help="show the presets and exit")
    args = ap.parse_args(argv)

    if args.list or not (args.chains or args.rics_file):
        print("Chain presets (constituent counts verified against this account):")
        for name, chains in CHAIN_PRESETS.items():
            print(f"  {name:<8} {' '.join(chains)}")
        print("\nPass preset names or raw chain RICs to --chains, and/or an "
              "explicit RIC list to --rics-file.")
        return 0

    csv_path = settings.config_dir / "security_master.csv"
    existing = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"security_master.csv: {len(existing)} rows")

    chains: list[str] = []
    for token in args.chains:
        chains.extend(CHAIN_PRESETS.get(token, [token]))
    explicit: list[str] = []
    if args.rics_file:
        explicit = [ln.strip() for ln in pathlib.Path(args.rics_file).read_text().splitlines()
                    if ln.strip() and not ln.startswith("#")]

    open_lseg_session()
    try:
        frames = []
        if chains:
            print("Expanding chains:")
            frames.extend(f for f in (fetch_chain(c) for c in chains) if not f.empty)
        if explicit:
            print(f"Fetching {len(explicit)} explicit RICs:")
            df = fetch_rics(explicit)
            if not df.empty:
                frames.append(df)
    finally:
        close_lseg_session()

    if not frames:
        print("Nothing returned.")
        return 1
    ref = pd.concat(frames, ignore_index=True)

    new_rows = build_rows(ref, existing, args.limit)
    print(f"\n{len(new_rows)} new securities (existing rows are never modified)")
    if new_rows.empty:
        return 0
    by_type = new_rows["asset_type"].value_counts().to_dict()
    by_ccy = new_rows["currency"].value_counts().head(8).to_dict()
    print(f"  asset types: {by_type}")
    print(f"  currencies:  {by_ccy}")
    print(new_rows[["security_id", "ric", "ticker", "name", "country",
                    "currency", "asset_type"]].head(15).to_string(index=False))
    if len(new_rows) > 15:
        print(f"... and {len(new_rows) - 15} more")

    if not args.write:
        print("\nDry run — re-run with --write to update the CSV.")
        return 0

    merged = pd.concat([existing, new_rows], ignore_index=True)
    backup = csv_path.with_suffix(".csv.bak")
    csv_path.replace(backup)
    merged.to_csv(csv_path, index=False)
    print(f"\nWrote {len(merged)} rows to {csv_path} (previous version kept as {backup.name})")
    print("Next: capital-ingest eod --start <date> --resume, then fund / derived / factors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
