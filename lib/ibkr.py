"""
IBKR Flex Query XML parsers and position computation utilities.
All data access for the Performance page flows through lib/data.py,
which calls these functions. Do not import directly from pages.

Data flow:
  data/ibkr/trade_log.xml            → trade_log.parquet
  data/ibkr/prior_positions.xml      → prior_positions.parquet  (historical backfill)
  data/ibkr/open_positions/YYYYMMDD.xml → open_positions.parquet  (appended daily)

The open_positions.parquet is the daily extension point:
  each day, parse a new OpenPositions flex query XML and append via append_open_positions().
"""
import pathlib
import xml.etree.ElementTree as ET

import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parent.parent
IBKR_DIR = _ROOT / "data" / "ibkr"

_TRADE_LOG_PARQUET     = IBKR_DIR / "trade_log.parquet"
_PRIOR_POS_PARQUET     = IBKR_DIR / "prior_positions.parquet"
_OPEN_POS_PARQUET      = IBKR_DIR / "open_positions.parquet"
_FX_POS_PARQUET        = IBKR_DIR / "fx_positions.parquet"
_OPEN_POS_XML_DIR      = IBKR_DIR / "open_positions"


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_trades(xml_str: str) -> pd.DataFrame:
    """
    Parse Trades XML from IBKR Flex Query into a DataFrame.
    FX (CASH) trades are excluded — only equity/ETF trades are returned.
    Returns: trade_date, symbol, name, isin, currency, asset_type,
             quantity, proceeds, commission, buy_sell
    """
    root = ET.fromstring(xml_str)
    rows = []
    for t in root.iter("Trade"):
        # Skip zero-commission FX fills (tiny rounding/residual lots, not real trades)
        if t.get("assetCategory") == "CASH" and float(t.get("ibCommission", 0)) == 0:
            continue
        is_fx = t.get("assetCategory") == "CASH"
        rows.append({
            "trade_date": pd.to_datetime(t.get("tradeDate"), format="%Y%m%d"),
            "symbol":     t.get("symbol"),
            "name":       t.get("description"),
            "isin":       t.get("securityID"),
            "currency":   t.get("currency"),
            "asset_type": "FX" if is_fx else t.get("subCategory"),
            "quantity":   float(t.get("quantity", 0)),
            "proceeds":   float(t.get("proceeds", 0)),
            "commission": float(t.get("ibCommission", 0)),
            "buy_sell":   t.get("buySell"),
        })
    return pd.DataFrame(rows)


def parse_prior_positions(xml_str: str) -> pd.DataFrame:
    """
    Parse PriorPeriodPositions XML from IBKR Flex Query.
    Returns: date, symbol, name, isin, currency, asset_type,
             fx_rate_to_base, price, prior_mtm_pnl
    """
    root = ET.fromstring(xml_str)
    rows = []
    for p in root.iter("PriorPeriodPosition"):
        rows.append({
            "date":             pd.to_datetime(p.get("date"), format="%Y%m%d"),
            "symbol":           p.get("symbol"),
            "name":             p.get("description"),
            "isin":             p.get("securityID"),
            "currency":         p.get("currency"),
            "asset_type":       p.get("subCategory"),
            "fx_rate_to_base":  float(p.get("fxRateToBase", 1)),
            "price":            float(p.get("price", 0)),
            "prior_mtm_pnl":    float(p.get("priorMtmPnl", 0)),
        })
    return pd.DataFrame(rows)


def parse_fx_positions(xml_str: str) -> pd.DataFrame:
    """
    Parse FxPositions XML from IBKR Flex Query.
    Returns: date, fx_currency, quantity, cost_price, value_eur, unrealized_pl

    cost_price is the EUR rate at which the cash was acquired (inception FX rate).
    """
    root = ET.fromstring(xml_str)
    rows = []
    for p in root.iter("FxPosition"):
        rows.append({
            "date":          pd.to_datetime(p.get("reportDate"), format="%Y%m%d"),
            "fx_currency":   p.get("fxCurrency"),
            "quantity":      float(p.get("quantity", 0)),
            "cost_price":    float(p.get("costPrice", 0)),
            "value_eur":     float(p.get("value", 0)),
            "unrealized_pl": float(p.get("unrealizedPL", 0)),
        })
    return pd.DataFrame(rows)


def parse_open_positions(xml_str: str) -> pd.DataFrame:
    """
    Parse OpenPositions XML from IBKR Flex Query.
    Returns: date, symbol, name, isin, currency, asset_type,
             fx_rate_to_base, price, shares, cost_basis_price,
             pct_nav, fifo_pnl_unrealized
    This is the daily extension point — call append_open_positions() to persist.
    """
    root = ET.fromstring(xml_str)
    rows = []
    for p in root.iter("OpenPosition"):
        rows.append({
            "date":                pd.to_datetime(p.get("reportDate"), format="%Y%m%d"),
            "symbol":              p.get("symbol"),
            "name":                p.get("description"),
            "isin":                p.get("securityID"),
            "currency":            p.get("currency"),
            "asset_type":          p.get("subCategory"),
            "fx_rate_to_base":     float(p.get("fxRateToBase", 1)),
            "price":               float(p.get("markPrice", 0)),
            "shares":              float(p.get("position", 0)),
            "cost_basis_price":    float(p.get("costBasisPrice", 0)),
            "pct_nav":             float(p.get("percentOfNAV", 0)),
            "fifo_pnl_unrealized": float(p.get("fifoPnlUnrealized", 0)),
        })
    return pd.DataFrame(rows)


# ── Parquet helpers ───────────────────────────────────────────────────────────

def load_fx_positions() -> pd.DataFrame:
    return pd.read_parquet(_FX_POS_PARQUET)


def append_fx_positions(new_df: pd.DataFrame) -> None:
    """Append a parsed fx_positions DataFrame to the cumulative Parquet file (idempotent)."""
    if _FX_POS_PARQUET.exists():
        existing = pd.read_parquet(_FX_POS_PARQUET)
        existing_dates = set(existing["date"].dt.normalize().unique())
        to_add = new_df[~new_df["date"].dt.normalize().isin(existing_dates)]
        if to_add.empty:
            return
        combined = pd.concat([existing, to_add], ignore_index=True)
    else:
        combined = new_df.copy()
    combined = combined.sort_values(["date", "fx_currency"]).reset_index(drop=True)
    _FX_POS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_FX_POS_PARQUET, index=False)


def load_trade_log() -> pd.DataFrame:
    return pd.read_parquet(_TRADE_LOG_PARQUET)


def load_prior_positions() -> pd.DataFrame:
    return pd.read_parquet(_PRIOR_POS_PARQUET)


def load_open_positions() -> pd.DataFrame:
    return pd.read_parquet(_OPEN_POS_PARQUET)


def append_open_positions(new_df: pd.DataFrame) -> None:
    """
    Append a parsed open_positions DataFrame to the cumulative Parquet file.
    Dates already present are skipped (idempotent — safe to re-run).
    """
    if _OPEN_POS_PARQUET.exists():
        existing = pd.read_parquet(_OPEN_POS_PARQUET)
        existing_dates = set(existing["date"].dt.normalize().unique())
        new_dates = set(new_df["date"].dt.normalize().unique())
        to_add = new_df[~new_df["date"].dt.normalize().isin(existing_dates)]
        if to_add.empty:
            print(f"  open_positions: dates {new_dates} already present — skipped")
            return
        combined = pd.concat([existing, to_add], ignore_index=True)
    else:
        combined = new_df.copy()

    combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
    _OPEN_POS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_OPEN_POS_PARQUET, index=False)
    print(f"  open_positions: appended {len(new_df)} rows → {len(combined)} total")


# ── Position computation ──────────────────────────────────────────────────────

def _shares_from_trades(trades: pd.DataFrame) -> dict[str, float]:
    """Net share count per symbol (buys positive, sells negative)."""
    df = trades.copy()
    df.loc[df["buy_sell"] == "SELL", "quantity"] *= -1
    return df.groupby("symbol")["quantity"].sum().to_dict()


def build_cash_positions(
    prior_df: pd.DataFrame,
    fx_snap: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build daily cash position rows (CASH_EUR, CASH_GBP, CASH_USD) for every date.

    Cash quantities are constant from fund inception (all FX trades happened May 7-8).
    Daily EUR values are computed using daily FX rates from prior_df where available,
    with the FX snapshot's costPrice used as the inception/fallback rate.

    Schema matches equity build_daily_positions output:
      date, symbol, name, isin, currency, asset_type,
      fx_rate_to_base, price, shares, value_eur
    """
    # Cash quantities and inception FX rates from the latest FX snapshot
    fx_map: dict[str, dict] = {}
    for _, row in fx_snap.iterrows():
        ccy = row["fx_currency"]
        fx_map[ccy] = {
            "quantity":   row["quantity"],
            "cost_price": row["cost_price"] if row["cost_price"] > 0 else 1.0,
        }

    # Daily GBP and USD FX rates from equity prior positions
    # Use TRNl for GBP, URNU for USD — both have daily fx_rate_to_base
    gbp_rates = (
        prior_df[prior_df["symbol"] == "TRNl"][["date", "fx_rate_to_base"]]
        .rename(columns={"fx_rate_to_base": "fx_gbp"})
        .set_index("date")["fx_gbp"]
    )
    usd_rates = (
        prior_df[prior_df["symbol"] == "URNU"][["date", "fx_rate_to_base"]]
        .rename(columns={"fx_rate_to_base": "fx_usd"})
        .set_index("date")["fx_usd"]
    )

    cash_rows = []
    for dt in all_dates:
        # GBP rate: use prior_positions if available, else inception cost_price
        fx_gbp = gbp_rates.get(dt, fx_map.get("GBP", {}).get("cost_price", 1.165320039))
        # USD rate: use prior_positions if available, else inception cost_price
        fx_usd = usd_rates.get(dt, fx_map.get("USD", {}).get("cost_price", 0.853018161))

        for ccy, sym, name_str, fx in [
            ("EUR", "CASH_EUR", "Euro Cash",           1.0),
            ("GBP", "CASH_GBP", "British Pound Cash",  fx_gbp),
            ("USD", "CASH_USD", "US Dollar Cash",       fx_usd),
        ]:
            qty = fx_map.get(ccy, {}).get("quantity", 0.0)
            cash_rows.append({
                "date":            dt,
                "symbol":          sym,
                "name":            name_str,
                "isin":            "",
                "currency":        ccy,
                "asset_type":      "CASH",
                "fx_rate_to_base": fx,
                "price":           1.0,
                "shares":          qty,
                "value_eur":       qty * fx,
            })

    return pd.DataFrame(cash_rows)


def build_daily_positions(
    prior_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    open_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a unified daily positions table from IBKR data sources.

    Schema: date, symbol, name, isin, currency, asset_type,
            fx_rate_to_base, price, shares, value_eur

    Logic:
    - Prior period rows are enriched with share counts derived from the trade log.
    - Open positions rows already carry share counts from IBKR.
    - For any date present in open_df, the open_df row takes precedence.
    - Prices are forward-filled within each symbol to cover exchange holidays
      where IBKR omits a row (e.g., May 25 Whit Monday).
    """
    shares_map = _shares_from_trades(trades_df)

    # Enrich historical positions with share counts from trade log
    hist = prior_df.copy()
    hist["shares"] = hist["symbol"].map(shares_map)
    hist = hist.dropna(subset=["shares"])
    hist["value_eur"] = hist["price"] * hist["shares"] * hist["fx_rate_to_base"]

    # Open positions → unified schema (shares already included)
    cur = open_df[["date", "symbol", "name", "isin", "currency", "asset_type",
                   "fx_rate_to_base", "price", "shares"]].copy()
    cur["value_eur"] = cur["price"] * cur["shares"] * cur["fx_rate_to_base"]

    # Drop prior rows for dates already covered by open_positions (no duplicates)
    open_dates = set(open_df["date"].dt.normalize().unique())
    hist = hist[~hist["date"].dt.normalize().isin(open_dates)]

    cols = ["date", "symbol", "name", "isin", "currency", "asset_type",
            "fx_rate_to_base", "price", "shares", "value_eur"]
    combined = pd.concat([hist[cols], cur[cols]], ignore_index=True)

    # Forward-fill within each symbol across the full date grid so exchange
    # holidays don't create gaps in portfolio weights
    all_dates = sorted(combined["date"].unique())
    all_symbols = sorted(combined["symbol"].unique())
    grid = pd.MultiIndex.from_product([all_dates, all_symbols], names=["date", "symbol"])
    combined = (
        combined.set_index(["date", "symbol"])
        .reindex(grid)
        .reset_index()
        .sort_values(["symbol", "date"])
    )

    # Static fields: fill from any available row for that symbol
    for col in ["name", "isin", "currency", "asset_type", "shares"]:
        combined[col] = combined.groupby("symbol")[col].ffill().bfill()

    # Time-varying fields: forward-fill only (don't carry future prices backward)
    combined["price"] = combined.groupby("symbol")["price"].ffill()
    combined["fx_rate_to_base"] = combined.groupby("symbol")["fx_rate_to_base"].ffill()

    # Drop dates where a symbol has no price yet (e.g., SAN/TRNl on May 8)
    combined = combined.dropna(subset=["price"])
    combined["value_eur"] = combined["price"] * combined["shares"] * combined["fx_rate_to_base"]

    return combined.sort_values(["date", "symbol"]).reset_index(drop=True)


def compute_weightings(
    positions: pd.DataFrame,
    open_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive pct_nav, daily_return, and cumulative_return for each (date, symbol) row.

    pct_nav          — position value as % of total portfolio EUR value on that date
    daily_return     — FX-adjusted price return vs previous trading day;
                       for the first day each symbol appears, return is vs cost basis
    cumulative_return — FX-adjusted return since purchase (inception), using
                        cost_basis_price from the latest open_positions snapshot
                        and the first available FX rate for non-EUR positions

    Returns all input columns plus: pct_nav, daily_return, cumulative_return
    """
    df = positions.copy()

    # Portfolio NAV (EUR) per day — used only for pct_nav, not for returns
    daily_nav = df.groupby("date")["value_eur"].sum().rename("nav_eur")
    df = df.merge(daily_nav, on="date")
    df["pct_nav"] = df["value_eur"] / df["nav_eur"] * 100

    # Cost basis per symbol from the latest open_positions snapshot (IBKR cost_basis_price
    # is in local currency and already includes commissions, matching what IBKR shows)
    latest_snap = open_df.sort_values("date").groupby("symbol").last().reset_index()
    cost_basis  = latest_snap[["symbol", "cost_basis_price"]]
    df = df.merge(cost_basis, on="symbol", how="left")

    # Cumulative return in local currency — price/cost_basis - 1.
    # Matches IBKR's unrealised P&L display (e.g. URNU: 31.055/37.489-1 = -17.16%).
    # FX effects on the EUR portfolio value are captured separately via value_eur/pct_nav.
    df["cumulative_return"] = df["price"] / df["cost_basis_price"] - 1

    # Daily return in local currency: price_t / price_{t-1} - 1.
    # First day in the series uses cost_basis_price as the prior reference so the
    # first-day return reflects the move from purchase price to first end-of-day price.
    df = df.sort_values(["symbol", "date"])
    df["prev_price"] = df.groupby("symbol")["price"].shift(1)
    df["prev_price"] = df["prev_price"].fillna(df["cost_basis_price"])
    df["daily_return"] = df["price"] / df["prev_price"] - 1

    drop = ["nav_eur", "cost_basis_price", "prev_price"]
    return df.drop(columns=[c for c in drop if c in df.columns])
