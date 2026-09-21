"""
Quarterly report exports: TWR daily returns, AUM/flows, theme/position
snapshots, and Brinson-Fachler theme attribution.

Pure functions over `capital.data.loaders` DataFrames, parameterised by
(start, end) / as-of dates so the same code produces any past or future
quarter's report, not just the current one.
"""
import pandas as pd

from capital.data import loaders

BENCHMARK_TICKERS = ("SPX", "MSCI_WORLD", "MSCI_EUROPE", "60_40")

#: ticker -> export column name. Public: shared with the dashboard's HTML
#: table rendering (performance.py) so headers stay in sync with the CSVs.
BENCHMARK_EXPORT_COLS = {
    "SPX":         "sp500_return_pct",
    "MSCI_WORLD":  "msci_world_return_pct",
    "MSCI_EUROPE": "msci_europe_return_pct",
    "60_40":       "blend_60_40_return_pct",
}


# ── 1. Daily returns time series ──────────────────────────────────────────────

def daily_returns_timeseries(end, benchmarks: tuple = BENCHMARK_TICKERS) -> pd.DataFrame:
    """One row per trading day, portfolio inception -> `end`.

    portfolio_return_pct is time-weighted: it tracks pct-change of the fund's
    unit price (nav_history.fund_nav), which moves only with investment
    performance — deposits/withdrawals buy or redeem units at the prior day's
    unit price and never affect it (see loaders.get_nav_history). Safe to
    chain for QTD/YTD/trailing-12M/since-inception returns over any range.

    Columns: date, portfolio_return_pct, <one *_return_pct column per benchmark>
    """
    df = loaders.get_portfolio_and_benchmarks()
    df = df[df["date"] <= pd.Timestamp(end)]
    tickers = ["PORTFOLIO"] + [b for b in benchmarks if b != "PORTFOLIO"]
    df = df[df["ticker"].isin(tickers)]

    wide = df.pivot(index="date", columns="ticker", values="daily_return") * 100
    rename = {"PORTFOLIO": "portfolio_return_pct"}
    rename.update({t: BENCHMARK_EXPORT_COLS.get(t, f"{t.lower()}_return_pct")
                   for t in tickers if t != "PORTFOLIO"})
    wide = wide.rename(columns=rename)
    ordered = ["portfolio_return_pct"] + [rename[t] for t in tickers
                                          if t != "PORTFOLIO" and rename[t] in wide.columns]
    wide = wide[ordered].round(4)
    return wide.reset_index().sort_values("date").reset_index(drop=True)


# ── 5a. Returns summary table (QTD / YTD / Trailing 12M / Since Inception) ───

_PERIOD_LABELS = ("QTD", "YTD", "Trailing 12M", "Since Inception")


def _baseline_date(end_ts: pd.Timestamp, period: str) -> pd.Timestamp:
    """The reference date whose index_value anchors a period's return."""
    if period == "QTD":
        q_month = ((end_ts.month - 1) // 3) * 3 + 1
        return pd.Timestamp(year=end_ts.year, month=q_month, day=1) - pd.Timedelta(days=1)
    if period == "YTD":
        return pd.Timestamp(year=end_ts.year, month=1, day=1) - pd.Timedelta(days=1)
    if period == "Trailing 12M":
        return end_ts - pd.DateOffset(years=1)
    raise ValueError(period)


def returns_summary_table(end, benchmarks: tuple = BENCHMARK_TICKERS) -> pd.DataFrame:
    """Compounded TWR returns over QTD / YTD / Trailing 12M / Since Inception,
    for the portfolio and each tracked benchmark, all as of `end`.

    Built from each series' own index_value — already a growth-of-1 index
    from that series' own first available date (see
    loaders.get_portfolio_and_benchmarks) — so this chains the same TWR daily
    returns as daily_returns_timeseries rather than averaging anything.

    Trailing 12M is None ("N/A" once rendered) for a ticker with under 12
    months of history as of `end`, rather than a misleadingly short partial
    period silently labelled as if it were a full year.
    """
    df = loaders.get_portfolio_and_benchmarks()
    end_ts = pd.Timestamp(end)
    tickers = ["PORTFOLIO"] + [b for b in benchmarks if b != "PORTFOLIO"]
    col_of = {"PORTFOLIO": "portfolio_return_pct", **BENCHMARK_EXPORT_COLS}

    data: dict[str, dict[str, float | None]] = {}
    for t in tickers:
        col = col_of[t]
        series = df[(df["ticker"] == t) & (df["date"] <= end_ts)].sort_values("date")
        if series.empty:
            data[col] = {p: None for p in _PERIOD_LABELS}
            continue

        inception_date = series["date"].iloc[0]
        inception_value = float(series["index_value"].iloc[0])
        end_value = float(series["index_value"].iloc[-1])

        def value_on_or_before(ts, _series=series):
            avail = _series[_series["date"] <= ts]
            return float(avail["index_value"].iloc[-1]) if not avail.empty else None

        vals: dict[str, float | None] = {}
        for period in _PERIOD_LABELS:
            if period == "Since Inception":
                base = inception_value
            elif period == "Trailing 12M":
                cutoff = end_ts - pd.DateOffset(years=1)
                base = value_on_or_before(cutoff) if inception_date <= cutoff else None
            else:
                base = value_on_or_before(_baseline_date(end_ts, period))
                if base is None:
                    base = inception_value  # inception fell inside this period
            vals[period] = round((end_value / base - 1) * 100, 4) if base else None
        data[col] = vals

    out = pd.DataFrame(data)
    ordered_cols = ["portfolio_return_pct"] + [col_of[t] for t in tickers if t != "PORTFOLIO"]
    out = out[[c for c in ordered_cols if c in out.columns]]
    out.index.name = "period"
    return out.reindex(list(_PERIOD_LABELS))


# ── 2. AUM & net flows time series ────────────────────────────────────────────

def aum_and_flows_timeseries(end) -> pd.DataFrame:
    """One row per day, portfolio inception -> `end`.

    aum_eur is end-of-day fund NAV; external_flow_eur is that day's net IBKR
    deposit/withdrawal cash transactions (positive = contribution, negative =
    withdrawal, 0 on days with no flow).

    Reconciliation note: this fund's unit-price accounting applies a flow at
    the PRIOR day's unit price, so newly deposited capital participates in
    that day's return (see loaders.get_nav_history / the ingestion Lambda's
    recompute_history). The exact identity is therefore

        aum(t) = (aum(t-1) + external_flow_eur(t)) * (1 + portfolio_return_pct(t)/100)

    rather than aum(t-1)*(1+r) + flow(t) applied strictly after the day's
    return. The two differ only by flow(t) x return(t), negligible for
    realistic flow sizes.
    """
    nav = loaders.get_nav_history()
    nav = nav[nav["date"] <= pd.Timestamp(end)][["date", "raw_nav_eur"]] \
        .rename(columns={"raw_nav_eur": "aum_eur"})

    dep = loaders.get_deposit_log()
    if dep.empty:
        flows = pd.DataFrame(columns=["date", "external_flow_eur"])
    else:
        flows = (dep[dep["date"] <= pd.Timestamp(end)]
                 .groupby("date", as_index=False)["amount_eur"].sum()
                 .rename(columns={"amount_eur": "external_flow_eur"}))

    out = nav.merge(flows, on="date", how="left")
    out["external_flow_eur"] = out["external_flow_eur"].fillna(0.0).round(2)
    out["aum_eur"] = out["aum_eur"].round(2)
    return out.sort_values("date").reset_index(drop=True)


# ── 3. Theme / position snapshot ──────────────────────────────────────────────

_SNAPSHOT_COLS = ["as_of_date", "theme", "symbol", "name", "isin",
                  "weight_pct_of_nav", "period_return_pct", "cumulative_return_pct"]


def theme_position_snapshot(as_of, period_start) -> pd.DataFrame:
    """One row per held position, grouped by theme, as of `as_of` (or the
    latest date on/before it — supports both a "current" snapshot and an
    arbitrary historical as-of date).

    period_return_pct is compounded from `period_start` (exclusive) through
    the snapshot date; cumulative_return_pct is since the position's own
    inception (cost basis).

    CAVEAT — cash positions: if a currency balance was fully swept to zero
    and later refilled during the window (a cash-management event, not an
    investment loss), that day's daily_return reads -100% and the compounded
    period_return_pct for that cash symbol will too. Pre-existing
    characteristic of daily_weightings, not specific to this function.
    """
    df = loaders.get_daily_weightings_with_themes()
    as_of_ts = pd.Timestamp(as_of)
    avail = sorted(d for d in df["date"].unique() if d <= as_of_ts)
    if not avail:
        return pd.DataFrame(columns=_SNAPSHOT_COLS)
    snap_date = avail[-1]
    snap = df[df["date"] == snap_date].copy()

    period_start_ts = pd.Timestamp(period_start)
    window = df[(df["date"] > period_start_ts) & (df["date"] <= snap_date)].copy()
    window["gross"] = 1 + window["daily_return"]
    period_ret = window.groupby("symbol")["gross"].prod() - 1

    snap["as_of_date"] = snap_date
    snap["weight_pct_of_nav"] = snap["pct_nav"].round(4)
    snap["period_return_pct"] = (snap["symbol"].map(period_ret).fillna(0.0) * 100).round(4)
    snap["cumulative_return_pct"] = (snap["cumulative_return"] * 100).round(4)

    return (snap[_SNAPSHOT_COLS]
            .sort_values(["theme", "weight_pct_of_nav"], ascending=[True, False])
            .reset_index(drop=True))


# ── 4. Attribution by theme (Brinson-Fachler, simplified) ────────────────────

_ATTRIBUTION_COLS = ["benchmark", "theme", "portfolio_weight_pct", "benchmark_weight_pct",
                     "selection_effect_bps", "allocation_effect_bps", "total_effect_bps"]


def theme_attribution(start, end, benchmarks: tuple = BENCHMARK_TICKERS) -> pd.DataFrame:
    """Brinson-Fachler attribution by theme over (start, end], one row per
    (benchmark, theme).

    SIMPLIFICATION: the benchmarks this fund tracks (S&P 500 / MSCI World /
    MSCI Europe / a 60-40 blend) have no native breakdown by this portfolio's
    themes (e.g. "AI & Semis"), so there is no real per-theme benchmark
    weight or return to use. Rather than invent one, this falls back to a
    benchmark-neutral, single-factor model per theme:
      - benchmark_weight_pct   := portfolio_weight_pct (assume the benchmark
        is allocated exactly like the portfolio across themes)
      - benchmark_theme_return := benchmark_total_return (assume the
        benchmark does not tilt within a theme either)
    Both substitutions zero allocation_effect_bps by construction — there is
    no data to measure it from — so all measurable excess return per theme is
    reported as selection_effect_bps. This is a labelled approximation, not a
    true per-theme Brinson-Fachler split; treat allocation_effect_bps as
    structurally zero rather than a real "0 bps of allocation skill" result.

    CAVEAT — cash themes: a cash position's daily_return is only a genuine
    return while its FX balance stays nonzero. If a currency balance is fully
    swept to zero and later refilled (a real, if unusual, cash-management
    event — see loaders.get_daily_weightings_history), that day's
    daily_return reads -100%, which then poisons any theme whose weight
    includes that cash bucket even though no investment loss occurred. This
    is a pre-existing characteristic of the underlying daily_weightings data
    (same one behind the "Day"/"Since Inception" figures already shown for
    CASH_USD/CASH_GBP on the Holdings table), not specific to this function —
    a cash-heavy theme's selection_effect_bps here should be sanity-checked
    against theme_position_snapshot.csv before being read at face value.
    """
    weights = loaders.get_daily_weightings_with_themes()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    start_avail = sorted(d for d in weights["date"].unique() if d <= start_ts)
    end_avail = sorted(d for d in weights["date"].unique() if d <= end_ts)
    if not start_avail or not end_avail:
        return pd.DataFrame(columns=_ATTRIBUTION_COLS)
    start_date, end_date = start_avail[-1], end_avail[-1]

    start_snap = weights[weights["date"] == start_date]
    port_w = start_snap.groupby("theme")["pct_nav"].sum()
    start_w = start_snap.set_index("symbol")["pct_nav"]
    theme_of = start_snap.set_index("symbol")["theme"]

    window = weights[(weights["date"] > start_date) & (weights["date"] <= end_date)].copy()
    window["gross"] = 1 + window["daily_return"]
    pos_ret = window.groupby("symbol")["gross"].prod() - 1

    contrib = start_w * pos_ret.reindex(start_w.index).fillna(0.0)
    theme_ret = contrib.groupby(theme_of).sum() / start_w.groupby(theme_of).sum()

    bench_df = loaders.get_portfolio_and_benchmarks()
    rows = []
    for bm in benchmarks:
        bser = bench_df[(bench_df["ticker"] == bm)
                        & (bench_df["date"] >= start_date)
                        & (bench_df["date"] <= end_date)].sort_values("date")
        if bser.empty:
            continue
        bench_total_ret = bser["index_value"].iloc[-1] / bser["index_value"].iloc[0] - 1
        for theme, pw in port_w.items():
            pw = float(pw)
            bw = pw  # benchmark-neutral assumption (see docstring)
            pt_ret = float(theme_ret.get(theme, 0.0))
            selection_bps = pw / 100 * (pt_ret - bench_total_ret) * 10000
            allocation_bps = 0.0  # (pw - bw) == 0 by construction
            rows.append({
                "benchmark": bm, "theme": theme,
                "portfolio_weight_pct": round(pw, 4), "benchmark_weight_pct": round(bw, 4),
                "selection_effect_bps": round(selection_bps, 2),
                "allocation_effect_bps": round(allocation_bps, 2),
                "total_effect_bps": round(selection_bps + allocation_bps, 2),
            })

    return (pd.DataFrame(rows, columns=_ATTRIBUTION_COLS)
            .sort_values(["benchmark", "portfolio_weight_pct"], ascending=[True, False])
            .reset_index(drop=True))
