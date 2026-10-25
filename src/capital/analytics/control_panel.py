"""
Control-panel analytics — the numbers behind the /control-panel overview.

Everything here is pure pandas/numpy on frames the page loads through the data
contract; nothing reads the store. One lookback convention throughout:

    WINDOW = 21 trading days  ("this month", rolling, not calendar MTD)
    LONG   = 252 trading days ("this year", the yardstick WINDOW is judged against)

Rolling rather than calendar month-to-date because MTD is three days of noise on
the 3rd of the month and a full month on the 30th; a fixed window makes today's
ranking comparable with last week's.
"""
import numpy as np
import pandas as pd

from capital.analytics.factors.risk import periods_per_year
from capital.analytics.trend import KalmanFilter2D, forecast_linear
from capital.analytics.volatility import ann_vol_from_var, fit_models

WINDOW = 21
LONG = 252

#: Strategies expressed as signed style-factor returns. A style factor return is
#: the payoff to one unit of exposure, so "low volatility" is the *negative* of
#: the residual-volatility factor, "small cap" the negative of size, and so on.
#: (name, factor key, sign, one-line description)
STRATEGIES = [
    ("Momentum", "momentum", 1, "Long 12-1 month winners, short losers."),
    ("Mean reversion", "reversal", 1, "Long last month's losers (short-term reversal)."),
    ("Quality", "quality", 1, "Long high profitability (ROE, ROA, gross margin)."),
    ("Value", "value", 1, "Long cheap on book-to-price."),
    ("Earnings yield", "earnyield", 1, "Long cheap on earnings, cash flow and sales."),
    ("Growth", "growth", 1, "Long fast revenue and earnings growth."),
    ("Low volatility", "resvol", -1, "Long calm stocks, short jumpy ones (−Residual Vol)."),
    ("Low beta", "beta", -1, "Long defensive, short high-beta (−Beta)."),
    ("Small cap", "size", -1, "Long small, short large (−Size)."),
    ("Mid cap", "midcap", 1, "Long mid caps against both size extremes."),
    ("Dividend yield", "divyield", 1, "Long high trailing dividend yield."),
    ("Balance-sheet strength", "leverage", -1, "Long low leverage (−Leverage)."),
    ("Illiquidity", "liquidity", -1, "Long thinly traded names (−Liquidity)."),
]

#: Cross-asset sleeve: ETF returns, deliberately listed apart from the factor
#: strategies — a factor return is a dollar-neutral long/short payoff, an ETF
#: return is a long-only one, and ranking them in one list would mislead.
CROSS_ASSET = {
    "SPY": "US equities (S&P 500)",
    "QQQ": "US growth (Nasdaq 100)",
    "IWM": "US small caps (Russell 2000)",
    "TLT": "Long Treasuries (20y+)",
    "HYG": "High-yield credit",
    "GLD": "Gold",
}

#: SPDR sector ETFs -> the GICS sector name used by the factor model's meta.
SECTOR_ETFS = {
    "XLK": "Information Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}


# ── Window helpers ────────────────────────────────────────────────────────────

def _compound(r: pd.Series) -> float:
    r = r.dropna()
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan


def trailing(df: pd.DataFrame | pd.Series, days: int) -> pd.DataFrame | pd.Series:
    """Rows within the last `days` trading days of the frame's own last date.

    Works for daily and lower-frequency frames alike: a weekly factor-return
    frame keeps the ~4 periods that end inside the window.
    """
    if len(df) == 0:
        return df
    end = df.index.max()
    start = end - pd.offsets.BDay(days)
    return df[df.index > start]


def price_return(prices: pd.Series, days: int) -> float:
    """Simple return over the last `days` trading days of a price series."""
    p = prices.dropna()
    if len(p) <= days:
        return np.nan
    return float(p.iloc[-1] / p.iloc[-days - 1] - 1.0)


def realised_vol(returns: pd.Series, days: int) -> pd.Series:
    """Rolling annualised realised vol (same units as `returns`, e.g. % in -> % out)."""
    return returns.rolling(days, min_periods=max(5, days // 2)).std() * np.sqrt(252)


# ── Strategy leaderboard ──────────────────────────────────────────────────────

def strategy_returns(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Per-period strategy returns (signed style-factor returns), one column each."""
    out = {}
    for name, key, sign, _ in STRATEGIES:
        if key in factor_returns.columns:
            out[name] = sign * factor_returns[key]
    return pd.DataFrame(out, index=factor_returns.index)


def strategy_leaderboard(strat: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Rank strategies by their trailing-WINDOW return, with persistence stats.

    Persistence is what separates "worked this month" from "is working":
      t_6m      mean / standard error over ~6 months of periods. |t| > 2 is a
                factor paying consistently rather than one lucky week.
      hit_6m    share of positive periods over the same span.
      ac_1y     lag-1 autocorrelation over a year — positive means this
                factor's returns trend (they persist into the next period).
    """
    ppy = periods_per_year(frequency)
    n6, n12 = max(4, ppy // 2), max(8, ppy)
    # Whole periods inside the window: 21 trading days is 4 weekly periods, not
    # the 5 a date cut-off would catch by including a partly-covered week.
    win = strat.tail(max(1, round(WINDOW * ppy / 252)))
    yr = strat.tail(max(1, round(LONG * ppy / 252)))
    descr = {name: d for name, _, _, d in STRATEGIES}
    factor_of = {name: key for name, key, _, _ in STRATEGIES}
    rows = []
    for col in strat.columns:
        s = strat[col].dropna()
        last6, last12 = s.tail(n6), s.tail(n12)
        sd = last6.std(ddof=1)
        rows.append({
            "strategy": col,
            "factor": factor_of.get(col, col),
            "description": descr.get(col, ""),
            "ret_window": _compound(win[col]),
            "ret_long": _compound(yr[col]),
            "t_6m": float(last6.mean() / (sd / np.sqrt(len(last6)))) if sd and len(last6) > 2 else np.nan,
            "hit_6m": float((last6 > 0).mean()) if len(last6) else np.nan,
            "ac_1y": float(last12.autocorr(1)) if len(last12) > 3 else np.nan,
            "n_window": int(win[col].notna().sum()),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("ret_window", ascending=False).reset_index(drop=True)


# ── Cross-asset and sectors ───────────────────────────────────────────────────

def asset_table(prices: dict[str, pd.Series], labels: dict[str, str],
                reference: pd.Series | None = None) -> pd.DataFrame:
    """WINDOW and LONG returns per ticker, plus the WINDOW return relative to
    `reference` (SPY) and the ticker's 21-day realised vol."""
    ref_w = price_return(reference, WINDOW) if reference is not None else np.nan
    rows = []
    for tk, label in labels.items():
        p = prices.get(tk)
        if p is None or len(p.dropna()) <= WINDOW:
            continue
        r = p.pct_change().dropna()
        rw = price_return(p, WINDOW)
        rows.append({
            "ticker": tk, "label": label,
            "ret_window": rw,
            "ret_long": price_return(p, LONG),
            "rel_window": rw - ref_w if np.isfinite(ref_w) else np.nan,
            "rv_window": float(r.tail(WINDOW).std() * np.sqrt(252)),
            "as_of": p.dropna().index.max(),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("ret_window", ascending=False).reset_index(drop=True) if len(df) else df


def us_sector_weights(meta: pd.DataFrame, min_names: int = 100) -> pd.Series:
    """Approximate S&P 500 sector weights from the factor run's US common stocks.

    The universe's US names are the S&P 500 / Nasdaq-100 chains, so their
    cap-weighted sector mix is a close proxy for SPY's. Returns an empty series
    when coverage is too thin to be honest about (fewer than `min_names`).
    """
    if meta is None or meta.empty or not {"country", "sector", "market_cap"} <= set(meta.columns):
        return pd.Series(dtype=float)
    us = meta[(meta["country"] == "US") & (meta.get("asset_type", "COMMON") == "COMMON")]
    us = us.dropna(subset=["market_cap", "sector"])
    if len(us) < min_names:
        return pd.Series(dtype=float)
    w = us.groupby("sector")["market_cap"].sum()
    return (w / w.sum()).sort_values(ascending=False)


# ── Portfolio ─────────────────────────────────────────────────────────────────

def position_contributions(weightings: pd.DataFrame, days: int = WINDOW) -> pd.DataFrame:
    """Each holding's contribution to portfolio return over the last `days`.

    contribution_t = weight_{t-1} x return_t, summed over the window and quoted
    in basis points of NAV. Contribution rather than raw return is the honest
    ranking: a +40% move on a 0.5% position is worth less to the fund than −8%
    on a 10% one. Yesterday's weight is used because that is what was at risk
    over today's move.
    """
    w = weightings.sort_values(["symbol", "date"]).copy()
    w["w_prev"] = w.groupby("symbol")["pct_nav"].shift(1) / 100.0
    dates = np.sort(w["date"].unique())
    if len(dates) == 0:
        return pd.DataFrame()
    window_dates = dates[-days:]
    win = w[w["date"].isin(window_dates)].copy()
    win["ctr"] = win["w_prev"].fillna(0.0) * win["daily_return"].fillna(0.0)
    last = w[w["date"] == dates[-1]].set_index("symbol")
    g = win.groupby("symbol")
    out = pd.DataFrame({
        "name": g["name"].last(),
        "category": g["category"].last(),
        "ctr_bps": g["ctr"].sum() * 1e4,
        "ret_window": g["daily_return"].apply(lambda r: float((1 + r.fillna(0)).prod() - 1)),
    })
    out["weight"] = last["pct_nav"].reindex(out.index).fillna(0.0) / 100.0
    out = out[~out.index.str.startswith("CASH_")]
    return out.sort_values("ctr_bps", ascending=False)


def to_eur(prices_usd: pd.Series, fx: pd.DataFrame) -> pd.Series:
    """Translate a USD price series into EUR with the store's EUR-per-USD rate.
    Falls back to the USD series when no FX is available."""
    if fx is None or fx.empty or "USD" not in fx.columns:
        return prices_usd
    rate = fx["USD"].reindex(prices_usd.index.union(fx.index)).ffill().reindex(prices_usd.index)
    return (prices_usd * rate).dropna()


def portfolio_kpis(nav: pd.Series, bench: pd.Series) -> dict:
    """Headline numbers for the fund (TWR unit price) against the benchmark.

    Beta and tracking error use only the dates both series actually observed,
    so a holiday (or a stale benchmark) does not show up as a run of zero
    returns on one side. A benchmark that stops more than a few days before
    the NAV gets no window return at all rather than a flat one.
    """
    nav = nav.dropna()
    bench = bench.dropna()
    r = nav.pct_change().dropna()
    b = bench.pct_change().dropna()
    both = pd.concat([r.rename("p"), b.rename("b")], axis=1, join="inner").dropna()
    bench_fresh = len(bench) and bench.index.max() >= nav.index[-1] - pd.offsets.BDay(3)
    n = len(r)
    long_n = min(LONG, n)

    ytd_base = nav[nav.index < pd.Timestamp(nav.index[-1].year, 1, 1)]
    ytd = float(nav.iloc[-1] / ytd_base.iloc[-1] - 1) if len(ytd_base) else float(nav.iloc[-1] / nav.iloc[0] - 1)

    peak = nav.cummax()
    dd = nav / peak - 1.0

    q = r.tail(long_n)
    var95 = float(-np.quantile(q, 0.05)) if len(q) >= 20 else np.nan
    cvar95 = float(-q[q <= np.quantile(q, 0.05)].mean()) if len(q) >= 20 else np.nan

    tail_both = both.tail(long_n)
    beta = float(np.cov(tail_both["p"], tail_both["b"])[0, 1] / np.var(tail_both["b"], ddof=1)) \
        if len(tail_both) > 20 else np.nan
    te = float((tail_both["p"] - tail_both["b"]).std() * np.sqrt(252)) if len(tail_both) > 20 else np.nan

    ret_w = price_return(nav, WINDOW)
    bench_w = price_return(bench[bench.index <= nav.index[-1]], WINDOW) if bench_fresh else np.nan
    return {
        "as_of": nav.index[-1],
        "bench_as_of": bench.index.max() if len(bench) else None,
        "n_days": n,
        "ret_window": ret_w,
        "bench_window": bench_w,
        "active_window": ret_w - bench_w if np.isfinite(ret_w) and np.isfinite(bench_w) else np.nan,
        "ytd": ytd,
        "since_inception": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "drawdown": float(dd.iloc[-1]),
        "max_drawdown": float(dd.min()),
        "vol_window": float(r.tail(WINDOW).std() * np.sqrt(252)),
        "vol_long": float(q.std() * np.sqrt(252)),
        "sharpe_long": float(q.mean() / q.std() * np.sqrt(252)) if q.std() else np.nan,
        "beta": beta,
        "tracking_error": te,
        "var95": var95,
        "cvar95": cvar95,
        "long_label": f"{long_n}d" if long_n < LONG else "1y",
    }


# ── Volatility in perspective (GJR-GARCH) ─────────────────────────────────────

def vol_perspective(returns_pct: pd.Series, horizon: int = WINDOW,
                    model: str = "GJR-GARCH") -> dict:
    """Today's GARCH volatility next to the yardsticks a PM actually compares it with.

    - the 21-day and 252-day realised vol (is the model above or below what
      the market has actually delivered, short and long?)
    - the model's own long-run level (where the forecast mean-reverts to)
    - its percentile in its own history and over the last year
    - the 21-day change (rising or falling vol, the "trend")
    - the average forecast vol over the next `horizon` days, which is the
      right number for sizing a position held for that long

    All in annualised %.
    """
    r = returns_pct.dropna()
    res = fit_models(r, (model,))[model]
    fit, p = res["fit"], res["params"]
    cond = pd.Series(np.asarray(fit.conditional_volatility) * np.sqrt(252),
                     index=r.index[-len(fit.conditional_volatility):])

    fc_var = fit.forecast(horizon=horizon, reindex=False).variance.iloc[-1].to_numpy()
    future = pd.bdate_range(r.index[-1] + pd.Timedelta(days=1), periods=horizon)
    fc = ann_vol_from_var(pd.Series(fc_var, index=future))

    persistence = p["alpha"] + p["beta"] + 0.5 * p["gamma"]
    long_run = float(np.sqrt(p["omega"] / (1.0 - persistence) * 252)) if persistence < 1 else np.nan
    half_life = float(np.log(0.5) / np.log(persistence)) if 0 < persistence < 1 else np.nan

    rv21 = realised_vol(r, WINDOW)
    rv252 = realised_vol(r, LONG)
    now = float(cond.iloc[-1])
    last_year = cond.tail(LONG)
    prior = cond.iloc[-WINDOW - 1] if len(cond) > WINDOW else np.nan

    return {
        "as_of": r.index[-1],
        "garch": cond,
        "forecast": fc,
        "rv_window": rv21,
        "rv_long": rv252,
        "now": now,
        "forecast_avg": float(np.sqrt(np.mean(fc ** 2))),  # average variance, not average vol
        "forecast_end": float(fc.iloc[-1]),
        "long_run": long_run,
        "half_life_d": half_life,
        "rv_window_now": float(rv21.iloc[-1]),
        "rv_long_now": float(rv252.iloc[-1]) if rv252.notna().any() else np.nan,
        "pct_all": float((cond <= now).mean()),
        "pct_1y": float((last_year <= now).mean()),
        "change_window": now - float(prior) if np.isfinite(prior) else np.nan,
        "n_obs": len(r),
        "params": p,
    }


# ── Portfolio outlook (Kalman trend) ──────────────────────────────────────────

def kalman_outlook(nav: pd.Series, horizon: int = WINDOW) -> dict:
    """Kalman level/slope on the NAV rebased to 100, projected `horizon` days.

    Rebased because the filter's noise settings (obs 1.0, process 0.01) are in
    price units: on a unit price near 1.0 they would make the trend almost flat
    whatever the data did. At 100 they mean ~1% observation noise and ~0.1 point
    a day of slope drift — sensible for a fund unit price.

    This is a trend extrapolation with a widening ±1σ band, not a return forecast.
    """
    idx = (nav.dropna() / nav.dropna().iloc[0]) * 100.0
    filt = KalmanFilter2D().run(idx)
    fc = forecast_linear(filt, horizon)
    last = float(idx.iloc[-1])
    slope = float(filt["slope"].iloc[-1])
    return {
        "index": idx,
        "filter": filt,
        "forecast": fc,
        "slope_ann": slope * 252 / last,          # trend speed as a fraction of NAV per year
        "slope_window_ago": float(filt["slope"].iloc[-WINDOW - 1]) * 252 / last
        if len(filt) > WINDOW else np.nan,
        "projected_ret": float(fc["trend"].iloc[-1] / last - 1.0),
        "band_ret": (float(fc["lower_1s"].iloc[-1] / last - 1.0),
                     float(fc["upper_1s"].iloc[-1] / last - 1.0)),
        "gap": float((idx.iloc[-1] - filt["trend"].iloc[-1]) / filt["trend"].iloc[-1]),
    }


# ── Alerts ────────────────────────────────────────────────────────────────────

#: Thresholds in one place so the methodology text and the rules cannot drift.
LIMITS = {
    "max_position": 0.10,       # single position above 10% of NAV
    "position_drop": -0.15,     # a holding down >15% inside WINDOW
    "drawdown": -0.10,          # fund drawdown beyond 10%
    "vol_pct": 0.80,            # GARCH vol above its own 80th percentile
    "vol_expansion": 1.5,       # 21d realised above 1.5x the 252d realised
    "regime_flip_days": 5,      # regime changed within the last week
    "exposure_z": 0.5,          # |active exposure| worth flagging
    "stale_bdays": 3,           # data older than 3 business days
    "stale_model_days": 10,     # factor run older than 10 calendar days
}


def _bdays_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return max(0, len(pd.bdate_range(a, b)) - 1)


def _streak_start(mask: pd.Series) -> pd.Timestamp | None:
    """First date of the run of True values that ends on the last row, or None
    when the condition does not hold today. Dates an ongoing condition by when
    it *began*, which is what "how recent" means for an alert."""
    m = mask.dropna().astype(bool)
    if m.empty or not m.iloc[-1]:
        return None
    off = np.flatnonzero(~m.to_numpy())
    return m.index[off[-1] + 1] if len(off) else m.index[0]


#: Relevance weight per level. The ranking score is relevance x recency, with
#: recency = 0.4 + 0.6 * 0.5 ** (age / RECENCY_HALF_LIFE): a fresh orange outranks
#: a month-old red, but an old red never sinks below a fresh green.
LEVEL_WEIGHT = {"red": 3.0, "orange": 2.0, "green": 1.0}
RECENCY_HALF_LIFE = 10  # business days


def build_alerts(*, today: pd.Timestamp, regime: dict | None, vol: dict | None,
                 kpis: dict | None, nav: pd.Series | None, positions: pd.DataFrame | None,
                 positions_as_of: pd.Timestamp | None, active_exposure: pd.Series | None,
                 leaderboard: pd.DataFrame | None, factor_end: pd.Timestamp | None,
                 freshness: dict[str, pd.Timestamp | None]) -> list[dict]:
    """Rule-based flags ranked by relevance x recency.

    Each alert: level (red/orange/green), title, detail, since (when the
    condition began), age_bd (business days since then), score.
    """
    L = LIMITS
    out: list[dict] = []

    def add(level, title, detail, since):
        since = pd.Timestamp(since) if since is not None and not pd.isna(since) else today
        out.append({"level": level, "title": title, "detail": detail, "since": since})

    for label, ts in freshness.items():
        if ts is None or pd.isna(ts):
            add("orange", f"{label}: no data", "The boxes that use this input are empty.", today)
            continue
        ts = pd.Timestamp(ts)
        if label == "Factor model":
            age = (today - ts).days
            if age > L["stale_model_days"]:
                add("orange", "Factor model stale",
                    f"Run ends {ts:%d %b} ({age}d ago); strategy and exposure numbers lag.",
                    ts + pd.Timedelta(days=L["stale_model_days"]))
        else:
            lag = _bdays_between(ts, today)
            if lag > L["stale_bdays"]:
                add("orange", f"{label} data stale", f"Last observation {ts:%d %b}, {lag} business days ago.",
                    ts + pd.offsets.BDay(L["stale_bdays"]))

    if regime:
        # The champion regime: direction (Kalman slope) and volatility (GARCH nowcast).
        cur, state, vol_state = regime["current"], regime["state"], regime["vol_state"]
        run_start = _streak_start(state == cur["direction"])
        if cur["direction"] == "Bear":
            add("red", "Bear regime", f"SPY trend slope {cur['slope']:+.0%}/yr, below the Bear line.", run_start)
        if cur["days_direction"] <= L["regime_flip_days"]:
            add("orange", f"Regime change → {cur['direction']}",
                f"Switched {cur['days_direction']}d ago; trend slope {cur['slope']:+.0%}/yr.", run_start)
        if cur["volatility"] == "Turbulent":
            add("orange", "Turbulent regime", f"SPY vol nowcast {cur['vol_nowcast']:.0%}.",
                _streak_start(vol_state == "Turbulent"))

    if vol:
        g = vol["garch"]
        if vol["pct_all"] > L["vol_pct"]:
            add("orange", "Volatility elevated",
                f"SPY GARCH vol {vol['now']:.1f}% is at the {vol['pct_all']:.0%} percentile.",
                _streak_start(g > g.quantile(L["vol_pct"])))
        rv21, rv252 = vol["rv_window"], vol["rv_long"]
        if np.isfinite(vol["rv_long_now"]) and vol["rv_window_now"] > L["vol_expansion"] * vol["rv_long_now"]:
            add("orange", "Volatility expanding",
                f"SPY 21d realised {vol['rv_window_now']:.1f}% vs 252d {vol['rv_long_now']:.1f}%.",
                _streak_start(rv21 > L["vol_expansion"] * rv252))

    if kpis and nav is not None and np.isfinite(kpis.get("drawdown", np.nan)) and kpis["drawdown"] < L["drawdown"]:
        dd = nav / nav.cummax() - 1
        add("red", "Fund drawdown limit", f"{kpis['drawdown']:.1%} below peak (limit {L['drawdown']:.0%}).",
            _streak_start(dd < L["drawdown"]))

    if positions is not None and len(positions):
        for sym, row in positions[positions["weight"] > L["max_position"]].iterrows():
            add("orange", f"Concentration: {sym}",
                f"{row['weight']:.1%} of NAV (limit {L['max_position']:.0%}).", positions_as_of)
        for sym, row in positions[positions["ret_window"] < L["position_drop"]].iterrows():
            level = "red" if row["ret_window"] < 2 * L["position_drop"] else "orange"
            add(level, f"Drawdown: {sym}",
                f"{row['ret_window']:.1%} over {WINDOW}d, {row['ctr_bps']:+.0f} bps to the fund.", positions_as_of)

    if active_exposure is not None and len(active_exposure) and leaderboard is not None and len(leaderboard):
        ret = dict(zip(leaderboard["factor"], leaderboard["ret_window"]))
        sign = {key: s for _, key, s, _ in STRATEGIES}
        name = {key: n for n, key, _, _ in STRATEGIES}
        for factor, x in active_exposure.items():
            if factor not in ret or abs(x) < L["exposure_z"] or not np.isfinite(ret[factor]):
                continue
            # Strategy return r = sign * factor return, so our P&L from the tilt is x * sign * r.
            if np.sign(x * sign[factor] * ret[factor]) < 0:
                add("green", f"Tilt against the tape: {name[factor]}",
                    f"Active {x:+.2f}σ to {factor}; {name[factor]} returned {ret[factor]:+.2%} over {WINDOW}d.",
                    factor_end)

    for a in out:
        a["age_bd"] = _bdays_between(min(a["since"], today), today)
        a["score"] = LEVEL_WEIGHT[a["level"]] * (0.4 + 0.6 * 0.5 ** (a["age_bd"] / RECENCY_HALF_LIFE))
    return sorted(out, key=lambda a: (-a["score"], a["age_bd"]))


# ── Scorecard: fund vs SPY ────────────────────────────────────────────────────

#: (metric, higher_is_better). Drawdown is negative, so "higher" = shallower.
SCORECARD_METRICS = [
    ("Return", True), ("Volatility", False), ("Sharpe", True),
    ("Max drawdown", True), ("VaR 95% (1d)", False), ("CVaR 95% (1d)", False), ("Hit rate", True),
]


def _window_stats(prices: pd.Series) -> dict:
    p = prices.dropna()
    r = p.pct_change().dropna()
    if len(r) < 2:
        return {m: np.nan for m, _ in SCORECARD_METRICS}
    q = np.quantile(r, 0.05) if len(r) >= 10 else np.nan
    sd = r.std()
    return {
        "Return": float(p.iloc[-1] / p.iloc[0] - 1),
        "Volatility": float(sd * np.sqrt(252)),
        "Sharpe": float(r.mean() / sd * np.sqrt(252)) if sd else np.nan,
        "Max drawdown": float((p / p.cummax() - 1).min()),
        "VaR 95% (1d)": float(-q) if np.isfinite(q) else np.nan,
        "CVaR 95% (1d)": float(-r[r <= q].mean()) if np.isfinite(q) else np.nan,
        "Hit rate": float((r > 0).mean()),
    }


def scorecard(nav: pd.Series, bench: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Fund vs benchmark on each metric over each window, with who won.

    Both sides cover the same calendar span (the fund's last `n` trading days),
    each measured on its own trading days, so a US holiday does not appear as a
    zero-return day for SPY. The benchmark starts at its last close on or
    before the fund's start date. Returns a long frame:
    window, metric, fund, bench, better (True/False/NaN).
    """
    nav = nav.dropna()
    bench = bench.dropna()
    rows = []
    for n in windows:
        seg = nav.iloc[-(n + 1):]
        start, end = seg.index[0], seg.index[-1]
        before = bench[bench.index <= start]
        b = bench[(bench.index >= (before.index[-1] if len(before) else start)) & (bench.index <= end)]
        fs, bs = _window_stats(seg), _window_stats(b)
        for metric, higher in SCORECARD_METRICS:
            f, bv = fs[metric], bs[metric]
            better = (f > bv if higher else f < bv) if np.isfinite(f) and np.isfinite(bv) else np.nan
            rows.append({"window": n, "metric": metric, "fund": f, "bench": bv, "better": better})
    return pd.DataFrame(rows)
