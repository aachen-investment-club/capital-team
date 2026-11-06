"""
Catalyst Watch analytics: each held position's cost, next report date and analyst
consensus, on frames the page loads through the data contract.

Agreement and trend are rule-based labels, thresholds in one place below so the
page's methodology text and the rules cannot drift.
"""
from collections import deque

import numpy as np
import pandas as pd

#: (consistent below, contradictory above) for each measure of disagreement.
TARGET_SPREAD = (0.15, 0.30)    # std of price targets / mean target
EPS_SPREAD = (0.05, 0.15)       # std of FY EPS estimates / |mean|
NOT_BUY_SHARE = (0.20, 0.40)    # share of analysts at Hold or below
MIN_ANALYSTS = 5                # fewer and no verdict is drawn
FLAT_REVISION = 0.01            # a revision inside +/-1% counts as unchanged


def fifo_cost(trades: pd.DataFrame) -> pd.DataFrame:
    """Average buy price of the shares still held, per symbol.

    FIFO on commission-inclusive prices, which is how IBKR reports `costBasisPrice`:
    a sale removes the oldest lots first, so a round trip does not distort the
    cost of what is left. Columns: symbol (index), avg_cost, cost_ccy, shares.
    """
    rows = {}
    t = trades.assign(_sell=trades["buy_sell"] == "SELL").sort_values(["trade_date", "_sell"])
    for symbol, g in t.groupby("symbol"):
        lots: deque = deque()
        for r in g.itertuples():
            if r.buy_sell == "BUY":
                lots.append([r.quantity, r.effective_price])
                continue
            left = abs(r.quantity)
            while left > 1e-9 and lots:
                used = min(left, lots[0][0])
                lots[0][0] -= used
                left -= used
                if lots[0][0] <= 1e-9:
                    lots.popleft()
        shares = sum(q for q, _ in lots)
        if shares > 1e-9:
            rows[symbol] = {"avg_cost": sum(q * p for q, p in lots) / shares,
                            "cost_ccy": g["currency"].iloc[-1], "shares": shares}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["avg_cost", "cost_ccy", "shares"])


def _revision(now: pd.Series, before: pd.Series) -> pd.Series:
    """Percent change, empty unless both are positive: a move through zero, such as
    a loss turning into a profit, has no meaningful percentage."""
    return (now / before - 1).where((now > 0) & (before > 0))


def _grade(x: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    """0 consistent / 1 mixed / 2 contradictory, NaN if `x` is."""
    return pd.Series(np.select([x < cuts[0], x <= cuts[1]], [0.0, 1.0], 2.0), index=x.index).where(x.notna())


def _agreement(df: pd.DataFrame) -> pd.Series:
    """Consistent / Mixed / Contradictory from the average grade of the three
    measures: one mixed measure is still Consistent, two make it Mixed."""
    score = pd.concat([_grade(df["target_spread"], TARGET_SPREAD), _grade(df["eps_spread"], EPS_SPREAD),
                       _grade(df["not_buy_share"], NOT_BUY_SHARE)], axis=1).mean(axis=1)
    label = pd.Series(np.select([score < 0.5, score < 1.34], ["Consistent", "Mixed"], "Contradictory"),
                      index=df.index).where(score.notna())
    return label.mask(df["n_analysts"] < MIN_ANALYSTS, "Too few analysts")


def _trend(target: pd.Series, eps: pd.Series) -> pd.Series:
    """Rising / Falling when target and EPS moved the same way over 90 days, Flat
    when neither moved, Mixed when they disagree."""
    signs = pd.concat([target, eps], axis=1).apply(
        lambda s: np.where(s.isna(), np.nan, np.where(s.abs() <= FLAT_REVISION, 0.0, np.sign(s))))
    up, down = (signs > 0).sum(axis=1), (signs < 0).sum(axis=1)
    label = pd.Series(np.select([(up > 0) & (down > 0), up > 0, down > 0], ["Mixed", "Rising", "Falling"], "Flat"),
                      index=target.index)
    return label.where(signs.notna().any(axis=1))


def catalyst_table(held: pd.DataFrame, consensus: pd.DataFrame, cost: pd.DataFrame,
                   today: pd.Timestamp) -> pd.DataFrame:
    """One row per position, largest first.

    Adds `days_to_report` (negative once the expected date has passed); `upside`,
    `upside_cost` and `pnl` against the price and the average buy price; the
    agreement measures; and the 30 and 90 day revisions. Target-based upside is
    left empty across currencies: a depositary receipt carries its underlying's
    consensus, and comparing the two would give a meaningless number.
    """
    df = held.merge(consensus, on="security_id", how="left").merge(cost, left_on="symbol", right_index=True, how="left")
    df["days_to_report"] = (df["expected_report_date"] - today).dt.days

    df["upside"] = (df["price_target"] / df["price"] - 1).where(df["target_ccy"] == df["price_ccy"])
    df["upside_cost"] = (df["price_target"] / df["avg_cost"] - 1).where(df["target_ccy"] == df["cost_ccy"])
    df["pnl"] = (df["price"] / df["avg_cost"] - 1).where(df["price_ccy"] == df["cost_ccy"])

    df["target_spread"] = df["target_std"] / df["price_target"]
    df["eps_spread"] = df["eps_fy_std"] / df["eps_fy"].abs()
    buys = df["n_strong_buy"] + df["n_buy"]
    not_buys = df["n_hold"] + df["n_sell"] + df["n_strong_sell"]
    df["not_buy_share"] = not_buys / (buys + not_buys)
    df["price_z"] = ((df["price"] - df["price_target"]) / df["target_std"]).where(df["target_ccy"] == df["price_ccy"])
    df["agreement"] = _agreement(df)

    # FY1 rolls to the next fiscal year after a full-year report, which would read
    # as a huge revision, so EPS is compared only within one fiscal year.
    same_year = df["eps_fy_end"] == df["eps_fy_end_90d"]
    df["eps_rev_30d"] = _revision(df["eps_fy"], df["eps_fy_30d"]).where(same_year)
    df["eps_rev_90d"] = _revision(df["eps_fy"], df["eps_fy_90d"]).where(same_year)
    df["target_rev_30d"] = _revision(df["price_target"], df["target_30d"])
    df["target_rev_90d"] = _revision(df["price_target"], df["target_90d"])
    df["rec_change_90d"] = df["rec_mean"] - df["rec_mean_90d"]
    df["trend"] = _trend(df["target_rev_90d"], df["eps_rev_90d"]).mask(df["n_analysts"] < MIN_ANALYSTS, "Too few analysts")
    return df.sort_values("pct_nav", ascending=False).reset_index(drop=True)
