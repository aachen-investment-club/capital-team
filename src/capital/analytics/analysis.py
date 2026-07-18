"""
Portfolio performance analytics — metrics computation and colour grading.
Inputs are plain pandas Series; no dependency on any specific data loader.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── Thresholds for colour grading ─────────────────────────────────────────────
# Each entry: (metric, green_condition, orange_condition)  — evaluated top-down.
# Values outside both conditions fall to red.

_THRESHOLDS: dict[str, tuple] = {
    "Annualized Return":            ("gt", 0.10, 0.0),
    "Annualized Volatility":        ("lt", 0.10, 0.20),   # lower is better
    "Sharpe Ratio":                 ("gt", 1.0,  0.5),
    "Sortino Ratio":                ("gt", 2.0,  1.0),
    "Calmar Ratio":                 ("gt", 1.0,  0.5),
    "R-Squared (Benchmark)":        ("gt", 0.8,  0.5),
    "Max Drawdown":                 ("gt", -0.10, -0.20), # less negative is better
    "Max Drawdown Duration (days)": ("lt", 63,   126),    # lower is better
    "Max Drawdown Duration (months)":("lt", 3,   6),
    "VaR (Monthly, 5%)":            ("gt", -0.03, -0.05),
    "CVaR (Monthly, 5%)":           ("gt", -0.05, -0.07),
    "Upside Capture %":             ("gt", 100,  50),
    "Downside Capture %":           ("lt", 50,   100),    # lower is better
    "Skewness":                     ("gt", 0.0, -0.5),
    "Kurtosis":                     ("lt", 3.0,  5.0),    # lower is better
}

_GREEN  = "#D1FAE5"
_ORANGE = "#FEF3C7"
_RED    = "#FEE2E2"
_GREEN_TEXT  = "#065F46"
_ORANGE_TEXT = "#92400E"
_RED_TEXT    = "#991B1B"


def _colour(metric: str, value: float) -> tuple[str, str]:
    """Return (bg_colour, text_colour) for a metric value."""
    if metric not in _THRESHOLDS:
        return "", ""
    direction, g_thresh, o_thresh = _THRESHOLDS[metric]
    if direction == "gt":
        if value > g_thresh:
            return _GREEN, _GREEN_TEXT
        if value > o_thresh:
            return _ORANGE, _ORANGE_TEXT
        return _RED, _RED_TEXT
    else:  # "lt" — lower is better
        if value < g_thresh:
            return _GREEN, _GREEN_TEXT
        if value < o_thresh:
            return _ORANGE, _ORANGE_TEXT
        return _RED, _RED_TEXT


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(
    daily_returns: pd.Series,
    bench_daily_returns: pd.Series | None = None,
    conf: float = 0.05,
) -> dict:
    """
    Compute the full set of portfolio performance metrics.

    Parameters
    ----------
    daily_returns       : daily portfolio returns (decimal, e.g. 0.01 = 1 %)
    bench_daily_returns : daily benchmark returns for R² and capture ratios
    conf                : VaR / CVaR confidence level (default 5 %)

    Returns
    -------
    dict of metric_name → float
    """
    dr = daily_returns.dropna()
    if len(dr) < 5:
        raise ValueError("Need at least 5 daily returns to compute metrics.")

    # Monthly returns (resampled from daily)
    mr = (1 + dr).resample("ME").prod() - 1

    ann_ret = float(dr.mean() * 252)
    ann_vol = float(dr.std() * np.sqrt(252))

    # Ratios
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    downside = mr[mr < 0].std() * np.sqrt(12)
    sortino  = (mr.mean() * 12) / downside if downside > 0 else float("nan")

    # Drawdown
    wealth = (1 + dr).cumprod()
    peaks  = wealth.expanding().max()
    dd     = (wealth - peaks) / peaks
    mdd    = float(dd.min())
    # duration = longest consecutive streak below zero
    in_dd     = (dd < 0).astype(int)
    streaks   = in_dd.groupby((in_dd == 0).cumsum()).sum()
    mdd_days  = int(streaks.max()) if not streaks.empty else 0
    mdd_months = mdd_days / 21.0

    calmar = ann_ret / abs(mdd) if mdd < 0 else float("nan")

    # VaR / CVaR (monthly)
    var_val  = float(np.percentile(mr, conf * 100))
    cvar_val = float(mr[mr <= var_val].mean()) if (mr <= var_val).any() else var_val

    # Higher-moment stats
    skew = float(mr.skew())
    kurt = float(mr.kurtosis())  # excess kurtosis (normal = 0)

    metrics: dict = {
        "Annualized Return":              ann_ret,
        "Annualized Volatility":          ann_vol,
        "Sharpe Ratio":                   sharpe,
        "Sortino Ratio":                  sortino,
        "Calmar Ratio":                   calmar,
        "Max Drawdown":                   mdd,
        "Max Drawdown Duration (days)":   mdd_days,
        "Max Drawdown Duration (months)": mdd_months,
        "VaR (Monthly, 5%)":              var_val,
        "CVaR (Monthly, 5%)":             cvar_val,
        "Skewness":                       skew,
        "Kurtosis":                       kurt,
    }

    # Benchmark-dependent metrics
    if bench_daily_returns is not None:
        bdr  = bench_daily_returns.dropna()
        common = dr.index.intersection(bdr.index)
        if len(common) >= 5:
            bmr = (1 + bdr.loc[common]).resample("ME").prod() - 1
            pmr = (1 + dr.loc[common]).resample("ME").prod() - 1
            ci  = pmr.index.intersection(bmr.index)
            pmr, bmr = pmr.loc[ci], bmr.loc[ci]

            # R-squared vs benchmark
            if len(pmr) >= 3:
                X  = sm.add_constant(bmr.values)
                r2 = float(sm.OLS(pmr.values, X).fit().rsquared)
            else:
                r2 = float("nan")

            # Capture ratios
            up_mask = bmr > 0
            dn_mask = bmr < 0
            up_cap = (
                (pmr[up_mask].mean() / bmr[up_mask].mean()) * 100
                if up_mask.any() and bmr[up_mask].mean() != 0 else float("nan")
            )
            dn_cap = (
                (pmr[dn_mask].mean() / bmr[dn_mask].mean()) * 100
                if dn_mask.any() and bmr[dn_mask].mean() != 0 else float("nan")
            )

            metrics["R-Squared (Benchmark)"] = r2
            metrics["Upside Capture %"]       = up_cap
            metrics["Downside Capture %"]     = dn_cap

    return metrics


# ── Display helpers ───────────────────────────────────────────────────────────

# How to format each metric for display
_FMT: dict[str, str] = {
    "Annualized Return":              "pct",
    "Annualized Volatility":          "pct",
    "Sharpe Ratio":                   "2f",
    "Sortino Ratio":                  "2f",
    "Calmar Ratio":                   "2f",
    "R-Squared (Benchmark)":          "2f",
    "Max Drawdown":                   "pct",
    "Max Drawdown Duration (days)":   "0f",
    "Max Drawdown Duration (months)": "1f",
    "VaR (Monthly, 5%)":              "pct",
    "CVaR (Monthly, 5%)":             "pct",
    "Upside Capture %":               "1f",
    "Downside Capture %":             "1f",
    "Skewness":                       "2f",
    "Kurtosis":                       "2f",
}

# Friendly descriptions shown as tooltips / legend
METRIC_DESCRIPTIONS: dict[str, str] = {
    "Annualized Return":              "Geometric mean return, scaled to one year",
    "Annualized Volatility":          "Standard deviation of daily returns × √252",
    "Sharpe Ratio":                   "Ann. return ÷ ann. vol (risk-free = 0)",
    "Sortino Ratio":                  "Ann. return ÷ downside deviation (monthly)",
    "Calmar Ratio":                   "Ann. return ÷ |max drawdown|",
    "R-Squared (Benchmark)":          "Variance explained by benchmark (monthly OLS)",
    "Max Drawdown":                   "Peak-to-trough decline (daily cumulative returns)",
    "Max Drawdown Duration (days)":   "Longest consecutive streak below previous peak",
    "Max Drawdown Duration (months)": "Same, converted to approximate months (÷ 21)",
    "VaR (Monthly, 5%)":              "5th percentile of monthly returns",
    "CVaR (Monthly, 5%)":             "Mean of returns below VaR (expected shortfall)",
    "Upside Capture %":               "Portfolio return ÷ benchmark in up months × 100",
    "Downside Capture %":             "Portfolio return ÷ benchmark in down months × 100",
    "Skewness":                       "Asymmetry of monthly return distribution",
    "Kurtosis":                       "Excess kurtosis of monthly returns (normal = 0)",
}

# Colour legend entries: {metric: (green_label, orange_label, red_label)}
COLOUR_LEGEND: dict[str, tuple[str, str, str]] = {
    "Sharpe Ratio":                   ("> 1.0",   "0.5 – 1.0",  "< 0.5"),
    "Sortino Ratio":                  ("> 2.0",   "1.0 – 2.0",  "< 1.0"),
    "Calmar Ratio":                   ("> 1.0",   "0.5 – 1.0",  "< 0.5"),
    "Annualized Return":              ("> 10%",   "0 – 10%",    "< 0%"),
    "Annualized Volatility":          ("< 10%",   "10 – 20%",   "> 20%"),
    "R-Squared (Benchmark)":          ("> 0.80",  "0.50 – 0.80","< 0.50"),
    "Max Drawdown":                   ("> -10%",  "-10 – -20%", "< -20%"),
    "Max Drawdown Duration (days)":   ("< 63",    "63 – 126",   "> 126"),
    "VaR (Monthly, 5%)":              ("> -3%",   "-3 – -5%",   "< -5%"),
    "CVaR (Monthly, 5%)":             ("> -5%",   "-5 – -7%",   "< -7%"),
    "Upside Capture %":               ("> 100%",  "50 – 100%",  "< 50%"),
    "Downside Capture %":             ("< 50%",   "50 – 100%",  "> 100%"),
    "Skewness":                       ("> 0",     "-0.5 – 0",   "< -0.5"),
    "Kurtosis":                       ("< 3",     "3 – 5",      "> 5"),
}


def _fmt_value(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return "—"
    fmt = _FMT.get(metric, "2f")
    if fmt == "pct":
        return f"{value:+.2%}"
    if fmt == "0f":
        return f"{int(round(value))}"
    return f"{value:.{fmt}}"


def metrics_html(metrics: dict) -> str:
    """Render the metrics dict as a colour-coded HTML table."""
    BORDER   = "#E2E8F0"
    TEXT     = "#0F172A"
    TEXT_MUT = "#64748B"
    BG_HEAD  = "#F8FAFC"

    # Group metrics into two columns for a compact layout
    items = [(k, v) for k, v in metrics.items() if k in _FMT]
    mid   = (len(items) + 1) // 2
    left  = items[:mid]
    right = items[mid:]

    def _row(metric, value):
        bg, fg = _colour(metric, value)
        val_str = _fmt_value(metric, value)
        desc    = METRIC_DESCRIPTIONS.get(metric, "")
        style_cell = (
            f"background:{bg};color:{fg};font-weight:600;"
            if bg else f"color:{TEXT};"
        )
        return (
            f'<tr style="border-bottom:1px solid {BORDER};">'
            f'<td style="padding:7px 14px;color:{TEXT_MUT};font-size:12px;" title="{desc}">{metric}</td>'
            f'<td style="padding:7px 14px;text-align:right;font-size:13px;{style_cell}">{val_str}</td>'
            f'</tr>'
        )

    def _col(rows_data):
        return (
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr style="background:{BG_HEAD};border-bottom:2px solid {BORDER};">'
            f'<th style="padding:8px 14px;text-align:left;color:{TEXT_MUT};font-size:11px;'
            f'font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Metric</th>'
            f'<th style="padding:8px 14px;text-align:right;color:{TEXT_MUT};font-size:11px;'
            f'font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Value</th>'
            f'</tr></thead><tbody>'
            + "".join(_row(m, v) for m, v in rows_data)
            + "</tbody></table>"
        )

    return (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;'
        f'font-family:Inter,system-ui,sans-serif;">'
        + _col(left)
        + _col(right)
        + "</div>"
    )


def legend_html() -> str:
    """Render the colour-scale legend as a compact HTML block."""
    BORDER   = "#E2E8F0"
    TEXT     = "#0F172A"
    TEXT_MUT = "#64748B"
    BG_HEAD  = "#F8FAFC"

    rows = ""
    for metric, (g, o, r) in COLOUR_LEGEND.items():
        rows += (
            f'<tr style="border-bottom:1px solid {BORDER};">'
            f'<td style="padding:5px 14px;color:{TEXT_MUT};font-size:12px;">{metric}</td>'
            f'<td style="padding:5px 10px;text-align:center;font-size:11px;font-weight:600;'
            f'background:{_GREEN};color:{_GREEN_TEXT};border-radius:4px;">{g}</td>'
            f'<td style="padding:5px 10px;text-align:center;font-size:11px;font-weight:600;'
            f'background:{_ORANGE};color:{_ORANGE_TEXT};border-radius:4px;">{o}</td>'
            f'<td style="padding:5px 10px;text-align:center;font-size:11px;font-weight:600;'
            f'background:{_RED};color:{_RED_TEXT};border-radius:4px;">{r}</td>'
            f'</tr>'
        )

    return (
        f'<div style="font-family:Inter,system-ui,sans-serif;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="background:{BG_HEAD};border-bottom:2px solid {BORDER};">'
        f'<th style="padding:8px 14px;text-align:left;color:{TEXT_MUT};font-size:11px;'
        f'font-weight:500;text-transform:uppercase;letter-spacing:.05em;">Metric</th>'
        f'<th style="padding:8px 14px;text-align:center;color:{_GREEN_TEXT};font-size:11px;'
        f'font-weight:600;background:{_GREEN};">Green</th>'
        f'<th style="padding:8px 14px;text-align:center;color:{_ORANGE_TEXT};font-size:11px;'
        f'font-weight:600;background:{_ORANGE};">Amber</th>'
        f'<th style="padding:8px 14px;text-align:center;color:{_RED_TEXT};font-size:11px;'
        f'font-weight:600;background:{_RED};">Red</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
