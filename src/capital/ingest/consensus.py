"""
Analyst consensus and expected report dates for the positions held, from LSEG.

One `get_data` request per night, stored as a dated snapshot, so the table also
builds a history of estimate and target revisions. Only held positions are
fetched: a name bought today appears after the next run.

The price is pulled with the target rather than read from `eod_prices`, so
upside compares two numbers from the same moment. Currencies are stored because
they can differ: a depositary receipt carries the consensus of its underlying
(CATH.SI trades in SGD, its target is CATL's in CNY).
"""
import logging

import pandas as pd

from capital.data import loaders, store

log = logging.getLogger(__name__)

#: Store column -> TR field. Columns are mapped by position: LSEG labels are
#: decorated, and FQ1 and FY1 EPS even come back under the same label.
FIELDS = {
    "expected_report_date": "TR.ExpectedReportDate",
    "eps_next_q": "TR.EPSMean(Period=FQ1)",
    "eps_next_q_end": "TR.EPSMean(Period=FQ1).periodenddate",
    "eps_fy": "TR.EPSMean(Period=FY1)",
    "eps_ccy": "TR.EPSMean(Period=FQ1).currency",
    "rec_label": "TR.RecLabel",
    "n_analysts": "TR.NumOfRecommendations",
    "price_target": "TR.PriceTargetMean",
    "n_targets": "TR.PriceTargetNumOfEstimates",
    "target_ccy": "TR.PriceTargetMean.currency",
    "price": "TR.PriceClose",
    "price_ccy": "TR.PriceClose.currency",
    # how far analysts disagree
    "target_high": "TR.PriceTargetHigh",
    "target_low": "TR.PriceTargetLow",
    "target_std": "TR.PriceTargetStdDev",
    "eps_fy_std": "TR.EPSStdDev(Period=FY1)",
    "n_strong_buy": "TR.NumOfStrongBuy",
    "n_buy": "TR.NumOfBuy",
    "n_hold": "TR.NumOfHold",
    "n_sell": "TR.NumOfSell",
    "n_strong_sell": "TR.NumOfStrongSell",
    # where the consensus stood, so revisions need no snapshot history
    "rec_mean": "TR.RecMean",
    "rec_mean_90d": "TR.RecMean(SDate=-90D)",
    "target_30d": "TR.PriceTargetMean(SDate=-30D)",
    "target_90d": "TR.PriceTargetMean(SDate=-90D)",
    "eps_fy_30d": "TR.EPSMean(Period=FY1,SDate=-30D)",
    "eps_fy_90d": "TR.EPSMean(Period=FY1,SDate=-90D)",
    "eps_fy_end": "TR.EPSMean(Period=FY1).periodenddate",
    "eps_fy_end_90d": "TR.EPSMean(Period=FY1,SDate=-90D).periodenddate",
}
_DATES = ("expected_report_date", "eps_next_q_end", "eps_fy_end", "eps_fy_end_90d")
_TEXT = ("eps_ccy", "rec_label", "target_ccy", "price_ccy")


def run_consensus() -> dict:
    """Snapshot consensus for every held position. Needs an open LSEG session."""
    import lseg.data as ld

    held = loaders.get_held_positions().dropna(subset=["ric"])
    if held.empty:
        return {"securities": 0}
    raw = ld.get_data(universe=held["ric"].tolist(), fields=list(FIELDS.values()))
    if raw is None or raw.shape != (len(held), len(FIELDS) + 1):
        raise ValueError(f"LSEG returned shape {None if raw is None else raw.shape} for {len(held)} "
                         f"securities and {len(FIELDS)} fields; cannot map them by position.")

    df = raw.iloc[:, 1:].copy()
    df.columns = list(FIELDS)
    for col in df.columns:
        if col in _DATES:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif col in _TEXT:
            df[col] = df[col].astype(object).where(df[col].notna(), None)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df.insert(0, "security_id", raw.iloc[:, 0].map(dict(zip(held["ric"], held["security_id"]))))
    df.insert(1, "date", pd.Timestamp.today().normalize())
    df = df.dropna(subset=["security_id"])
    store.flush("consensus", [df], bump=True)
    return {"securities": len(df), "with_consensus": int(df["price_target"].notna().sum())}
