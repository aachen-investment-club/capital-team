"""
Regime model zoo and scorecards for the Market Regime page.

Every model maps a price series to the same frame, indexed by date:

    state       "Bull" / "Sideways" / "Bear" (NaN before the model is warmed up)
    confidence  how sure the model is of that state, 0-1 (NaN where the model
                has no probabilistic reading, e.g. the champion)
    reading     the model's own underlying number (slope, probability, distance)

so any two models can be laid side by side, scored on what followed each
label, and checked for agreement.

**Point in time.** The champion and the SMA baseline are causal by
construction: day t uses prices up to t only, so one run over the whole history
is what they would have said on every date. An HMM is not: its parameters are
estimated from a sample. With `fit="asof"` it is fitted on data up to the
as-of date only and then run forward with frozen parameters, so everything
after that date is genuinely out of sample. `fit="full"` uses the whole
history (parameter look-ahead) and `probs="smoothed"` adds the
forward-backward pass (state look-ahead): both are there to show how much
hindsight flatters a regime model, not to be traded on.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from capital.analytics.regime import champion as ch
from capital.analytics.regime.hmm import STATES, _filtered_proba

_EMPTY_FRAME = ["state", "confidence", "reading"]


@dataclass(frozen=True)
class RegimeModel:
    key: str
    label: str
    kind: str                       # "champion" | "hmm" | "sma"
    params: dict = field(default_factory=dict)
    description: str = ""


#: The models the page offers. HMM entries share the page's HMM settings
#: (restarts, fit window, covariance) and differ in states / features / labelling.
MODELS: dict[str, RegimeModel] = {m.key: m for m in [
    RegimeModel("champion", "Champion", "champion",
                description="Kalman trend slope with hysteresis; GJR-GARCH vol nowcast."),
    RegimeModel("hmm3", "HMM 3-state", "hmm", {"n_states": 3, "features": "ret_vol", "label_by": "return"},
                description="Daily return + log 21d realised vol, labelled by mean return."),
    RegimeModel("hmm2", "HMM 2-state", "hmm", {"n_states": 2, "features": "ret_vol", "label_by": "return"},
                description="Bull / Bear only: the classic risk-on / risk-off split."),
    RegimeModel("hmm4", "HMM 4-state", "hmm", {"n_states": 4, "features": "ret_vol", "label_by": "return"},
                description="Four states; the middle two are both reported as Sideways."),
    RegimeModel("hmm3_ret", "HMM 3-state · returns only", "hmm",
                {"n_states": 3, "features": "ret", "label_by": "return"},
                description="Daily return as the only feature: shows why realised vol is needed."),
    RegimeModel("hmm3_vol", "HMM 3-state · vol-labelled", "hmm",
                {"n_states": 3, "features": "ret_vol", "label_by": "vol"},
                description="Same fit as HMM 3-state, labelled by volatility (calmest = Bull)."),
    RegimeModel("sma", "SMA 200 trend", "sma", {"window": 200, "band": 0.02},
                description="Baseline: price vs its 200-day average, ±2% neutral band."),
]}

DEFAULT_MODELS = ("champion", "hmm3", "hmm2", "sma")


def _labels_for(k: int) -> list[str]:
    """Rank 0 (best) .. k-1 (worst) onto the shared three-way scale."""
    return ["Bull" if r == 0 else "Bear" if r == k - 1 else "Sideways" for r in range(k)]


def _hmm_features(px: pd.Series, features: str, rv_window: int) -> pd.DataFrame:
    r = np.log(px).diff().dropna() * 100
    if features == "ret":
        return r.to_frame("ret")
    rv = r.rolling(rv_window).std() * np.sqrt(252)
    return pd.DataFrame({"ret": r, "log_rv": np.log(rv)}).dropna()


def fit_hmm(px: pd.Series, n_states: int = 3, features: str = "ret_vol", label_by: str = "return",
            fit: str = "asof", asof: pd.Timestamp | None = None, probs: str = "filtered",
            covariance: str = "full", restarts: int = 3, rv_window: int = 21, seed: int = 7) -> dict:
    """Gaussian HMM regime, optionally fitted on data up to `asof` only.

    Returns {"frame", "params", "transmat", "train_end", "loglik"}; see the
    module docstring for the frame and for what `fit` / `probs` change.
    """
    from hmmlearn.hmm import GaussianHMM

    feats = _hmm_features(px.dropna().astype(float), features, rv_window)
    train = feats.loc[:pd.Timestamp(asof)] if fit == "asof" and asof is not None else feats
    if len(train) < 250:
        raise ValueError(f"Need at least 250 days to fit an HMM, got {len(train)} before the as-of date.")
    X_train, X = train.to_numpy(dtype=float), feats.to_numpy(dtype=float)

    best, best_ll = None, -np.inf
    for i in range(restarts):
        m = GaussianHMM(n_components=n_states, covariance_type=covariance, n_iter=200,
                        tol=1e-3, random_state=seed + i)
        try:
            m.fit(X_train)
            ll = m.score(X_train)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("HMM failed to converge from every start.")

    covars = best.covars_  # full (k, d, d) whatever covariance_type was fitted
    vol = np.sqrt(covars[:, 0, 0] * 252)
    order = np.argsort(vol) if label_by == "vol" else np.argsort(-best.means_[:, 0])
    names = _labels_for(n_states)

    raw = best.predict_proba(X) if probs == "smoothed" else _filtered_proba(best, X)
    p = pd.DataFrame(raw[:, order], index=feats.index, columns=[f"s{r}" for r in range(n_states)])
    # Collapse the ranked states onto Bull / Sideways / Bear (4-state: two Sideways).
    by_label = pd.DataFrame({s: p[[f"s{r}" for r, n in enumerate(names) if n == s]].sum(axis=1)
                             for s in STATES if s in names})
    state = by_label.idxmax(axis=1)
    frame = pd.DataFrame({"state": state, "confidence": by_label.max(axis=1)})
    frame["reading"] = frame["confidence"]
    frame = frame.join(by_label.add_prefix("p_"))

    A = best.transmat_[np.ix_(order, order)]
    rank_names = [f"{n} ({r + 1})" if names.count(n) > 1 else n for r, n in enumerate(names)]
    params = pd.DataFrame({
        "ann_ret_pct": best.means_[order, 0] * 252,
        "ann_vol_pct": vol[order],
        "exp_duration_d": 1.0 / np.maximum(1.0 - np.diag(A), 1e-9),
        "share": [float((p.idxmax(axis=1) == f"s{r}").mean()) for r in range(n_states)],
    }, index=rank_names)
    return {"frame": frame.reindex(px.index), "params": params,
            "transmat": pd.DataFrame(A, index=rank_names, columns=rank_names),
            "train_end": train.index[-1], "loglik": float(best_ll)}


def sma_regime(px: pd.Series, window: int = 200, band: float = 0.02) -> pd.DataFrame:
    """Bull above the moving average by more than `band`, Bear below by more, else Sideways."""
    px = px.dropna().astype(float)
    gap = px / px.rolling(window).mean() - 1
    state = pd.Series(np.select([gap > band, gap < -band], ["Bull", "Bear"], "Sideways"),
                      index=px.index).where(gap.notna())
    return pd.DataFrame({"state": state, "confidence": np.nan, "reading": gap})


def champion_regime(px: pd.Series, config: ch.ChampionConfig = ch.CHAMPION) -> pd.DataFrame:
    """The champion in the common frame, plus its volatility axis and readings."""
    reg = ch.classify(px, config)
    return pd.DataFrame({
        "state": reg["direction_name"], "confidence": np.nan, "reading": reg["slope"],
        "vol_state": reg["volatility_name"], "vol_nowcast": reg["vol_nowcast"],
    })


# ── Scoring ───────────────────────────────────────────────────────────────────

def forward_returns(px: pd.Series, horizon: int) -> pd.DataFrame:
    """Per day t: log return over t..t+h and annualised realised vol over t+1..t+h."""
    lp = np.log(px.dropna().astype(float))
    r = lp.diff()
    fwd_vol = r.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
    return pd.DataFrame({"ret": lp.shift(-horizon) - lp, "vol": fwd_vol})


def days_in_state(state: pd.Series, asof: pd.Timestamp) -> int:
    s = state.loc[:asof].dropna()
    if s.empty:
        return 0
    v = s.to_numpy()
    changed = np.flatnonzero(v != v[-1])
    return int(len(v) - (changed[-1] + 1)) if len(changed) else len(v)


def scorecard(state: pd.Series, px: pd.Series, asof: pd.Timestamp, horizon: int = 21) -> dict:
    """How well a model's labels separated what came next.

    Two samples: `before` = days whose forward window had closed by the as-of
    date (what you could have checked then); `after` = days after the as-of
    date (the future, relative to that date). Each holds per-state share of
    days, mean forward return (annualised) and forward vol, the Bull-minus-Bear
    return spread, and switches per year.
    """
    fwd = forward_returns(px, horizon)
    df = fwd.join(state.rename("state")).dropna()
    idx = px.dropna().index
    pos = idx.searchsorted(asof, side="right") - 1
    closed = idx[max(pos - horizon, 0)]
    out = {}
    for name, part in (("before", df.loc[:closed]), ("after", df.loc[df.index > asof])):
        if len(part) < 2 * horizon:
            out[name] = None
            continue
        g = part.groupby("state")
        by = pd.DataFrame({"share": g.size() / len(part),
                           "ann_ret": g["ret"].mean() * 252 / horizon,
                           "fwd_vol": g["vol"].mean(),
                           "days": g.size()}).reindex(list(STATES))
        lab = state.loc[part.index[0]:part.index[-1]].dropna()
        years = max((lab.index[-1] - lab.index[0]).days / 365.25, 1e-9)
        out[name] = {
            "by_state": by,
            "spread": float(by.loc["Bull", "ann_ret"] - by.loc["Bear", "ann_ret"]),
            "vol_spread": float(by.loc["Bear", "fwd_vol"] - by.loc["Bull", "fwd_vol"]),
            "switches_per_year": float((lab != lab.shift()).iloc[1:].sum() / years),
            "n_days": int(len(part)),
        }
    return out


def agreement(states: dict[str, pd.Series], start=None, end=None) -> pd.DataFrame:
    """Share of days on which each pair of models gives the same label."""
    df = pd.DataFrame(states).loc[start:end]
    names = list(states)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            # Column by column: df[[a, a]] on the diagonal would return a frame.
            sa, sb = df[a], df[b]
            ok = sa.notna() & sb.notna()
            if ok.any():
                out.loc[a, b] = float((sa[ok] == sb[ok]).mean())
    return out


def what_happened_next(px: pd.Series, asof: pd.Timestamp,
                       horizons: tuple[int, ...] = (21, 63, 126, 252)) -> pd.DataFrame:
    """Realised path after the as-of date: return, worst drawdown and vol per horizon
    (rows only for horizons the data has reached)."""
    px = px.dropna().astype(float)
    after = px.loc[asof:]
    if after.empty:
        return pd.DataFrame()
    p0 = after.iloc[0]
    rows = []
    for h in horizons:
        if len(after) <= h:
            continue
        path = after.iloc[:h + 1]
        r = np.log(path).diff().dropna()
        rows.append({"horizon_d": h, "end": path.index[-1], "return": float(path.iloc[-1] / p0 - 1),
                     "max_drawdown": float((path / path.cummax() - 1).min()),
                     "ann_vol": float(r.std() * np.sqrt(252))})
    return pd.DataFrame(rows)
