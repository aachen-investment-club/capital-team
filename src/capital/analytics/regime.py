"""
Market regime detection — a 3-state Gaussian hidden Markov model (hmmlearn).

The market is assumed to switch between a small number of unobserved states,
each with its own return distribution. Two features per day:

    r_t        daily log return (%)
    rv_t       21-day realised volatility (annualised %), log-scaled

Returns alone separate states poorly (daily noise swamps the mean); adding
realised vol lets the model distinguish "calm, drifting up" from "volatile,
falling" even when a single day's return is ambiguous.

Two details decide whether the output can be trusted:

- **Label switching.** EM returns states in arbitrary order, so "state 0" can be
  the bull state one night and the bear state the next. States are therefore
  relabelled after every fit by their mean return: highest = Bull, lowest =
  Bear, the middle one = Sideways.
- **Filtered, not smoothed, probabilities.** hmmlearn's `predict_proba` runs the
  forward-backward pass, so the probability on day t uses returns from t+1
  onwards. That is fine for describing history but is look-ahead for "what is
  the state today?" and for any backtest. The forward pass below uses data up
  to t only; the last row is the honest current-state estimate.
"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

STATES = ("Bull", "Sideways", "Bear")
STATE_COLORS = {"Bull": "#2E7D6B", "Sideways": "#3A6B9C", "Bear": "#C0392B"}

_RV_WINDOW = 21


def regime_features(returns_pct: pd.Series) -> pd.DataFrame:
    """Daily log return (%) and log 21-day realised vol, aligned and NaN-free."""
    r = returns_pct.dropna()
    rv = r.rolling(_RV_WINDOW).std() * np.sqrt(252)
    return pd.DataFrame({"ret": r, "log_rv": np.log(rv)}).dropna()


def _filtered_proba(model, X: np.ndarray) -> np.ndarray:
    """Forward-algorithm state probabilities P(s_t | x_1..x_t), in a stable log form."""
    k = model.n_components
    covars = model.covars_  # full (k, d, d) whatever covariance_type was fitted
    log_b = np.column_stack([
        multivariate_normal(mean=model.means_[j], cov=covars[j], allow_singular=True).logpdf(X)
        for j in range(k)
    ])
    log_A = np.log(np.maximum(model.transmat_, 1e-300))
    out = np.empty_like(log_b)
    log_alpha = np.log(np.maximum(model.startprob_, 1e-300)) + log_b[0]
    out[0] = log_alpha - np.logaddexp.reduce(log_alpha)
    for t in range(1, len(X)):
        prev = out[t - 1][:, None] + log_A          # P(s_{t-1}) * A, in logs
        log_alpha = np.logaddexp.reduce(prev, axis=0) + log_b[t]
        out[t] = log_alpha - np.logaddexp.reduce(log_alpha)
    return np.exp(out)


def fit_regime_model(returns_pct: pd.Series, n_restarts: int = 6,
                     seed: int = 7) -> dict:
    """Fit the 3-state HMM and return labelled, filtered state probabilities.

    Several random restarts are fitted and the highest log-likelihood kept: EM
    finds a local optimum, and with three states a bad start regularly collapses
    two of them into one.

    Returns
    -------
    dict: {
        "proba":      DataFrame (date x Bull/Sideways/Bear), filtered probabilities,
        "state":      Series of the most likely state per day,
        "transmat":   DataFrame, row = from-state, column = to-state,
        "params":     DataFrame per state: mean daily return (%), ann. vol (%),
                      expected duration (days), share of history,
        "current":    dict for today: state, probabilities, days in regime,
                      expected duration, probability of still being there in 21d,
        "loglik":     float,
    }
    """
    from hmmlearn.hmm import GaussianHMM

    feats = regime_features(returns_pct)
    if len(feats) < 250:
        raise ValueError(f"Need at least 250 days for a regime model, got {len(feats)}.")
    X = feats.to_numpy(dtype=float)

    best, best_ll = None, -np.inf
    for i in range(n_restarts):
        m = GaussianHMM(n_components=3, covariance_type="full", n_iter=300,
                        tol=1e-4, random_state=seed + i)
        try:
            m.fit(X)
            ll = m.score(X)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("HMM failed to converge from every start.")

    # Relabel by mean return: highest -> Bull, lowest -> Bear.
    order = np.argsort(-best.means_[:, 0])
    labels = {int(raw): STATES[rank] for rank, raw in enumerate(order)}

    proba_raw = _filtered_proba(best, X)
    proba = pd.DataFrame(proba_raw, index=feats.index).rename(columns=labels)[list(STATES)]
    state = proba.idxmax(axis=1)

    A = pd.DataFrame(best.transmat_).rename(index=labels, columns=labels)
    A = A.loc[list(STATES), list(STATES)]

    covars = best.covars_
    params = pd.DataFrame({
        "mean_ret_pct": [best.means_[raw, 0] for raw in order],
        "ann_vol_pct": [np.sqrt(covars[raw][0, 0] * 252) for raw in order],
        "exp_duration_d": [1.0 / max(1.0 - A.loc[s, s], 1e-9) for s in STATES],
        "share": [float((state == s).mean()) for s in STATES],
    }, index=list(STATES))
    params["ann_ret_pct"] = params["mean_ret_pct"] * 252

    cur = state.iloc[-1]
    changed = state.ne(cur)
    days_in = int(len(state) - (np.flatnonzero(changed.to_numpy())[-1] + 1)) \
        if changed.any() else len(state)
    p_stay = float(A.loc[cur, cur])
    # 21-day-ahead state distribution: today's filtered row times A^21.
    ahead = proba.iloc[-1].to_numpy() @ np.linalg.matrix_power(A.to_numpy(), 21)

    current = {
        "date": proba.index[-1],
        "state": cur,
        "proba": proba.iloc[-1].to_dict(),
        "days_in_regime": days_in,
        "exp_duration_d": float(params.loc[cur, "exp_duration_d"]),
        "p_stay_1d": p_stay,
        "proba_21d": dict(zip(STATES, ahead)),
    }
    return {"proba": proba, "state": state, "transmat": A, "params": params,
            "current": current, "loglik": float(best_ll)}


def regime_conditional_returns(period_returns: pd.DataFrame, state: pd.Series,
                               periods_per_year: int) -> pd.DataFrame:
    """Average return of each column in each regime, without look-ahead.

    Each period is assigned the regime that was most likely at the *start* of
    the period (the last daily state on or before the previous period end), so
    the table answers "given the regime we can see today, what has each strategy
    tended to earn next?" rather than "what did it earn in hindsight?".

    Returns a long frame: strategy, regime, ann_return, hit_rate, n_periods.
    """
    if period_returns.empty or state.empty:
        return pd.DataFrame(columns=["strategy", "regime", "ann_return", "hit_rate", "n_periods"])
    starts = period_returns.index.to_series().shift(1).dropna()
    known = state.reindex(state.index.union(starts.values)).ffill()
    regime_at_start = pd.Series(known.reindex(starts.values).to_numpy(), index=starts.index)
    df = period_returns.loc[regime_at_start.index].copy()
    df["regime"] = regime_at_start
    df = df.dropna(subset=["regime"])
    rows = []
    for regime, block in df.groupby("regime"):
        vals = block.drop(columns="regime")
        for col in vals.columns:
            s = vals[col].dropna()
            if s.empty:
                continue
            rows.append({"strategy": col, "regime": regime,
                         "ann_return": float(s.mean() * periods_per_year),
                         "hit_rate": float((s > 0).mean()),
                         "n_periods": int(len(s))})
    return pd.DataFrame(rows)
