import sys
import pathlib

from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns
from pypfopt.efficient_frontier import EfficientFrontier

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.theme import inject_css, FAVICON, NAVY
from lib.data import (
    get_daily_weightings_history,
    get_security_master,
    get_eod_prices,
    _eod_data_version,
)

st.set_page_config(page_title="Black-Litterman Model · AIC", page_icon=FAVICON, layout="wide")
inject_css()
st.title("Black-Litterman Model")
st.caption("The model computes stable expected returns, composed of equilibrium and own views.")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _latest_weights() -> pd.Series:
    """
    Latest portfolio weights (symbol -> decimal), excluding cash.
    """
    dw = get_daily_weightings_history()
    dw = dw[dw["category"] != "Cash"]
    latest_date = dw["date"].max()
    latest = dw[dw["date"] == latest_date].set_index("symbol")["pct_nav"]
    return latest / latest.sum()


@st.cache_data(ttl=3600)
def _eod_returns_matrix(symbols: tuple[str, ...], lookback_years: int) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Daily price returns matrix (date x symbol), built from each security's own
    EOD price history rather than the portfolio's ~2-month track record — the
    underlying securities have traded for years, so this gives Sigma a much
    longer, less noisy reference window. Restricted to the last `lookback_years`.
    Also returns (symbol, reason) pairs for symbols dropped, so the page can tell
    the user exactly why (no security-master mapping / no EOD data ingested yet /
    a trading history shorter than the requested window) instead of a vague catch-all.
    """
    ticker_to_id = get_security_master().set_index("ticker")["security_id"]
    version = _eod_data_version()
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=lookback_years)

    prices: dict[str, pd.Series] = {}
    dropped: list[tuple[str, str]] = []
    for sym in symbols:
        sec_id = ticker_to_id.get(sym)
        if sec_id is None:
            dropped.append((sym, "no security-master mapping"))
            continue
        df = get_eod_prices(sec_id, cache_version=version)
        if df.empty:
            dropped.append((sym, "no EOD data ingested yet"))
            continue
        px = df.set_index("date")["adj_close"].sort_index()
        first_trade = px.index.min()
        px = px[px.index >= cutoff]
        if first_trade > cutoff + pd.Timedelta(days=30):
            dropped.append((sym, f"only trading since {first_trade:%d %b %Y}"))
            continue
        prices[sym] = px

    price_matrix = pd.DataFrame(prices).sort_index()
    returns = price_matrix.pct_change(fill_method=None).dropna(how="any")

    return returns, sorted(dropped)


def _latex_matrix(df, label: str, fmt) -> str:
    """
    Render a DataFrame's values as a plain bracketed matrix, prefixed by its symbol — no embedded row/column labels.
    """
    body = r" \\ ".join(
        " & ".join(fmt(v) for v in df.loc[idx]) for idx in df.index
    )
    bracketed = r"\left[\begin{array}{" + "r" * len(df.columns) + "}" + body + r"\end{array}\right]"
    return f"{label} = " + bracketed


def _latex_vector(s, label: str, fmt) -> str:
    """
    Render a Series' values as a plain bracketed row vector, prefixed by its symbol — no embedded column labels.
    """
    body = " & ".join(fmt(v) for v in s.values)
    bracketed = r"\left[\begin{array}{" + "r" * len(s) + "}" + body + r"\end{array}\right]"
    return f"{label} = " + bracketed


def _fmt_pct(v: float, signed: bool = False, decimals: int = 1) -> str:
    s = f"{v:+.{decimals}%}" if signed else f"{v:.{decimals}%}"
    return s.replace("%", r"\%")


def _fmt_num(v: float, decimals: int = 4) -> str:
    r"""
    Pad non-negative values with a same-width \kern (pure spacing, no glyph) so the
    "-" on negative values elsewhere in the column doesn't shift its digits out of
    alignment. \kern is used instead of \phantom{-} because a phantom glyph is still
    present in the DOM (just invisible) and can show up as a stray character when
    the rendered matrix is text-selected.
    """
    s = f"{abs(v):.{decimals}f}"
    return f"-{s}" if v < 0 else rf"\kern{{0.6em}}{s}"


# ─────────────────────────────────────────────────────────────────────────────
# Model Inputs
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Inputs")


# ─── Prior ───────────────────────────────────────────────────────────────────
st.markdown("##### Prior")

with st.expander("Show approach"):
    st.markdown(
        "There is no external market-cap benchmark for this set of holdings, so the "
        "portfolio's own current weights *w* stand in for \"market\" weights — the "
        "current book is treated as the equilibrium to reverse-optimize from.\n\n"
        "- **Σ** (covariance): estimated from each security's own EOD price history "
        "(years, not the portfolio's ~2-month track record), with Ledoit-Wolf shrinkage "
        "on top of the raw sample covariance for extra stability.\n"
        "- **π** (prior/implied returns): `π = 𝛿 · Σ · w` — the returns that would "
        "make the current weights mean-variance optimal, given Σ and risk aversion 𝛿."
    )

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    delta = st.number_input(
        "Risk aversion (𝛿)",
        min_value=1.0, max_value=10.0, value=2.5,
        help="Common way to calculate: "
             "\"Sharpe Ratio\" / \"Portfolio standard deviation\"",
    )

with c2:
    tau = st.number_input(
        "Proportionality (𝜏)",
        min_value=0.01, max_value=1.0, value=0.05, step=0.01,
        help="𝜏 is the proportionality factor for the covariance matrix "
             "of historic returns to get the prior covariance matrix."
    )

with c3:
    lookback_years = st.number_input(
        "Covariance window (years)",
        min_value=1, max_value=10, value=3,
        help="How far back to pull each security's own EOD price history for Σ. "
             "Longer = more stable estimate, but less reflective of the current regime.",
    )


# ─── Views ───────────────────────────────────────────────────────────────────
st.markdown("##### Views")

with st.expander("Show approach"):
    st.markdown(
        "Each row is one view, expressed either as an **absolute** view "
        "(\"AAPL will return +8% p.a.\") or a **relative** view "
        "(\"AAPL will outperform MSFT by +3% p.a.\"). Confidence (0–100%) is converted "
        "into the view-uncertainty matrix **Ω** via Idzorek's method — low confidence "
        "pulls the posterior back toward the prior π, high confidence pulls it toward "
        "the view itself.\n\n"
        "PyPortfolioOpt computes Ω via the closed-form solution Jay Walters derived "
        "as equivalent to Idzorek's original (2005) iterative-optimization approach "
        "(*The Black-Litterman Model in Detail*, 2014), rather than running that "
        "iterative search directly.\n\n"
        "No views entered ⇒ the posterior equals the prior."
    )

# Cheap, cached preview of the investable universe at the current lookback — just to
# populate the views editor's asset dropdown before Run is clicked.
_preview_weights = _latest_weights()
_preview_returns, _ = _eod_returns_matrix(tuple(_preview_weights.index), lookback_years)
asset_list = sorted(_preview_returns.columns.tolist()) if not _preview_returns.empty else sorted(_preview_weights.index.tolist())

views_seed = pd.DataFrame(
    [{"Type": "Absolute", "Asset": asset_list[0], "Vs Asset": "—", "View (% p.a.)": 0.0, "Confidence (%)": 50.0}]
).iloc[0:0]  # empty, correctly typed

views_df = st.data_editor(
    views_seed,
    num_rows="dynamic",
    width="stretch",
    key="bl_views_editor",
    column_config={
        "Type": st.column_config.SelectboxColumn(options=["Absolute", "Relative"], required=True, default="Absolute"),
        "Asset": st.column_config.SelectboxColumn(options=asset_list, required=True, default=asset_list[0]),
        "Vs Asset": st.column_config.SelectboxColumn(options=["—"] + asset_list, default="—"),
        "View (% p.a.)": st.column_config.NumberColumn(format="%.2f", step=0.5, default=0.0),
        "Confidence (%)": st.column_config.NumberColumn(min_value=1.0, max_value=100.0, step=5.0, default=50.0),
    },
)

st.markdown("##### Recommended Portfolio")

with st.expander("Show approach"):
    st.markdown(
        "Standard Markowitz mean-variance optimization (via PyPortfolioOpt's "
        "`EfficientFrontier`), but fed the Black-Litterman posterior instead of raw "
        "historical estimates — expected returns and risk reflect the equilibrium prior "
        "as tilted by your views, not just the sample mean.\n\n"
        "- **Max Sharpe**: maximises `(E[R] - r_f) / σ`, the risk-adjusted excess return "
        "over the risk-free rate.\n"
        "- **Min Volatility**: minimises portfolio σ regardless of expected return.\n\n"
        "Constrained long-only and fully invested — weights are bounded to [0, 1] and "
        "sum to 100%, so the optimizer can't short or leave cash on the sidelines."
    )

c1, c2 = st.columns([1, 1])
with c1:
    objective = st.selectbox("Objective", ["Max Sharpe", "Min Volatility"])
with c2:
    rf_pct = st.number_input("Risk-free rate (% p.a.)", min_value=0.0, max_value=10.0, value=3.5, step=0.25)

if st.button("Run", type="primary"):
    st.session_state["bl_committed"] = {
        "delta": delta,
        "tau": tau,
        "lookback_years": lookback_years,
        "views_df": views_df.copy(),
        "objective": objective,
        "rf_pct": rf_pct,
    }

committed = st.session_state.get("bl_committed")
if not committed:
    st.info("Configure the inputs above and click **Run** to calculate the model.")
    st.stop()

delta = committed["delta"]
tau = committed["tau"]
lookback_years = committed["lookback_years"]
views_df = committed["views_df"]
objective = committed["objective"]
rf_pct = committed["rf_pct"]


# ─────────────────────────────────────────────────────────────────────────────
# Model Output
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Output")


# ─────────────────────────────────────────────────────────────────────────────
# Prior Distribution
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("##### Prior Distribution")

weights = _latest_weights()
returns, excluded_symbols = _eod_returns_matrix(tuple(weights.index), lookback_years)

if returns.empty:
    reasons = "; ".join(f"{sym} ({reason})" for sym, reason in excluded_symbols) or "(none)"
    st.error(
        "No EOD price history available for any current holding — cannot compute Σ. "
        f"Excluded: {reasons}. Run `scripts/ingest_eod.py` to backfill EOD prices."
    )
    st.stop()

weights = weights[returns.columns] / weights[returns.columns].sum()

n_obs = len(returns)
warn_parts = []
if excluded_symbols:
    warn_parts.append(
        f"Excluded from the {lookback_years}-year Black-Litterman universe:\n"
        + "\n".join(f"- **{sym}** — {reason}" for sym, reason in excluded_symbols)
    )
if n_obs < 60:
    warn_parts.append(
        f"Only {n_obs} daily observations ({returns.index.min():%d %b %Y} – "
        f"{returns.index.max():%d %b %Y}) remain — the covariance matrix and implied "
        "prior are still statistically noisy. Using Ledoit-Wolf shrinkage for Σ to help stabilise it."
    )
if warn_parts:
    st.warning("\n\n".join(warn_parts))

sigma = CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()
pi = market_implied_prior_returns(weights, delta, sigma)

with st.expander("Show step-by-step calculation", expanded=False):
    st.markdown("##### Step 1 — Daily returns matrix (date × symbol)")
    st.caption(
        f"{n_obs} observations across {len(returns.columns)} symbols "
        f"({returns.index.min():%d %b %Y} – {returns.index.max():%d %b %Y}) — last 10 rows shown."
    )
    st.dataframe(returns.tail(10).style.format("{:+.2%}"), width="stretch")

    st.markdown("##### Step 2 — Latest weights *w* (stand-in for market weights)")
    st.markdown("Transposed matrix:")
    st.caption("Columns (left→right): " + ", ".join(weights.index))
    st.latex(_latex_vector(weights, "w^T", lambda v: _fmt_pct(v, decimals=1)))

    st.markdown("##### Step 3 — Covariance matrix Σ (Ledoit-Wolf shrinkage, annualised)")
    st.caption("Rows and columns (in order): " + ", ".join(weights.index))
    st.latex(_latex_matrix(sigma, r"\Sigma", lambda v: _fmt_num(v, 4)))

    st.markdown("##### Step 4 — Implied equilibrium returns π = 𝛿 · Σ · w")
    st.markdown("Transposed matrix:")
    st.caption("Columns (left→right): " + ", ".join(weights.index))
    st.latex(_latex_vector(pi, r"\pi^T", lambda v: _fmt_pct(v, signed=True, decimals=2)))


# ─────────────────────────────────────────────────────────────────────────────
# Conditional Distribution (Views)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("##### Conditional Distribution (Views)")

asset_pos = {a: i for i, a in enumerate(weights.index)}
n_assets = len(asset_pos)

P_rows, Q_vals, conf_vals, view_labels, bad_rows = [], [], [], [], []
for _, row in views_df.iterrows():
    vtype, asset, vs_asset, view_pct, conf_pct = (
        row.get("Type"), row.get("Asset"), row.get("Vs Asset"), row.get("View (% p.a.)"), row.get("Confidence (%)")
    )
    if not asset or pd.isna(view_pct) or pd.isna(conf_pct):
        continue
    if asset not in asset_pos:
        bad_rows.append(f"{asset} is not in the current universe")
        continue

    p_row = np.zeros(n_assets)
    if vtype == "Relative":
        if not vs_asset or vs_asset == "—" or vs_asset not in asset_pos:
            bad_rows.append(f"Relative view on {asset} needs a valid \"Vs Asset\"")
            continue
        if vs_asset == asset:
            bad_rows.append(f"Relative view on {asset} can't reference itself")
            continue
        p_row[asset_pos[asset]] = 1.0
        p_row[asset_pos[vs_asset]] = -1.0
        view_labels.append(f"{asset} outperforms {vs_asset} by {view_pct:+.1f}% p.a. (confidence {conf_pct:.0f}%)")
    else:
        p_row[asset_pos[asset]] = 1.0
        view_labels.append(f"{asset} returns {view_pct:+.1f}% p.a. (confidence {conf_pct:.0f}%)")

    P_rows.append(p_row)
    Q_vals.append(view_pct / 100.0)
    conf_vals.append(conf_pct / 100.0)

if bad_rows:
    st.warning("Skipped invalid view rows: " + "; ".join(bad_rows) + ".")

bl_model = None
if P_rows:
    P = np.vstack(P_rows)
    Q = np.array(Q_vals)
    confidences = np.array(conf_vals)
    bl_model = BlackLittermanModel(
        sigma, pi=pi, Q=Q, P=P, omega="idzorek", view_confidences=confidences,
        tau=tau, risk_aversion=delta,
    )
    st.markdown("###### Views entered")
    st.dataframe(
        pd.DataFrame({"View": view_labels, "Ω (view variance)": np.diag(bl_model.omega)}),
        width="stretch", hide_index=True,
    )

    with st.expander("Show step-by-step calculation", expanded=False):
        view_ids = [f"View {i + 1}" for i in range(len(view_labels))]
        sorted_assets = sorted(asset_pos, key=lambda a: asset_pos[a])

        st.markdown("##### Step 1 — Pick matrix P (view × asset)")
        st.caption(
            "Each row is one view: +1 on the asset it's about, and for relative views, "
            "-1 on the comparison asset. Rows follow the view order in the table above.\n\n"
            "Columns (left→right): " + ", ".join(sorted_assets)
        )
        P_df = pd.DataFrame(P, index=view_ids, columns=sorted_assets)
        st.latex(_latex_matrix(P_df, "P", lambda v: _fmt_num(v, 0)))

        st.markdown("##### Step 2 — View vector Q (annualised view return)")
        st.markdown("Transposed matrix:")
        st.caption("Entries follow the view order in the table above.")
        Q_series = pd.Series(Q, index=view_ids)
        st.latex(_latex_vector(Q_series, "Q^T", lambda v: _fmt_pct(v, signed=True, decimals=2)))

        st.markdown(
            "##### Step 3 — Uncertainty matrix Ω = diag(𝜏·𝛼ₖ·Pₖ·Σ·Pₖᵀ), 𝛼ₖ = (1−cₖ)/cₖ "
            "(Idzorek, from confidence)"
        )
        st.caption(
            "Per view: your entered confidence cₖ converts to 𝛼ₖ, which scales Σ's own "
            "prior variance for that view's portfolio Pₖ into the view's uncertainty Ωₖ. "
            "100% confidence ⇒ 𝛼ₖ=0 ⇒ Ωₖ=0 (view treated as certain); low confidence ⇒ "
            "large 𝛼ₖ ⇒ large Ωₖ (view barely moves the posterior). "
            "Rows follow the view order in the table above."
        )
        sigma_arr = np.asarray(sigma)
        alpha = (1 - confidences) / confidences
        prior_view_var = np.array([P[k] @ sigma_arr @ P[k] for k in range(len(P))])
        idzorek_df = pd.DataFrame(
            {
                "Confidence cₖ": confidences,
                "𝛼ₖ = (1−cₖ)/cₖ": alpha,
                "Pₖ·Σ·Pₖᵀ": prior_view_var,
                "Ωₖ = 𝜏·𝛼ₖ·Pₖ·Σ·Pₖᵀ": np.diag(bl_model.omega),
            },
            index=view_ids,
        )
        st.dataframe(idzorek_df.style.format("{:.4f}"), width="stretch")

        st.caption("Rows and columns (in order): " + ", ".join(view_ids))
        omega_df = pd.DataFrame(bl_model.omega, index=view_ids, columns=view_ids)
        st.latex(_latex_matrix(omega_df, r"\Omega", lambda v: _fmt_num(v, 4)))
else:
    st.info("No views entered yet — add a row above to tilt the prior. Until then, the posterior equals the prior.")


# ─────────────────────────────────────────────────────────────────────────────
# Posterior Distribution
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("##### Posterior Distribution")

if bl_model is not None:
    posterior_rets = bl_model.bl_returns()
    posterior_cov = bl_model.bl_cov()

    st.markdown("###### Posterior expected returns — prior π vs posterior")
    cmp = pd.DataFrame({"Prior π": pi, "Posterior": posterior_rets})
    st.dataframe(cmp.T.style.format("{:+.2%}"), width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=cmp.index, y=cmp["Prior π"], name="Prior π", marker_color=NAVY))
    fig.add_trace(go.Bar(x=cmp.index, y=cmp["Posterior"], name="Posterior", marker_color="#B8962E"))
    fig.update_layout(
        title="Expected Returns: Prior vs Posterior",
        yaxis_tickformat=".1%", yaxis_title="Expected return (ann.)", xaxis_title="",
        barmode="group", template="capital",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Show step-by-step calculation", expanded=False):
        st.markdown("##### Step 1 — Posterior returns 𝜇 = 𝜋 + 𝜏Σ·Pᵀ·(P𝜏ΣPᵀ+Ω)⁻¹·(Q−P𝜋)")
        st.caption("Columns (left→right): " + ", ".join(weights.index))
        st.latex(_latex_vector(posterior_rets, r"\mu", lambda v: _fmt_pct(v, signed=True, decimals=2)))

        st.markdown("##### Step 2 — M = 𝜏Σ − 𝜏ΣPᵀ·(P𝜏ΣPᵀ+Ω)⁻¹·P𝜏Σ (posterior covariance of the mean estimate)")
        st.caption(
            "Uncertainty about the *estimate* of returns after updating on your views — "
            "shrinks toward zero the more (and more confident) views you add.\n\n"
            "Rows and columns (in order): " + ", ".join(weights.index)
        )
        M = posterior_cov - sigma
        st.latex(_latex_matrix(M, "M", lambda v: _fmt_num(v, 4)))

        st.markdown("##### Step 3 — Posterior covariance Σ_post = Σ + M")
        st.caption(
            "Σ (uncertainty in returns themselves) plus M (uncertainty in the mean estimate) — "
            "always ≥ Σ, since views only ever shrink M, never go negative.\n\n"
            "Rows and columns (in order): " + ", ".join(weights.index)
        )
        st.latex(_latex_matrix(posterior_cov, r"\Sigma_{post}", lambda v: _fmt_num(v, 4)))
else:
    posterior_rets = pi
    posterior_cov = sigma
    st.info("No views entered — showing the prior distribution (π, Σ) unchanged.")


# ─────────────────────────────────────────────────────────────────────────────
# Recommended Portfolio
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recommended Portfolio")
st.caption(f"Mean-variance optimization on the posterior (π, Σ) — long-only, fully invested. Objective: {objective}.")

rf = rf_pct / 100.0
try:
    ef = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0, 1))
    if objective == "Max Sharpe":
        ef.max_sharpe(risk_free_rate=rf)
    else:
        ef.min_volatility()
    rec_weights = pd.Series(ef.clean_weights())
    exp_ret, exp_vol, sharpe = ef.portfolio_performance(risk_free_rate=rf)
except Exception as e:
    st.error(f"Optimization failed: {e}")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Expected Return (ann.)", f"{exp_ret:.1%}")
m2.metric("Volatility (ann.)", f"{exp_vol:.1%}")
m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

cmp_w = pd.DataFrame({"Current": weights, "Recommended": rec_weights}).fillna(0.0)
cmp_w = cmp_w.sort_values("Recommended", ascending=True)

fig = go.Figure()
fig.add_trace(go.Bar(
    y=cmp_w.index, x=cmp_w["Current"], name="Current", orientation="h",
    marker_color=NAVY, hovertemplate="%{y}: %{x:.1%}<extra>Current</extra>",
))
fig.add_trace(go.Bar(
    y=cmp_w.index, x=cmp_w["Recommended"], name="Recommended", orientation="h",
    marker_color="#B8962E", hovertemplate="%{y}: %{x:.1%}<extra>Recommended</extra>",
))
fig.update_layout(
    title=f"Portfolio Weights: Current vs Recommended ({objective})",
    barmode="group", xaxis_title="Weight", yaxis_title="", xaxis_tickformat=".0%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="capital", height=max(400, 32 * len(cmp_w)),
)
st.plotly_chart(fig, width="stretch")
