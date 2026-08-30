"""
Model specification: what the factor model is made of, and how a run is configured.

The model follows the Barra USE4 / Axioma convention of three layers:

    descriptor  →  style factor  →  exposure matrix

A *descriptor* is one raw measurement (log market cap, 12-1 momentum, book-to-
price...). A *style factor* is a weighted average of standardised descriptors
(Value = 0.5·B/P + 0.3·E/P + 0.2·CF/P). The exposure matrix additionally carries
the market intercept, industry dummies and (optionally) country dummies.

Everything is data-driven: descriptors whose inputs are missing from the store
are dropped, their style factor's remaining descriptor weights are renormalised,
and a style with no usable descriptor is dropped from the model entirely. That
is what lets the same spec run against today's thin fundamentals (P/B + market
cap only) and against the extended set once `capital-ingest fund` has back-
filled it, without editing code.
"""
from dataclasses import asdict, dataclass, field, replace

# ── Descriptors ───────────────────────────────────────────────────────────────
# source="price" descriptors are derived from EOD prices/volume alone.
# source="fund"  descriptors need a column in the `fundamentals` store table.


@dataclass(frozen=True)
class Descriptor:
    key: str
    label: str
    source: str                     # "price" | "fund" | "derived"
    column: str = ""                # fundamentals column, when source == "fund"
    transform: str = "identity"     # "identity" | "reciprocal" | "log"
    invert: bool = False            # flip the sign so higher = more of the style
    detail: str = ""


DESCRIPTORS: dict[str, Descriptor] = {d.key: d for d in [
    # -- price / volume -------------------------------------------------------
    Descriptor("lncap", "Log market cap", "fund", "market_cap", "log",
               detail="Natural log of full market capitalisation."),
    Descriptor("midcap", "Cube of size", "derived",
               detail="Standardised size cubed, then orthogonalised to Size — "
                      "captures the non-linear mid-cap effect."),
    Descriptor("beta", "Historical beta", "price",
               detail="EWMA slope of the security on the cap-weighted estimation "
                      "universe, Vasicek-shrunk toward 1."),
    Descriptor("rstr", "Relative strength", "price",
               detail="Exponentially weighted mean daily log return over the past "
                      "252 days, skipping the most recent 21."),
    Descriptor("mom6", "6-month momentum", "price",
               detail="Log return over 126 days, skipping the most recent 21."),
    Descriptor("strev", "Short-term reversal", "price", invert=True,
               detail="Negated log return over the past 21 days."),
    Descriptor("dastd", "Daily return volatility", "price",
               detail="EWMA standard deviation of daily residual returns, annualised."),
    Descriptor("cmra", "Cumulative range", "price",
               detail="Range of cumulative 12-month log returns measured monthly."),
    Descriptor("hsigma", "Residual volatility", "price",
               detail="Std deviation of residuals from the beta regression."),
    Descriptor("stom", "1-month turnover", "price",
               detail="Log of one month's traded value over market cap."),
    Descriptor("stoq", "3-month turnover", "price",
               detail="Log of average monthly traded value over a quarter."),
    Descriptor("stoa", "12-month turnover", "price",
               detail="Log of average monthly traded value over a year."),
    # -- fundamentals ---------------------------------------------------------
    Descriptor("btop", "Book to price", "fund", "pb_ratio", "reciprocal",
               detail="Inverse price-to-book."),
    Descriptor("etop", "Earnings yield", "fund", "pe_ratio", "reciprocal",
               detail="Inverse trailing price-to-earnings."),
    Descriptor("stop", "Sales to price", "fund", "ps_ratio", "reciprocal",
               detail="Inverse price-to-sales."),
    Descriptor("cetop", "Cash-flow yield", "fund", "pcf_ratio", "reciprocal",
               detail="Inverse price-to-cash-flow."),
    Descriptor("dyld", "Dividend yield", "fund", "dividend_yield",
               detail="Trailing 12-month dividend yield."),
    Descriptor("roe", "Return on equity (consensus)", "fund", "roe",
               detail="Analyst consensus mean return on equity — a forward-looking "
                      "estimate, not realised ROE, which is what our LSEG "
                      "entitlement provides. It carries analyst-coverage bias: "
                      "widely-covered large caps are measured better than the tail."),
    Descriptor("roa", "Return on assets", "fund", "roa",
               detail="Net income over total assets."),
    Descriptor("gpm", "Gross margin", "fund", "gross_margin",
               detail="Gross profit over revenue."),
    Descriptor("dtoa", "Debt to enterprise value", "fund", "debt_to_ev",
               detail="Total debt over enterprise value. Debt-to-assets is not "
                      "available on our LSEG entitlement; EV gearing is the "
                      "closest entitled measure and moves with the same driver."),
    Descriptor("sgro", "Sales growth", "fund", "revenue_ttm", "yoy_growth",
               detail="Year-on-year change in trailing revenue, derived from the "
                      "stored level series rather than a vendor growth field."),
    Descriptor("egro", "Earnings growth", "fund", "eps_ttm", "yoy_growth",
               detail="Year-on-year change in trailing earnings per share, "
                      "derived from the stored level series."),
]}


# ── Style factors ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StyleFactor:
    key: str
    label: str
    weights: dict[str, float]          # descriptor key -> weight (renormalised)
    orthogonalise_to: tuple[str, ...] = ()
    summary: str = ""
    interpretation: str = ""


STYLES: dict[str, StyleFactor] = {s.key: s for s in [
    StyleFactor(
        "size", "Size", {"lncap": 1.0},
        summary="How large the company is, measured as the log of market cap.",
        interpretation="Positive exposure = large cap. The size premium has "
                       "historically been negative (small beats large) but is "
                       "unstable and regime-dependent.",
    ),
    StyleFactor(
        "midcap", "Mid Cap", {"midcap": 1.0}, orthogonalise_to=("size",),
        summary="Non-linear size: separates mid caps from both extremes.",
        interpretation="Positive exposure = mid cap relative to the size trend. "
                       "Only meaningful once Size has been removed from it.",
    ),
    StyleFactor(
        "beta", "Beta", {"beta": 1.0},
        summary="Sensitivity of the security to the estimation universe's return.",
        interpretation="Positive exposure = amplifies market moves. Dominates "
                       "portfolio risk whenever the market factor is volatile.",
    ),
    StyleFactor(
        "momentum", "Momentum", {"rstr": 0.7, "mom6": 0.3},
        summary="Trailing 12-month return excluding the most recent month.",
        interpretation="Positive exposure = recent winner. Skipping the last "
                       "month avoids contaminating momentum with reversal.",
    ),
    StyleFactor(
        "reversal", "Short-Term Reversal", {"strev": 1.0},
        summary="Negated one-month return.",
        interpretation="Positive exposure = recent loser. Captures the "
                       "microstructure bounce that momentum deliberately skips.",
    ),
    StyleFactor(
        "resvol", "Residual Volatility",
        {"dastd": 0.74, "cmra": 0.16, "hsigma": 0.10},
        orthogonalise_to=("beta", "size"),
        summary="Volatility that beta and size do not already explain.",
        interpretation="Positive exposure = jumpy relative to peers of the same "
                       "size and beta. Historically earns a negative premium "
                       "(the low-volatility anomaly).",
    ),
    StyleFactor(
        "value", "Value", {"btop": 1.0},
        summary="Book value per unit of price.",
        interpretation="Positive exposure = cheap on assets. Weak on its own for "
                       "asset-light businesses — read it with Earnings Yield.",
    ),
    StyleFactor(
        "earnyield", "Earnings Yield", {"etop": 0.5, "cetop": 0.3, "stop": 0.2},
        summary="Earnings, cash flow and sales per unit of price.",
        interpretation="Positive exposure = cheap on flows. Usually the more "
                       "robust half of the value complex.",
    ),
    StyleFactor(
        "growth", "Growth", {"sgro": 0.5, "egro": 0.5},
        summary="Realised growth in revenue and earnings.",
        interpretation="Positive exposure = fast-growing. Strongly negatively "
                       "correlated with Value — check the factor correlation "
                       "matrix before reading the two independently.",
    ),
    StyleFactor(
        "quality", "Profitability", {"roe": 0.4, "roa": 0.3, "gpm": 0.3},
        summary="How profitably the company converts capital into earnings.",
        interpretation="Positive exposure = high-quality compounder.",
    ),
    StyleFactor(
        "leverage", "Leverage", {"dtoa": 1.0},
        summary="Balance-sheet gearing, measured against enterprise value.",
        interpretation="Positive exposure = more indebted relative to what the "
                       "business is worth. A credit-stress proxy — and note it "
                       "moves when the equity re-rates, not only when debt changes.",
    ),
    StyleFactor(
        "divyield", "Dividend Yield", {"dyld": 1.0},
        summary="Trailing dividend yield.",
        interpretation="Positive exposure = income stock; overlaps Value and is "
                       "sensitive to rates.",
    ),
    StyleFactor(
        "liquidity", "Liquidity", {"stom": 0.35, "stoq": 0.35, "stoa": 0.30},
        summary="Share turnover — traded value relative to market cap.",
        interpretation="Positive exposure = heavily traded. Negative exposure "
                       "carries an illiquidity premium and real exit risk for a "
                       "portfolio our size.",
    ),
]}

#: Every style the model knows about. There is deliberately no "core" subset:
#: a style whose descriptors are missing from the store is dropped *and reported*
#: at run time (see exposures.build_styles), so asking for everything and letting
#: the data decide is both simpler and more honest than maintaining a hand-picked
#: list that silently goes stale as the ingest gains columns.
ALL_STYLES = tuple(STYLES)

FACTOR_GROUPS = ("Market", "Style", "Industry", "Country")


# ── Run specification ─────────────────────────────────────────────────────────


@dataclass
class ModelSpec:
    """One factor-model run. Plain data — serialised into the job's params."""

    name: str = "Factor model"
    start: str = ""                      # ISO date; "" = as_of minus 3 years
    as_of: str = ""                      # ISO date; "" = latest date in the store
    frequency: str = "W-FRI"             # "B" daily | "W-FRI" weekly | "ME" monthly

    # Universe selection. The defaults take the whole security master: every
    # style, every common stock with enough history, ETFs priced off the model.
    # Growing config/security_master.csv is therefore all it takes to grow the
    # model — no spec or code change.
    styles: tuple[str, ...] = ALL_STYLES
    asset_types: tuple[str, ...] = ("COMMON",)      # estimation universe
    include_etfs: bool = True            # priced off the model by time-series regression
    min_market_cap: float = 0.0          # in units of the fundamentals column
    min_history_days: int = 250          # statistical floor, not a universe cap
    max_securities: int = 0              # 0 = no cap; the whole master is used

    # Estimation
    industry_factors: bool = True
    country_factors: bool = True
    regression_weight: str = "sqrt_cap"  # "sqrt_cap" | "cap" | "equal"
    robust: bool = True                  # Huber IRLS against cross-sectional outliers
    winsor_sigma: float = 3.0
    min_coverage: float = 0.5            # per-descriptor, per-date
    numeraire: str = "local"             # "local" | "EUR" | "USD"

    # Half-lives (trading days)
    beta_halflife: int = 63
    resvol_halflife: int = 42
    momentum_halflife: int = 126
    cov_var_halflife: int = 90
    cov_corr_halflife: int = 252
    specific_halflife: int = 90
    newey_west_lags: int = 5

    # Diagnostics
    ic_horizons: tuple[int, ...] = (1, 5, 21, 63, 126, 252)
    n_quantiles: int = 5

    # ---- (de)serialisation ---------------------------------------------------

    _TUPLES = ("styles", "asset_types", "ic_horizons")

    def to_dict(self) -> dict:
        out = asdict(self)
        for key in self._TUPLES:
            out[key] = list(out[key])
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSpec":
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        clean = {k: v for k, v in data.items() if k in known}
        for key in cls._TUPLES:
            if key in clean and clean[key] is not None:
                clean[key] = tuple(clean[key])
        return cls(**clean)

    def replace(self, **kwargs) -> "ModelSpec":
        return replace(self, **kwargs)

    # ---- derived -------------------------------------------------------------

    @property
    def active_styles(self) -> tuple[StyleFactor, ...]:
        return tuple(STYLES[k] for k in self.styles if k in STYLES)

    def required_descriptors(self) -> set[str]:
        keys: set[str] = set()
        for style in self.active_styles:
            keys |= set(style.weights)
        if "midcap" in keys:
            keys.add("lncap")            # midcap is derived from it
        keys.add("lncap")                # always needed: regression weights
        return keys

    def required_fund_columns(self) -> list[str]:
        cols = {DESCRIPTORS[k].column for k in self.required_descriptors()
                if DESCRIPTORS[k].source == "fund" and DESCRIPTORS[k].column}
        return sorted(cols)
