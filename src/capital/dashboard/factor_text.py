"""
Explanatory copy for the factor screen and the Methodology tab.

Kept out of the page modules so the same wording backs the collapsed "how this
works" panels on the screen and the long-form methodology page, a student who
expands an explanation on a chart and then opens the Methodology tab should
read the same model described the same way, not two drifting descriptions.

Markdown, rendered with MathJax: $...$ inline, $$
...
$$ display.
"""

# ── Page-level ────────────────────────────────────────────────────────────────

PAGE_INFO = (
    "A full cross-sectional factor model over the whole universe. Estimating it "
    "takes real compute, so runs are queued as background jobs: start one and "
    "keep using the dashboard. Use it to see what the portfolio is actually "
    "exposed to, and what a trade would do to that."
)

METHOD_WHAT = r"""
**The question this page answers.** Two portfolios can hold completely different
stocks and still be the same bet. If everything we own is expensive, profitable
and has gone up recently, then we do not hold twenty independent ideas; we hold
one idea twenty times, and it will unwind all at once. A factor model measures
that directly: it decomposes every security into a handful of shared
characteristics, so the portfolio's real exposures become visible.

**The model in one line.** Each period, we regress the returns of every security
in the universe on their characteristics *measured before the period started*:

$$
r_{i,t} \;=\; \sum_k X_{i,k,t-1}\, f_{k,t} \;+\; u_{i,t}
$$

- $X_{i,k,t-1}$: security $i$'s **exposure** to factor $k$, known at $t-1$.
  Standardised, so $+1$ means one cross-sectional standard deviation above the
  market on that characteristic.
- $f_{k,t}$, the **factor return**: what one unit of that exposure paid in
  period $t$. It is an output of the regression, not an input.
- $u_{i,t}$, the **specific return**: the part of the move that is about this
  company alone. It is what diversification actually removes.

Everything else on this page is a view of those three objects.

**What it is not.** The model is descriptive, not predictive. It tells you what
you are exposed to and what those exposures have paid historically. It does not
forecast factor returns, and a factor with a strong historical premium is not a
recommendation to load up on it.
"""

METHOD_JOBS = r"""
A run touches every security, on every estimation date: descriptors, robust
standardisation, one constrained regression per period, then a covariance matrix
and a full diagnostic suite. Over a universe of 1,000 to 2,000 names
that is tens of seconds to minutes, long enough that doing it inside a page callback would freeze the
dashboard for everyone.

So a run is a **job**, not a callback:

1. Submitting writes a small job file and returns immediately.
2. A separate **process** picks it up and does the work, so the computation
   cannot compete with the dashboard for CPU, and a crash in it cannot take the
   app down.
3. It reports progress as it goes; this page polls that a few times a minute.
4. The finished run is written to disk as a set of tables.

Consequences worth knowing: you can close the tab and come back, everyone sees
the same queue and the same finished runs, and a completed run is a permanent
artefact, it keeps showing exactly what it computed even after the next nightly
data load. When the data has moved on underneath it, the run is flagged
**stale**;
the numbers are still valid for the data they were computed from.
"""

# ── Exposures and the portfolio view ──────────────────────────────────────────

READING_EXPOSURES = r"""
Exposures are **z-scores against the market**, not raw quantities. The
standardisation is deliberately asymmetric:

- The mean subtracted is **cap-weighted**, so the market portfolio has exposure
  $0$ to every style by construction.
- The standard deviation divided by is **equal-weighted**, so one unit is one
  typical security's distance from the market.

$$
X_{i,k} \;=\; \frac{d_{i,k} - \sum_j w^{\text{cap}}_j d_{j,k}}{\sigma_{\text{EW}}(d_{\cdot,k})}
$$

That is what makes "the portfolio is $+0.8$ on Momentum" a meaningful sentence:
it means the book sits four-fifths of a cross-sectional standard deviation to
the momentum side **of the market**, not of some equal-weighted average nobody
holds.

**Reading the numbers.** $|X| < 0.25$ is noise. $0.25$ to $0.75$ is a real but
moderate tilt. Above $1.0$ is a deliberate bet, and it should be one you can
name. Sign conventions are always "more of the label": positive Value is *cheap*,
positive Leverage is *more indebted*, positive Reversal is a recent *loser*.

**Coverage matters.** The portfolio exposure is computed over the positions the
model actually covers, renormalised to those. If a chunk of the book is cash or
an unmodelled security, the honest statement is "the covered part has this tilt" - which is why the covered share is shown next to it.
"""

RISK_DECOMPOSITION = r"""
Exposures say what we are tilted towards. Risk says how much that tilt costs,
which is the number that should drive sizing.

$$
\sigma_P^2 \;=\; \underbrace{x' F x}_{\text{factor}} \;+\; \underbrace{\sum_i w_i^2 s_i^2}_{\text{specific}}, \qquad x = X'w
$$

- $F$: the factor covariance matrix (annualised).
- $s_i$: security $i$'s specific volatility.
- $x$: the portfolio's factor exposure vector.

Each factor's contribution to volatility is

$$
\mathrm{CTR}_k \;=\; \frac{x_k \,(F x)_k}{\sigma_P}, \qquad \sum_k \mathrm{CTR}_k = \frac{\sigma_F^2}{\sigma_P}
$$

so the bars add up exactly to the factor volatility and can be read directly as
"this factor is worth *N* volatility points to us". A large contribution needs
both a large exposure **and** a volatile factor: a big tilt to a quiet factor is
cheap, a small tilt to a violent one is not.

**The factor/specific split is the headline.** A high factor share means the
portfolio's fate is decided by a few common bets, and adding more names in the
same style will not help. A high specific share means risk is genuinely
diversified across company-level outcomes, which is where stock picking is
supposed to be paid.
"""

WHAT_IF = r"""
This is the same risk maths applied twice, once to the current book, once to
the book after your hypothetical trades, and differenced. Nothing is executed
and nothing is saved.

**Funding.** A trade has to be paid for. *Pro rata* scales every untouched
position so the book stays fully invested, which is the honest default: you
cannot add 5% of a new name without selling something. *From cash* lets the
total drift instead, which models drawing down or building a cash balance.

**What to look at, in order:**

1. **Exposure deltas**: did the trade move the tilt you intended it to move?
   A trade justified on one factor that mostly moves a different one is usually
   a trade about something you have not articulated.
2. **Risk contribution deltas**: a tilt that grows towards a *volatile* factor
   costs far more than the exposure change alone suggests.
3. **Total volatility and turnover**: the price of the change, in risk and in
   trading.

**The trap this exists to catch.** Adding a "diversifier" that happens to load on
the same factors as everything you already own reduces single-name risk while
leaving systematic risk untouched, or raising it. The exposure delta shows that
immediately; a correlation matrix of returns often does not.
"""

# ── Diagnostics ───────────────────────────────────────────────────────────────

IC_EXPLAIN = r"""
The **information coefficient** is the cross-sectional rank correlation between
an exposure today and the return that follows:

$$
\mathrm{IC}_{k,t} \;=\; \rho_{\text{Spearman}}\big(X_{\cdot,k,t},\; r_{\cdot,\,t \to t+h}\big)
$$

Rank correlation, not Pearson, so one blown-up small cap cannot manufacture a
signal. Typical real values are small: a monthly IC of $0.03$ to $0.05$ is a
genuinely useful factor, and anything above $0.15$ in a public-equity universe
should make you look for a bug or a look-ahead.

**Consistency beats magnitude.** The statistic that matters is the IC information
ratio,

$$
\mathrm{IR} \;=\; \frac{\overline{\mathrm{IC}}}{\sigma(\mathrm{IC})}, \qquad t \;=\; \mathrm{IR}\,\sqrt{T}
$$

A factor with IC $0.03$ every month is investable; one that averages $0.08$ by
being right twice and catastrophic once is not. Treat $|t| < 2$ as "not
distinguishable from zero over this sample", and remember $T$ here counts
overlapping periods at longer horizons, so long-horizon t-statistics are
optimistic.
"""

DECAY_EXPLAIN = r"""
The same IC measured at 1, 5, 21, 63, 126 and 252 trading days. The *shape* is
the diagnostic:

- **Rises then flattens**: a real, slow-moving characteristic. This is what an
  investable factor looks like; you can hold it and trade it rarely.
- **Spikes at one day and collapses**: microstructure. It is real, and it will
  be entirely eaten by spread and commission at our size.
- **Flat near zero everywhere**: the factor does not order returns in this
  universe over this sample. It may still be worth *measuring* for risk, which
  does not require the factor to be paid.

Decay also tells you the rebalancing frequency the factor can justify. A signal
whose IC peaks at six months does not need weekly trading.
"""

QUANTILE_EXPLAIN = r"""
Sort the universe into buckets by exposure each period, hold each bucket
cap-weighted, and record what they returned. The top-minus-bottom row is the
long/short factor portfolio.

**Monotonicity is the test.** A trustworthy factor pays progressively across the
buckets. A factor that only works in the extreme bucket is usually one crowded
trade, a sector in disguise, or a handful of names too small to hold: and it
will not survive the first time that bucket's composition changes.

Buckets are cap-weighted deliberately. An equal-weighted top bucket is dominated
by the smallest names in it, which flatters almost every factor and describes a
portfolio we could not actually own.
"""

PERSISTENCE_EXPLAIN = r"""
How much a security's exposure moves from one period to the next, measured as the
period-over-period cross-sectional rank correlation.

- **High (> 0.95)**: a stable characteristic. Cheap to hold: the portfolio
  keeps the tilt without trading. Size and Liquidity live here.
- **Low (< 0.7)**: the exposure is mostly reshuffling. Any strategy built on it
  pays turnover continuously, and the implied half-life tells you how fast the
  tilt decays if you stop trading.

Short-Term Reversal is *supposed* to sit at the bottom; that is what it is. The
warning sign is a factor you expected to be slow showing up fast, which usually
means its descriptor is dominated by a noisy input.
"""

VIF_EXPLAIN = r"""
The variance inflation factor asks how well each style is explained by all the
others:

$$
\mathrm{VIF}_k \;=\; \frac{1}{1 - R_k^2}
$$

Above roughly $5$, the factor's return is not separately identified: whatever the
regression assigns to it could as easily belong to the factors it overlaps with,
and both coefficients will be unstable period to period.

This is why the model orthogonalises Mid Cap against Size and Residual Volatility
against Beta and Size before estimating anything. Without it, Residual Volatility
is mostly Beta wearing a different hat, and neither factor return means what its
label claims.
"""

CORRELATION_EXPLAIN = r"""
Correlations of the **factor returns** - not of the exposures. Two factors can be
close to orthogonal in exposure and still move together, because the same
macro-economic news drives both.

Value/Growth strongly negative and Beta/Residual Volatility strongly positive are
normal and expected. What matters for the portfolio is that a positive tilt to
two positively correlated factors is one bet, not two, and the risk decomposition
already accounts for that through the off-diagonal terms of $F$ - which is why
contributions can look larger than exposures alone would suggest.
"""

SUBPERIOD_EXPLAIN = r"""
The same factor returns, compounded per calendar year. Almost every factor
"works" over a full sample; few work in most of its individual years.

Read the row, not the average. A factor that delivered its entire premium in one
year is a description of that year, not a property of the market: and the sample
here is short enough that this is the most common failure mode. Look for factors
whose sign is stable across years even when the magnitude is not.
"""

FIT_EXPLAIN = r"""
The $R^2$ of each period's cross-sectional regression: the share of that period's
return dispersion the factors explain.

Typical values are $0.2$ to $0.5$ for a daily cross-section and higher for weekly,
and $R^2$ spikes in crises, when everything moves together, factors explain
almost everything, and stock selection explains almost nothing.

**Watch the observations-per-factor ratio.** Fitting ~30 factors to a
cross-section of 100 securities produces a high $R^2$ that means very little; the
same 30 factors against 1,500 securities produce an honest one. Below about 10
observations per factor, treat industry and country returns as indicative only.
"""

# ── Security explorer ─────────────────────────────────────────────────────────

SECURITY_EXPLAIN = r"""
A single security's exposures over time, the per-name robustness view.

**A factor label is only meaningful if it is stable.** A stock whose Momentum
exposure swings between $+1.5$ and $-1.5$ within a year is not "a momentum name";
it is a name that happened to screen that way the day you looked. The stability
table gives the mean, range and autocorrelation of each exposure so you can tell
a persistent characteristic from a transient one.

**$R^2$ and specific risk.** $R^2$ is how much of this security's return variance
the factors explain. Low $R^2$ does not mean the model failed; it means the
security moves for reasons the factors do not carry. That risk is genuinely
idiosyncratic and genuinely diversifiable, which is precisely where stock picking
is supposed to earn its keep. It is also, position for position, the risk that
does not net off against anything else in the book.
"""

SCREEN_EXPLAIN = r"""
Every security in the run, with its exposures on the latest cross-section. Sort,
filter, and use it to find candidates that would move a specific exposure in a
specific direction, then test the actual trade in **What-if** before believing
the screen.

**Two ways a security gets its exposures**, shown in the *method* column:

- **cross-sectional**: a single stock, measured from its own descriptors. This
  is the model proper.
- **returns-based**: an ETF or fund. Funds have no book value and no sector, so
  their exposures come from regressing their returns on the *estimated* factor
  returns. It is a legitimate estimate on the same scale, which is what lets an
  ETF and a single stock be added into one portfolio exposure, but it is
  measured with error, so check its fit $R^2$ before leaning on it.

ETFs are deliberately excluded from the estimation universe: a fund is a
portfolio of the very securities being regressed, so including it would
double-count its holdings and let a fund's own bets contaminate the factor
returns it is being measured against.
"""

COVERAGE_EXPLAIN = r"""
What the model could and could not measure on this run.

Descriptors are dropped when too few securities have the underlying data on a
given date; a style factor with no surviving descriptor is dropped entirely and
listed here. Nothing is silently substituted: if Value could only be built from
book-to-price because earnings and cash-flow yields are not available, this
report says so, and the Value factor on this run means exactly "book to price".

This is also where a thin universe shows up. A cross-sectional model wants
hundreds of securities per period; with a hundred, the style factors are still
informative but the industry and country returns are fitted on very few names
each and should be read as indicative.
"""

# ── Methodology tab ───────────────────────────────────────────────────────────

METHOD_HOWTO = r"""
This tab is the reference for everything the other tabs show: what each factor
measures, the maths behind each stage of the model, and where the model stops
being trustworthy.

Read it in one of two ways. **Top-down**: start at the factor set below to see
what each style is built from, then work through the pipeline sections for how
exposures and factor returns are computed. **Bottom-up**: when a chart on
another tab raises a question, its own collapsed explanation answers it in one
paragraph, and the corresponding section here gives the equation.

Nothing here depends on the selected run; it describes the model itself.
"""


METHOD_OVERVIEW = r"""
This model follows the structure of Barra's USE4 and its descendants: a
cross-sectional (not time-series) model, estimated fresh each period, with
descriptors aggregated into style factors, industry and country factors carried
as constrained dummies, and a risk model built from the resulting factor returns.

The pipeline, in order:

$$
\text{descriptors} \;\to\; \text{styles} \;\to\; \text{exposure matrix}
\;\to\; \text{cross-sectional regression} \;\to\;
\begin{cases} \text{factor covariance} \\ \text{specific risk} \end{cases}
\;\to\; \text{portfolio risk}
$$

The rest of this page is each arrow in that diagram.
"""

METHOD_DESCRIPTORS = r"""
A descriptor is one raw measurement. Price-based descriptors come from EOD
prices and volume; fundamental descriptors come from the fundamentals table,
sampled point-in-time on each estimation date via an as-of join so a value is
never used before it existed.

**Beta.** An exponentially weighted regression of the security on the cap-weighted
estimation universe, with half-life $h_\beta$:

$$
\hat\beta_i = \frac{\mathrm{Cov}_w(r_i,\, r_M)}{\mathrm{Var}_w(r_M)}
$$

Raw betas are noisy, so they are **Vasicek-shrunk** toward the cross-sectional
mean by the estimate's own precision:

$$
\beta_i^{*} \;=\; v_i \hat\beta_i + (1 - v_i)\,\bar\beta, \qquad
   v_i = \frac{\sigma^2_{\text{cross}}}{\sigma^2_{\text{cross}} + \mathrm{se}^2(\hat\beta_i)}
$$

A name with a short history or a large residual variance has a big
$\mathrm{se}$, gets a small $v_i$, and is pulled toward the crowd rather than
trusted at face value.

**Momentum.** An exponentially weighted mean log return over 252 days, skipping
the most recent 21:

$$
\mathrm{RSTR}_i \;=\; \sum_{s=22}^{252} \omega_s \,\ln(1 + r_{i,t-s}), \qquad
   \omega_s \propto 2^{-s/h_m}
$$

The 21-day skip is not cosmetic: the most recent month is dominated by
short-term reversal, and including it cancels part of the momentum signal.

**Volatility.** Three separate descriptors: EWMA volatility of returns (DASTD),
volatility of residuals from the beta regression (HSIGMA), and the cumulative
range of trailing monthly returns (CMRA). They disagree often enough to be worth
blending.

**Turnover.** Traded shares over shares outstanding, summed over 1, 3 and 12
months and logged. Where shares outstanding is unavailable, traded value over
market capitalisation is the same quantity as long as price and market cap are
quoted in the same currency.

**Value and quality.** Ratios are inverted, book-to-price rather than
price-to-book, so that "more of the style" is always "a bigger number", and so
the descriptor stays finite and well-ordered as the ratio approaches zero.
"""

METHOD_STANDARDISATION = r"""
Raw descriptors are on incompatible scales with fat tails. Three steps fix that,
in this order:

**Robust scaling.** Centre on the median, scale by the MAD, clip at $\pm c$
(default $c = 3$):

$$
\tilde d_i = \mathrm{clip}\left( \frac{d_i - \mathrm{med}(d)}{1.4826 \cdot \mathrm{MAD}(d)},\ \pm c \right)
$$

The MAD rather than the standard deviation, because a single extreme value
inflates the standard deviation and thereby shrinks *everyone else's* z-score; the outlier ends up setting the scale for the whole universe.

**Cap-weighted standardisation.**

$$
X_i \;=\; \frac{\tilde d_i - \sum_j w^{\text{cap}}_j \tilde d_j}{\sigma_{\text{EW}}(\tilde d)}
$$

Cap-weighted mean, equal-weighted standard deviation. The consequence is the
identity that makes the whole model readable: **the cap-weighted market
portfolio has exposure zero to every style factor**, so every exposure is a
statement about deviation from the market.

**Missing values.** Filled with the security's *industry* cap-weighted mean of
the standardised descriptor, falling back to $0$ (the market) if the industry has
no data either. A security with no book value is not assumed to be average
overall, it is assumed to be average for its industry, which is the weaker and
more defensible assumption. A descriptor whose coverage on a given date falls
below the coverage floor is dropped for that date entirely rather than filled.
"""

METHOD_STYLES = r"""
A style is a weighted blend of its standardised descriptors, renormalised over
whichever ones are actually available, then re-standardised:

$$
S_i \;=\; \frac{\sum_j \omega_j X_{i,j}}{\sum_j \omega_j}
   \;\longrightarrow\; \text{cap-standardise}
$$

Some styles are then made orthogonal to others by taking the cap-weighted
cross-sectional residual:

$$
S^{\perp} = S - B \left(B^{\top} W B\right)^{-1} B^{\top} W S
$$

Two are orthogonalised by construction:

- **Mid Cap** $\perp$ Size, it is the cube of standardised size, so without
  removing the linear part it *is* Size.
- **Residual Volatility** $\perp$ (Beta, Size) - otherwise it is mostly Beta, and
  the regression cannot attribute a return to either one.

Skipping this step is the single most common way a factor model produces
confident, unstable, meaningless coefficients.
"""

METHOD_REGRESSION = r"""
Each period, one weighted least-squares problem across the universe:

$$
r_{i,t} \;=\; f_{M,t} + \sum_{p} \mathbb{1}[i \in p]\, f_{p,t}
   + \sum_{c} \mathbb{1}[i \in c]\, f_{c,t}
   + \sum_{s} X_{i,s,t-1} f_{s,t} + u_{i,t}
$$

with a market intercept, industry dummies $p$, country dummies $c$ and styles $s$.

**Identification.** Industry dummies sum to the intercept, so $[\mathbf{1} \mid
\text{industries}]$ is rank-deficient and the market/industry split is arbitrary.
The standard fix is a constraint that industry returns are cap-weighted zero-sum
(and likewise for countries):

$$
\sum_p \frac{C_p}{C}\, f_{p,t} \;=\; 0
$$

where $C_p$ is industry $p$'s total capitalisation. The market factor then *is*
the cap-weighted universe return, and each industry return is a deviation from
it, which is what makes an industry factor return interpretable.

We impose this exactly rather than by penalty. Writing the constraints as
$A f = 0$ and letting $R$ be an orthonormal basis of $\mathrm{null}(A)$, set
$f = Rg$ and solve the unconstrained problem in $g$:

$$
\hat g \;=\; (\tilde X' W \tilde X)^{-1} \tilde X' W r, \qquad
   \tilde X = XR, \qquad \hat f = R\hat g
$$

**Weights.** $W = \mathrm{diag}(\sqrt{C_i})$ by default. Full cap weighting lets
the largest handful of names dictate every factor return; equal weighting hands
the estimate to the smallest and noisiest. The square root is the usual
compromise, and it is also the weighting under which the residual variance is
roughly homoskedastic, since specific volatility scales approximately with
$C^{-1/2}$.

**Robustness.** Two Huber IRLS passes downweight residuals beyond $\approx 1.35$
robust sigmas:

$$
w_i \;\leftarrow\; w_i \cdot \min\!\left(1, \frac{1.345}{|u_i| / \hat\sigma_{\text{MAD}}}\right)
$$

Without this a single blown-up small cap can move a factor return by a large
fraction of its value. The cut-off is scaled by the period's own MAD, so it
adapts to crisis periods instead of rejecting half the universe in one.
"""

METHOD_RISK = r"""
**Factor covariance.** Volatility mean-reverts faster than correlation, so the
two are estimated with different memories and recombined:

$$
F \;=\; D_{h_v}\, C_{h_c}\, D_{h_v} \times T
$$

$D_{h_v}$ holds volatilities estimated with the short half-life $h_v$, $C_{h_c}$
is the correlation matrix from the long half-life $h_c$, and $T$ annualises. A
single half-life either lags a volatility spike or produces a correlation matrix
too jumpy to use.

Factor returns are serially correlated (non-synchronous trading, slow
information diffusion), so a Bartlett-kernel **Newey-West** adjustment corrects
the long-horizon understatement:

$$
\Sigma \;=\; \Gamma_0 + \sum_{l=1}^{L}\left(1 - \frac{l}{L+1}\right)(\Gamma_l + \Gamma_l')
$$

with exponentially weighted autocovariances $\Gamma_l$. The result is projected
back to the nearest positive semi-definite matrix, because the two-half-life
recombination and the lag sum can each push it slightly indefinite.

**Specific risk.** EWMA volatility of each security's residuals, then Bayesian
shrinkage toward its size bucket:

$$
s_i \;=\; v_i \bar s_{B(i)} + (1 - v_i) \hat s_i, \qquad
   v_i = \frac{q\,|\hat s_i - \bar s_{B(i)}|}{\Delta_{B(i)} + q\,|\hat s_i - \bar s_{B(i)}|}
$$

A security whose estimate sits far from its peers, in a bucket whose members are
tightly clustered, is the most likely to be mismeasured, and gets shrunk hardest.

**Portfolio risk.** With $x = X'w$:

$$
\sigma_P^2 = x'Fx + \sum_i w_i^2 s_i^2, \qquad
   \mathrm{MCTR}_k = \frac{(Fx)_k}{\sigma_P}, \qquad
   \mathrm{CTR}_k = x_k \mathrm{MCTR}_k
$$

Contributions sum exactly to the factor volatility, and tracking error against a
benchmark is the same formula on active weights $w - b$.
"""

METHOD_RETURNS_BASED = r"""
A fund has no book value and no sector, and it cannot go in the cross-section: it
is a portfolio of the very securities being regressed, so including it would
double-count its holdings and let its own bets contaminate the factor returns it
is meant to be measured against.

Instead, funds are priced *off* the finished model by time-series regression
(returns-based style analysis):

$$
r_{E,t} \;=\; \sum_k \beta_{E,k}\, \hat f_{k,t} + \varepsilon_{E,t}
$$

with a small ridge penalty for stability when factor returns are collinear over a
short window. The $\beta_{E,k}$ live on the same scale as cross-sectional
exposures, which is what makes it legitimate to add an ETF and a single stock
into one portfolio exposure number.

The estimate carries real error, so the fit $R^2$ and observation count travel
with it. A fund with $R^2$ of $0.4$ has exposures that are a rough sketch.
"""

METHOD_LIMITS = r"""
Stated explicitly, because a model whose limitations are undocumented gets used
outside them.

- **Universe size.** A cross-sectional model wants hundreds of securities per
  period. With a small universe the style factors remain informative, but the
  industry and country returns are fitted on very few names each. The
  observations-per-factor figure on the run summary is the check; below ~10,
  read industry and country returns as indicative.
- **Currency.** Returns are in local currency unless exchange rates are available.
  Country factors absorb most of the resulting drift, but a portfolio spanning
  several currencies carries FX risk this model will not name separately.
- **Point-in-time fundamentals.** The as-of join prevents using a value before
  the date it was recorded, but our data provider supplies its *current* view
  of history.
  Restatements are not reproduced as they were originally reported, so long
  fundamental backtests carry some restatement bias.
- **Survivorship.** The universe is the set of securities we cover today.
  Securities that
  were delisted are absent, which flatters historical factor returns.
- **No eigenfactor or volatility-regime adjustment.** Barra's production models
  add a simulation-based eigenfactor correction (cross-sectional risk is biased
  low along the covariance matrix's smallest eigenvectors) and a volatility
  regime adjustment. Neither is implemented here, so optimised, extreme-tilt
  portfolios would have their risk understated. For measuring a real book's
  exposures, the omission is minor.
- **Descriptive, not predictive.** Nothing here forecasts factor returns.
"""

METHOD_GLOSSARY = r"""
| Symbol | Meaning |
| --- | --- |
| $r_{i,t}$ | return of security $i$ over period $t$ |
| $X_{i,k,t}$ | exposure of security $i$ to factor $k$, known at $t$ |
| $f_{k,t}$ | factor return: what one unit of exposure $k$ paid in $t$ |
| $u_{i,t}$ | specific return: the company-only part of the move |
| $w_i$ | portfolio weight of security $i$ |
| $x = X'w$ | the portfolio's factor exposure vector |
| $C_i$ | market capitalisation of security $i$ |
| $F$ | annualised factor covariance matrix |
| $s_i$ | annualised specific volatility of security $i$ |
| $\sigma_P$ | portfolio volatility |
| $h$ | half-life of an exponential weighting, in periods |
"""
