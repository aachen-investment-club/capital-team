"""
Market regime models.

- `champion` : the production model (Kalman trend slope x GJR-GARCH vol
  nowcast). Drives the Control Panel.
- `hmm`      : the 3-state Gaussian HMM with filtered probabilities and the
  regime-conditional return table.
- `compare`  : the model zoo (champion, HMM variants, SMA baseline), point-in-
  time fitting and scorecards for the Market Regime page.

The HMM names are re-exported so `from capital.analytics import regime as rg`
keeps working.
"""
from capital.analytics.regime.hmm import (  # noqa: F401
    STATE_COLORS,
    STATES,
    fit_regime_model,
    regime_conditional_returns,
    regime_features,
)
