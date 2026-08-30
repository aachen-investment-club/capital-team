"""
Cross-sectional factor model (Barra USE4 / Axioma style).

    descriptors -> styles -> exposure matrix -> constrained WLS
                                                     |
                    factor covariance  <-------------+-------------> specific risk
                                                     |
                                                portfolio risk

`run_factor_model(spec)` is the entry point; everything else is a stage of that
pipeline and is importable on its own for testing. The maths is documented for
users in the dashboard's Factor Screen > Methodology tab.

Submodules are imported lazily by `model.py` rather than re-exported here, so a
page that only needs `ModelSpec` or the risk helpers does not pull scipy and the
whole estimation stack into the request path.
"""
from capital.analytics.factors.spec import (  # noqa: F401
    ALL_STYLES,
    DESCRIPTORS,
    STYLES,
    Descriptor,
    ModelSpec,
    StyleFactor,
)
