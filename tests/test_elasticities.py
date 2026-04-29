"""
Tests for elasticity methods: eyex, dyex, eydx.

Covers parity with statsmodels get_margeff, hand-derived checks for
Poisson canonical-link elasticities, and error guards for discrete
variables and unknown method names.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


ELASTICITY_METHODS = ["eyex", "dyex", "eydx"]
AT_VALUES = ["overall", "mean"]


# ---------------------------------------------------------------------------
# Logit elasticities vs statsmodels get_margeff
# ---------------------------------------------------------------------------

@pytest.mark.parity
@pytest.mark.parametrize("method", ELASTICITY_METHODS)
@pytest.mark.parametrize("at", AT_VALUES)
def test_logit_elasticity_vs_get_margeff(logit_fit, method, at):
    """Logit elasticity methods agree with statsmodels get_margeff."""
    ML = Margins(logit_fit)
    ours = ML.dydx("x1", method=method, at=at)
    sm = logit_fit.get_margeff(at=at, method=method)
    exog_names = logit_fit.model.exog_names
    x1_pos = [n for n in exog_names if n != "Intercept"].index("x1")
    assert_allclose(ours.estimate[0], sm.margeff[x1_pos], atol=1e-5)
    assert_allclose(ours.se[0], sm.margeff_se[x1_pos], rtol=1e-3)


# ---------------------------------------------------------------------------
# Poisson elasticities vs statsmodels get_margeff
# ---------------------------------------------------------------------------

@pytest.mark.parity
@pytest.mark.parametrize("method", ELASTICITY_METHODS)
def test_poisson_elasticity_vs_get_margeff(poisson_fit, method):
    """Poisson elasticity methods agree with statsmodels get_margeff."""
    MP = Margins(poisson_fit)
    ours = MP.dydx("x1", method=method)
    sm = poisson_fit.get_margeff(at="overall", method=method)
    exog_names = poisson_fit.model.exog_names
    x1_pos = [n for n in exog_names if n != "Intercept"].index("x1")
    assert_allclose(ours.estimate[0], sm.margeff[x1_pos], atol=1e-5)
    assert_allclose(ours.se[0], sm.margeff_se[x1_pos], rtol=1e-3)


# ---------------------------------------------------------------------------
# Hand-derived checks
# ---------------------------------------------------------------------------

def test_poisson_mem_eyex_hand_check(poisson_fit, sim_frame):
    """Poisson MEM eyex = beta_1 * mean(x1) for canonical log link."""
    MP = Margins(poisson_fit)
    mem_eyex = MP.dydx("x1", at="mean", method="eyex")
    hand = poisson_fit.params["x1"] * sim_frame["x1"].mean()
    assert_allclose(mem_eyex.estimate[0], hand, atol=1e-6)


# ---------------------------------------------------------------------------
# Error guards
# ---------------------------------------------------------------------------

def test_discrete_elasticity_raises(logit_fit):
    """Elasticity on a discrete variable must raise ValueError."""
    with pytest.raises(ValueError, match="continuous variable"):
        Margins(logit_fit).dydx("grp", method="eyex")


def test_unknown_method_raises(logit_fit):
    """An unknown method name must raise ValueError listing valid options."""
    with pytest.raises(ValueError, match="method must be one of"):
        Margins(logit_fit).dydx("x1", method="bogus")
