"""
Tests for elasticity methods: eyex, dyex, eydx.

Covers parity with statsmodels get_margeff, hand-derived checks for
Poisson canonical-link elasticities, and error guards for discrete
variables and unknown method names.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson
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
# Elasticity × scale composition
# ---------------------------------------------------------------------------

def test_logit_eyex_scale_response_explicit(logit_fit):
    """Explicit scale='response' should match default."""
    ML = Margins(logit_fit)
    default = ML.dydx("x1", method="eyex")
    explicit = ML.dydx("x1", method="eyex", scale="response")
    assert_allclose(default.estimate, explicit.estimate, atol=1e-10)
    assert_allclose(default.se, explicit.se, rtol=1e-6)


def test_logit_eyex_scale_linear_hand_derived():
    """eyex on linear scale: hand-derived check.

    For logit with scale='linear':
        y = η = Xβ
        eyex = (∂y/∂x) * (x/y) = β * x / η
        AME eyex = mean(β * x / η) = β * mean(x/η)
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })
    df["y"] = (0.5 + 1.0 * df["x1"] - 0.5 * df["x2"] + 0.5 * rng.standard_normal(n)) > 0
    df["y"] = df["y"].astype(int)
    fit = smf.logit("y ~ x1 + x2", df).fit(disp=0)
    M = Margins(fit)
    res = M.dydx("x1", method="eyex", scale="linear")

    beta1 = fit.params["x1"]
    eta = fit.model.exog @ fit.params.values
    expected = beta1 * np.mean(df["x1"] / eta)
    assert_allclose(res.estimate[0], expected, atol=1e-6)


def test_poisson_eyex_scale_response_canonical():
    """eyex on Poisson with scale='response' equals β_k * x̄_k exactly."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    df["y"] = rng.poisson(np.exp(1.0 + 0.5 * df["x1"] - 0.3 * df["x2"]))
    fit = smf.glm("y ~ x1 + x2", df, family=Poisson()).fit()
    M = Margins(fit)
    res = M.dydx("x1", method="eyex", scale="response")

    expected = fit.params["x1"] * df["x1"].mean()
    assert_allclose(res.estimate[0], expected, atol=1e-10)


def test_eyex_quadratic_transform_hand_check():
    """Elasticity × custom quadratic transform on small OLS fixture.

    For g(η)=η²:
        ∂g/∂x = 2ηβ
        eyex = (∂g/∂x) * (x/g) = 2ηβ * x / η² = 2βx / η
        AME eyex = mean(2βx / η)
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    df["y"] = 1.0 + 2.0 * df["x"] + 0.05 * rng.standard_normal(3)
    fit = smf.ols("y ~ x", df).fit()
    M = Margins(fit)

    from smmargins.transforms import Transform
    t = Transform(
        value=lambda eta: eta ** 2,
        grad=lambda eta: 2 * eta,
        hess=lambda eta: np.full_like(eta, 2.0),
        name="quad",
    )
    res = M.dydx("x", method="eyex", scale=t)

    beta = fit.params["x"]
    eta = fit.model.exog @ fit.params.values
    expected = np.mean(2 * beta * df["x"] / eta)
    assert_allclose(res.estimate[0], expected, atol=1e-8)


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
