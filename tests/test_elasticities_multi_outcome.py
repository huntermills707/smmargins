"""
Elasticity methods (eyex, dyex, eydx) on multi-outcome models.

Uses hand-derived references and internal consistency checks, since
statsmodels get_margeff is broken (MNLogit) or missing (OrderedModel)
for multi-outcome elasticities.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from smmargins import Margins


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mnlogit_elasticity_fit():
    """3-class MNLogit for elasticity testing."""
    rng = np.random.default_rng(99)
    n = 600
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
    })
    eta = np.column_stack([
        np.zeros(n),
        0.4 + 0.6 * df["x1"] - 0.3 * df["x2"],
        -0.2 + 0.5 * df["x1"] + 0.2 * df["x2"],
    ])
    probs = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice([0, 1, 2], p=p) for p in probs])
    return sm.MNLogit(df["y"], sm.add_constant(df[["x1", "x2"]])).fit(disp=False)


# ── shape smoke ─────────────────────────────────────────────────────────────

def test_mnlogit_eyex_shape(mnlogit_elasticity_fit):
    M = Margins(mnlogit_elasticity_fit)
    res = M.dydx("x1", method="eyex")
    assert res.estimate.shape == (M.n_outcomes,)


# ── internal consistency: dyex = dydx * x_mean ──────────────────────────────

def test_mnlogit_dyex_vs_dydx_times_x(mnlogit_elasticity_fit):
    """dyex at mean should equal dydx * mean(x1) per outcome."""
    M = Margins(mnlogit_elasticity_fit)
    dydx = M.dydx("x1", at="mean")
    dyex = M.dydx("x1", method="dyex", at="mean")
    x_mean = M.data["x1"].mean()
    assert_allclose(dyex.estimate, dydx.estimate * x_mean, atol=1e-3)


# ── internal consistency: eydx = dydx / y_mean ──────────────────────────────

def test_mnlogit_eydx_vs_dydx_over_y(mnlogit_elasticity_fit):
    """eydx at mean should equal dydx / mean(y) per outcome."""
    M = Margins(mnlogit_elasticity_fit)
    dydx = M.dydx("x1", at="mean")
    eydx = M.dydx("x1", method="eydx", at="mean")

    # mean predicted probability per outcome at the mean profile
    y_mean = M.predict(at="mean").estimate
    assert_allclose(eydx.estimate, dydx.estimate / y_mean, atol=1e-3)


# ── internal consistency: eyex = dyex / y_mean = dydx * x / y ───────────────

def test_mnlogit_eyex_vs_composition(mnlogit_elasticity_fit):
    """eyex at mean should equal dyex / mean(y) = dydx * x / y."""
    M = Margins(mnlogit_elasticity_fit)
    dydx = M.dydx("x1", at="mean")
    dyex = M.dydx("x1", method="dyex", at="mean")
    eyex = M.dydx("x1", method="eyex", at="mean")

    y_mean = M.predict(at="mean").estimate
    assert_allclose(eyex.estimate, dyex.estimate / y_mean, atol=1e-3)
    assert_allclose(eyex.estimate, dydx.estimate * M.data["x1"].mean() / y_mean, atol=1e-3)


# ── single-outcome regression: logit still matches get_margeff ──────────────

def test_logit_eyex_regression(logit_fit):
    """Guard removal must not break single-outcome elasticity parity."""
    M = Margins(logit_fit)
    ours = M.dydx("x1", method="eyex")
    sm = logit_fit.get_margeff(at="overall", method="eyex")
    exog_names = logit_fit.model.exog_names
    x1_pos = [n for n in exog_names if n != "Intercept"].index("x1")
    assert_allclose(ours.estimate[0], sm.margeff[x1_pos], atol=1e-5)
    assert_allclose(ours.se[0], sm.margeff_se[x1_pos], rtol=1e-3)


# ── blow-up guard ───────────────────────────────────────────────────────────

def test_mnlogit_eyex_warns_on_near_zero_class():
    """When one class has tiny probability, eyex emits RuntimeWarning."""
    rng = np.random.default_rng(77)
    n = 300
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
    })
    # Strongly imbalanced: class 1 is very rare
    eta = np.column_stack([
        np.zeros(n),
        -4.0 + 0.1 * df["x1"],
        1.0 + 0.3 * df["x1"],
    ])
    probs = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice([0, 1, 2], p=p) for p in probs])
    fit = sm.MNLogit(df["y"], sm.add_constant(df[["x1"]])).fit(disp=False)
    # Force one intercept to an extreme value so predictions dip below 1e-12
    p = fit.params.copy()
    p.iloc[0, 0] = -30.0
    fit._results.params = p
    M = Margins(fit)
    with pytest.warns(RuntimeWarning, match="predicted probabilities below"):
        M.dydx("x1", method="eyex")


# ── OrderedModel smoke ──────────────────────────────────────────────────────

def test_ordered_eyex_finite(ordered_fit):
    """OrderedModel eyex returns shape (K,) with finite values."""
    M = Margins(ordered_fit)
    res = M.dydx("x1", method="eyex")
    assert res.estimate.shape == (M.n_outcomes,)
    assert np.isfinite(res.estimate).all()
