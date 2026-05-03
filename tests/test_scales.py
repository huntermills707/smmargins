"""Tests for built-in ``scale=`` values and the scale machinery."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson

from smmargins import Margins
from smmargins.transforms import Transform, _BUILT_IN_SCALES
import smmargins._derivs as _derivs


# ---------------------------------------------------------------------------
# Basic parity and sanity
# ---------------------------------------------------------------------------

def test_response_explicit_equals_default():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(200)})
    df["y"] = 1.0 + 2.0 * df["x"] + 0.1 * rng.standard_normal(200)
    fit = smf.ols("y ~ x", df).fit()
    M = Margins(fit)
    default_pred = M.predict()
    explicit_pred = M.predict(scale="response")
    assert np.allclose(default_pred.estimate, explicit_pred.estimate)
    assert np.allclose(default_pred.se, explicit_pred.se)


def test_linear_ols_ame_exact_beta():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(200), "x2": rng.standard_normal(200)})
    df["y"] = 1.0 + 2.0 * df["x1"] - df["x2"] + 0.1 * rng.standard_normal(200)
    fit = smf.ols("y ~ x1 + x2", df).fit()
    M = Margins(fit)
    res = M.dydx("x1", scale="linear")
    assert pytest.approx(fit.params["x1"], rel=1e-10) == res.estimate[0]
    expected_se = np.sqrt(fit.cov_params().loc["x1", "x1"])
    assert pytest.approx(expected_se, rel=1e-6) == res.se[0]


def test_linear_logit_ame_exact_beta():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(500), "x2": rng.standard_normal(500)})
    df["y"] = (1.0 + 2.0 * df["x1"] - df["x2"] + 0.5 * rng.standard_normal(500)) > 0
    df["y"] = df["y"].astype(int)
    fit = Logit.from_formula("y ~ x1 + x2", df).fit(disp=0)
    M = Margins(fit)
    res = M.dydx("x1", scale="linear")
    assert pytest.approx(fit.params["x1"], rel=1e-6) == res.estimate[0]


def test_pr_identical_to_response_on_logit():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(200)})
    df["y"] = (df["x"] + 0.5 * rng.standard_normal(200)) > 0
    df["y"] = df["y"].astype(int)
    fit = Logit.from_formula("y ~ x", df).fit(disp=0)
    M = Margins(fit)
    res_pr = M.predict(scale="pr")
    res_resp = M.predict(scale="response")
    assert np.allclose(res_pr.estimate, res_resp.estimate)
    assert np.allclose(res_pr.se, res_resp.se)


# ---------------------------------------------------------------------------
# Self-consistency checks
# ---------------------------------------------------------------------------

def test_exp_on_log_link_poisson_equals_response():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(200)})
    df["y"] = rng.poisson(np.exp(1.0 + 0.5 * df["x"]))
    fit = GLM.from_formula("y ~ x", df, family=Poisson()).fit()
    M = Margins(fit)
    res_exp = M.dydx("x", scale="exp")
    res_resp = M.dydx("x", scale="response")
    # On log-link Poisson, response = exp(eta), so "exp" and "response"
    # should give identical point estimates (both are exp(eta)).
    assert np.allclose(res_exp.estimate, res_resp.estimate, rtol=1e-6)


# ---------------------------------------------------------------------------
# Hand-derived checks
# ---------------------------------------------------------------------------

def test_or_on_logit_hand_derived():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(300)})
    df["y"] = (df["x"] + 0.5 * rng.standard_normal(300)) > 0
    df["y"] = df["y"].astype(int)
    fit = Logit.from_formula("y ~ x", df).fit(disp=0)
    M = Margins(fit)
    res = M.predict(scale="or")

    # odds = p/(1-p) = exp(eta)
    # Mean odds = mean(exp(eta))
    eta = fit.model.exog @ fit.params.values
    expected = np.mean(np.exp(eta))
    assert pytest.approx(expected, rel=1e-5) == res.estimate[0]


# ---------------------------------------------------------------------------
# FD vs analytic Jacobian parity (Task 1 regression test)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scale_name", ["linear", "exp"])
def test_fd_vs_analytic_jac_parity(scale_name):
    """Toggle _DYDX_FD_ONLY and assert analytic and FD Jacobians agree."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(100), "x2": rng.standard_normal(100)})
    # Use Poisson to ensure positive predictions for all scales
    df["y"] = rng.poisson(np.exp(1.0 + 0.5 * df["x1"] - 0.3 * df["x2"]))
    fit = GLM.from_formula("y ~ x1 + x2", df, family=Poisson()).fit()
    M = Margins(fit)

    _derivs._DYDX_FD_ONLY = True
    try:
        res_fd = M.dydx("x1", scale=scale_name)
    finally:
        _derivs._DYDX_FD_ONLY = False

    res_analytic = M.dydx("x1", scale=scale_name)

    assert np.allclose(res_fd.estimate, res_analytic.estimate, rtol=1e-5, atol=1e-7)
    assert np.allclose(res_fd.se, res_analytic.se, rtol=1e-5, atol=1e-7)


def test_fd_vs_analytic_jac_parity_pr_on_logit():
    """pr scale parity on a Logit model."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(100), "x2": rng.standard_normal(100)})
    df["y"] = (df["x1"] + 0.5 * rng.standard_normal(100)) > 0
    df["y"] = df["y"].astype(int)
    fit = Logit.from_formula("y ~ x1 + x2", df).fit(disp=0)
    M = Margins(fit)

    _derivs._DYDX_FD_ONLY = True
    try:
        res_fd = M.dydx("x1", scale="pr")
    finally:
        _derivs._DYDX_FD_ONLY = False

    res_analytic = M.dydx("x1", scale="pr")

    assert np.allclose(res_fd.estimate, res_analytic.estimate, rtol=1e-5, atol=1e-7)
    assert np.allclose(res_fd.se, res_analytic.se, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Invalid combos
# ---------------------------------------------------------------------------

def test_ir_on_ols_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(50)})
    df["y"] = 1.0 + 2.0 * df["x"] + 0.1 * rng.standard_normal(50)
    fit = smf.ols("y ~ x", df).fit()
    M = Margins(fit)
    with pytest.raises(ValueError, match="ir"):
        M.dydx("x", scale="ir")


def test_or_on_poisson_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(50)})
    df["y"] = rng.poisson(np.exp(1.0 + 0.5 * df["x"]))
    fit = GLM.from_formula("y ~ x", df, family=Poisson()).fit()
    M = Margins(fit)
    with pytest.raises(ValueError, match="or"):
        M.dydx("x", scale="or")


def test_log_when_predictions_nonpos_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(50)})
    df["y"] = 1.0 + 2.0 * df["x"] + 0.1 * rng.standard_normal(50)
    fit = smf.ols("y ~ x", df).fit()
    M = Margins(fit)
    # OLS predictions can be negative; log of negative should raise
    with pytest.raises(ValueError, match="log"):
        M.predict(scale="log")
