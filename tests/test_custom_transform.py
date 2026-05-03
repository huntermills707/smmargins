"""Tests for custom ``Transform`` objects and strict analytic contract."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import MNLogit

from smmargins import Margins
from smmargins.transforms import Transform, _BUILT_IN_SCALES


# ---------------------------------------------------------------------------
# Byte-for-byte parity with built-ins
# ---------------------------------------------------------------------------

def test_custom_exp_matches_builtin_exp():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(100), "x2": rng.standard_normal(100)})
    df["y"] = 1.0 + 2.0 * df["x1"] - df["x2"] + 0.1 * rng.standard_normal(100)
    fit = smf.ols("y ~ x1 + x2", df).fit()
    M = Margins(fit)

    custom = Transform(np.exp, np.exp, np.exp, name="custom_exp")
    res_custom = M.dydx("x1", scale=custom)
    res_builtin = M.dydx("x1", scale="exp")

    assert np.allclose(res_custom.estimate, res_builtin.estimate)
    assert np.allclose(res_custom.se, res_builtin.se)


# ---------------------------------------------------------------------------
# Missing hess strict contract
# ---------------------------------------------------------------------------

def test_missing_hess_raises_on_dydx():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(50), "x2": rng.standard_normal(50)})
    df["y"] = 1.0 + 2.0 * df["x1"] - df["x2"] + 0.1 * rng.standard_normal(50)
    fit = smf.ols("y ~ x1 + x2", df).fit()
    M = Margins(fit)

    t = Transform(np.exp, np.exp, hess=None)
    with pytest.raises(ValueError, match="hess"):
        M.dydx("x1", scale=t)


def test_missing_hess_allowed_on_predict():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(50), "x2": rng.standard_normal(50)})
    df["y"] = 1.0 + 2.0 * df["x1"] - df["x2"] + 0.1 * rng.standard_normal(50)
    fit = smf.ols("y ~ x1 + x2", df).fit()
    M = Margins(fit)

    t = Transform(np.exp, np.exp, hess=None)
    res = M.predict(scale=t)
    assert not np.isnan(res.estimate[0])


# ---------------------------------------------------------------------------
# Quadratic transform hand-derived check
# ---------------------------------------------------------------------------

def test_quadratic_transform_hand_derived():
    r"""
    For a 3-row OLS model with one regressor:
        η_i = β_0 + β_1 x_i
        g(η) = η^2
        AME = (1/3) Σ_i ∂g/∂x_i = (1/3) Σ_i 2 η_i β_1

    Hand derivation:
        AME = (2 β_1 / 3) Σ_i (β_0 + β_1 x_i)
            = 2 β_1 (β_0 + β_1 x̄)

    Jacobian of AME w.r.t. β:
        ∂AME/∂β_0 = (2 β_1 / 3) Σ_i 1 = 2 β_1
        ∂AME/∂β_1 = (2/3) Σ_i (β_0 + β_1 x_i) + (2 β_1 / 3) Σ_i x_i
                   = 2 (β_0 + β_1 x̄) + 2 β_1 x̄
                   = 2 β_0 + 4 β_1 x̄

    SE = sqrt(J @ V @ J.T)
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.standard_normal(3)})
    df["y"] = 1.0 + 2.0 * df["x"] + 0.05 * rng.standard_normal(3)
    fit = smf.ols("y ~ x", df).fit()
    M = Margins(fit)

    t = Transform(
        value=lambda eta: eta ** 2,
        grad=lambda eta: 2 * eta,
        hess=lambda eta: np.full_like(eta, 2.0),
        name="quad",
    )
    res = M.dydx("x", scale=t)

    beta0, beta1 = fit.params["Intercept"], fit.params["x"]
    xbar = df["x"].mean()
    expected_ame = 2 * beta1 * (beta0 + beta1 * xbar)
    assert pytest.approx(expected_ame, rel=1e-10) == res.estimate[0]

    J = np.array([2 * beta1, 2 * beta0 + 4 * beta1 * xbar])
    V = fit.cov_params().values
    expected_se = np.sqrt(J @ V @ J.T)
    assert pytest.approx(expected_se, rel=1e-8) == res.se[0]


# ---------------------------------------------------------------------------
# Shape-mismatch error
# ---------------------------------------------------------------------------

def test_grad_shape_mismatch_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(20)})
    df["y"] = 1.0 + 2.0 * df["x"] + 0.1 * rng.standard_normal(20)
    fit = smf.ols("y ~ x", df).fit()
    M = Margins(fit)

    t = Transform(
        value=lambda eta: eta,
        grad=lambda eta: np.column_stack([eta, eta]),  # wrong shape
        hess=lambda eta: np.full_like(eta, 0.0),
    )
    with pytest.raises(ValueError):
        M.dydx("x", scale=t)


# ---------------------------------------------------------------------------
# Multi-output passthrough (MNLogit)
# ---------------------------------------------------------------------------

def test_custom_transform_mnlogit():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })
    # 3-class outcome
    z0 = 0.5 + 1.0 * df["x1"] - 0.5 * df["x2"] + 0.3 * rng.standard_normal(n)
    z1 = -0.3 + 0.5 * df["x1"] + 0.5 * df["x2"] + 0.3 * rng.standard_normal(n)
    df["y"] = np.argmax(np.column_stack([np.zeros(n), z0, z1]), axis=1)

    fit = MNLogit.from_formula("y ~ x1 + x2", df).fit(disp=0)
    M = Margins(fit)

    t = Transform(
        value=lambda eta: eta ** 2,
        grad=lambda eta: 2 * eta,
        hess=lambda eta: np.full_like(eta, 2.0),
        name="quad",
    )
    # Should not raise
    res = M.dydx("x1", scale=t)
    assert res.estimate.shape == (3,)
