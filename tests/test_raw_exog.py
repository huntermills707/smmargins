"""
Raw exog mode tests.

These tests verify that Margins works correctly when the model was fit with
raw design matrices (sm.OLS(y, X)) rather than formulas. In raw mode:
- Only the literal column matching the variable name is perturbed
- Interactions in the design matrix are NOT automatically tracked
- Column names come from model.exog_names
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.testing import assert_allclose
import pytest
from smmargins import Margins


# ── Additive model equivalence ─────────────────────────────────────────────

def test_raw_exog_additive_model_equivalence():
    """Formula and raw exog fits give identical results for additive models.

    For a purely additive model (no interactions, no transformations),
    formula mode and raw mode should produce identical marginal effects
    and predictions because perturbing a column in the data frame has
    the same effect as perturbing the corresponding column in the design
    matrix.
    """
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "x1": rng.normal(size=N),
        "x2": rng.normal(size=N),
    })
    df["y"] = 1 + 0.5*df["x1"] + 0.8*df["x2"] + rng.normal(scale=0.1, size=N)

    res_f = smf.ols("y ~ x1 + x2", data=df).fit()
    M_f = Margins(res_f)
    me_f = M_f.dydx("x1")
    pred_f = M_f.predict(at="mean")

    X_df = sm.add_constant(df[["x1", "x2"]])
    res_r = sm.OLS(df["y"], X_df).fit()
    M_r = Margins(res_r)
    assert M_r._raw_mode is True
    me_r = M_r.dydx("x1")
    pred_r = M_r.predict(at="mean")

    assert_allclose(me_f.estimate[0], me_r.estimate[0], atol=1e-10)
    assert_allclose(me_f.se[0], me_r.se[0], atol=1e-10)
    assert_allclose(pred_f.estimate[0], pred_r.estimate[0], atol=1e-10)


# ── Validation ─────────────────────────────────────────────────────────────

def test_raw_exog_validation():
    """Raw mode validates variable names and rejects constant/Intercept.

    In raw mode, dydx() checks that the variable exists in model.exog_names
    and raises a clear error for the constant term (which has no meaningful
    marginal effect).
    """
    rng = np.random.default_rng(42)
    N = 20
    X = sm.add_constant(rng.normal(size=(N, 2)))
    y = X @ np.array([1, 2, 3]) + rng.normal(size=N)
    res = sm.OLS(y, X).fit()

    M = Margins(res)
    assert M._raw_mode is True

    with pytest.raises(ValueError, match="not found in model's exog columns"):
        M.dydx("nonexistent")
    with pytest.raises(ValueError, match="Cannot compute marginal effect for the constant"):
        M.dydx("const")


# ── Interaction limitation (xfail) ─────────────────────────────────────────

@pytest.mark.xfail(strict=True, reason=(
    "Raw mode does not know about interactions between design matrix columns. "
    "Perturbing x1 does not automatically update x1x2 = x1*x2. "
    "Documented limitation: use formula mode for models with interactions."
))
def test_raw_exog_interaction_limitation():
    """Raw mode fails to track interactions -- xfail until fixed.

    In formula mode, y ~ x1 * x2 correctly handles the interaction:
    when x1 is perturbed, the interaction term x1:x2 is also updated.

    In raw mode with manually-built X, the interaction column x1x2
    is not updated when x1 is perturbed, giving wrong marginal effects.
    The true ME is beta_x1 + beta_x1x2 * mean(x2), but raw mode gives
    only beta_x1.

    When fixed, this xfail should flip to passing.
    """
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "x1": rng.normal(size=N),
        "x2": rng.normal(size=N),
    })
    df["x1x2"] = df["x1"] * df["x2"]
    df["y"] = 1 + 2*df["x1"] + 3*df["x2"] + 4*df["x1x2"] + rng.normal(scale=0.1, size=N)

    res_f = smf.ols("y ~ x1 * x2", data=df).fit()
    M_f = Margins(res_f)
    me_f = M_f.dydx("x1", at="mean")

    X_raw = sm.add_constant(df[["x1", "x2", "x1x2"]])
    res_r = sm.OLS(df["y"], X_raw).fit()
    M_r = Margins(res_r)
    me_r = M_r.dydx("x1", at="mean")

    # This assertion will fail because raw mode gives ~2.0 instead of ~2.0 + 4.0*mean(x2)
    # When fixed, this xfail should flip to passing
    assert_allclose(me_f.estimate[0], me_r.estimate[0], atol=1e-10)


# ── No names (auto-generated) ──────────────────────────────────────────────

def test_raw_exog_no_names():
    """Raw mode with numpy arrays gets auto-generated x0, x1, ... names.

    When fitting with a plain numpy array (no column names), statsmodels
    auto-generates exog_names as ['x1', 'x2', ...]. Margins should be
    able to reference variables by these auto-generated names.
    """
    rng = np.random.default_rng(42)
    N = 1000
    X = rng.normal(size=(N, 2))
    y = X @ np.array([2, 3]) + rng.normal(scale=0.1, size=N)
    res = sm.OLS(y, X).fit()

    M = Margins(res)
    assert M._raw_mode is True
    assert "x1" in res.model.exog_names
    me = M.dydx("x1")
    assert_allclose(me.estimate[0], 2, atol=0.1)
