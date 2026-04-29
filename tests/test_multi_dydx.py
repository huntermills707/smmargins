"""
Multi-variable dydx and '*' enumeration tests.

These tests verify:
1. dydx([x1, x2, group]) stacks individual results with correct joint vcov
2. Cross-variable contrasts using the joint vcov
3. '*' enumerates all non-response columns
4. Parity with statsmodels get_margeff
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.testing import assert_allclose
import pytest
from smmargins import Margins


# ── Multi-variable equivalence ─────────────────────────────────────────────

def test_multi_variable_equivalence():
    """dydx([x1, x2, group]) == concatenation of individual dydx calls.

    When requesting marginal effects for multiple variables simultaneously,
    the estimates and SEs should match the individual calls. The joint vcov
    should be a square matrix with nonzero off-diagonal blocks reflecting
    the shared parameter uncertainty.
    """
    rng = np.random.default_rng(42)
    N = 500
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, N),
        "x2": rng.normal(0, 1, N),
        "group": rng.choice(["A", "B"], N),
        "y": 0
    })
    z = 0.5 * df["x1"] - 0.3 * df["x2"] + 0.4 * (df["group"] == "B")
    df["y"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-z))).astype(int)

    model = smf.logit("y ~ x1 + x2 + C(group)", data=df).fit(disp=False)
    M = Margins(model)

    m1 = M.dydx("x1")
    m2 = M.dydx("x2")
    mg = M.dydx("group")
    joint = M.dydx(["x1", "x2", "group"])

    expected_est = np.concatenate([m1.estimate, m2.estimate, mg.estimate])
    expected_se = np.concatenate([m1.se, m2.se, mg.se])
    assert_allclose(joint.estimate, expected_est, atol=1e-12)
    assert_allclose(joint.se, expected_se, rtol=1e-10)

    # Joint vcov sanity
    n_rows = len(joint.estimate)
    assert joint.vcov.shape == (n_rows, n_rows)
    # Off-diagonal block between x1 and x2 should be nonzero
    # (they share the same parameter uncertainty)
    assert not np.isclose(joint.vcov[0, 1], 0.0)


# ── Cross-variable contrast ────────────────────────────────────────────────

def test_cross_variable_contrast():
    """Contrast using joint vcov: x1 AME - x2 AME.

    With the joint covariance from multi-variable dydx, we can test
    hypotheses involving multiple marginal effects, e.g.:
        H0: AME(x1) = AME(x2)  <=>  AME(x1) - AME(x2) = 0
    """
    rng = np.random.default_rng(42)
    N = 500
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, N),
        "x2": rng.normal(0, 1, N),
        "group": rng.choice(["A", "B"], N),
        "y": 0
    })
    z = 0.5 * df["x1"] - 0.3 * df["x2"] + 0.4 * (df["group"] == "B")
    df["y"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-z))).astype(int)

    model = smf.logit("y ~ x1 + x2 + C(group)", data=df).fit(disp=False)
    M = Margins(model)
    joint = M.dydx(["x1", "x2", "group"])

    c = np.array([1, -1, 0])
    res_contrast = joint.contrast(c)
    expected_contrast = joint.estimate[0] - joint.estimate[1]
    assert_allclose(res_contrast.estimate[0], expected_contrast)
    expected_vcov_c = c @ joint.vcov @ c.T
    assert_allclose(res_contrast.vcov[0, 0], expected_vcov_c)
    assert res_contrast.se[0] > 0


# ── Star enumeration ───────────────────────────────────────────────────────

def test_star_enumeration():
    """dydx('*') enumerates all non-response columns in the data frame.

    The '*' wildcard expands to all columns in self.data except the
    response variable (endog). This includes columns not used in the
    model formula (e.g., 'unused' here).
    """
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, N),
        "x2": rng.normal(0, 1, N),
        "unused": rng.normal(0, 1, N),
        "y": rng.normal(0, 1, N)
    })
    model = smf.ols("y ~ x1 + x2", data=df).fit()
    M = Margins(model)

    res = M.dydx("*")
    # Should include x1, x2, unused. Response 'y' is excluded.
    assert len(res.estimate) == 3
    labels_str = "".join(res.labels)
    assert "x1" in labels_str
    assert "x2" in labels_str
    assert "unused" in labels_str


# ── Parity with statsmodels get_margeff ────────────────────────────────────

@pytest.mark.parity
def test_star_parity_with_get_margeff():
    """dydx('*', at='overall') agrees with statsmodels get_margeff.

    This is a cross-implementation check: our AME for a simple logit
    should match statsmodels' own marginal effects computation.
    """
    rng = np.random.default_rng(42)
    N = 500
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, N),
        "x2": rng.normal(0, 1, N),
        "y": 0
    })
    df["y"] = (rng.uniform(0, 1, N) < 0.5).astype(int)

    model = smf.logit("y ~ x1 + x2", data=df).fit(disp=False)
    M = Margins(model)
    res = M.dydx("*", at="overall")
    sm_res = model.get_margeff(at="overall", method="dydx")

    for i, label in enumerate(res.labels):
        if label == "dx1":
            assert_allclose(res.estimate[i], sm_res.margeff[0], atol=1e-6)
            assert_allclose(res.se[i], sm_res.margeff_se[0], rtol=1e-3)
        elif label == "dx2":
            assert_allclose(res.estimate[i], sm_res.margeff[1], atol=1e-6)
            assert_allclose(res.se[i], sm_res.margeff_se[1], rtol=1e-3)
