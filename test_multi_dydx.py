import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from marginal_effects import Margins

def test_multi_variable_equivalence():
    rng = np.random.default_rng(42)
    N = 500
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, N),
        "x2": rng.normal(0, 1, N),
        "group": rng.choice(["A", "B"], N),
        "y": 0
    })
    # binary outcome
    z = 0.5 * df["x1"] - 0.3 * df["x2"] + 0.4 * (df["group"] == "B")
    df["y"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-z))).astype(int)

    model = smf.logit("y ~ x1 + x2 + C(group)", data=df).fit(disp=False)
    M = Margins(model)

    # Solo calls
    m1 = M.dydx("x1")
    m2 = M.dydx("x2")
    mg = M.dydx("group")

    # Joint call
    joint = M.dydx(["x1", "x2", "group"])

    # 1. Equivalence of estimates and SEs
    expected_est = np.concatenate([m1.estimate, m2.estimate, mg.estimate])
    expected_se = np.concatenate([m1.se, m2.se, mg.se])
    
    np.testing.assert_allclose(joint.estimate, expected_est, atol=1e-12)
    np.testing.assert_allclose(joint.se, expected_se, rtol=1e-10)

    # 2. Joint vcov sanity
    n_rows = len(joint.estimate)
    assert joint.vcov.shape == (n_rows, n_rows)
    # Off-diagonal blocks should generally be nonzero
    # block between x1 (row 0) and x2 (row 1)
    assert not np.isclose(joint.vcov[0, 1], 0.0)

    # 3. Cross-variable contrast
    # Contrast: x1 AME - x2 AME
    # joint.estimate is [AME(x1), AME(x2), AME(group: B vs A)]
    c = np.array([1, -1, 0])
    res_contrast = joint.contrast(c)
    
    expected_contrast = joint.estimate[0] - joint.estimate[1]
    np.testing.assert_allclose(res_contrast.estimate[0], expected_contrast)
    
    expected_vcov_c = c @ joint.vcov @ c.T
    np.testing.assert_allclose(res_contrast.vcov[0, 0], expected_vcov_c)
    assert res_contrast.se[0] > 0

def test_star_enumeration():
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
    # Total columns in df is 4. 4 - 1 = 3.
    assert len(res.estimate) == 3
    # Check labels contain the names
    labels_str = "".join(res.labels)
    assert "x1" in labels_str
    assert "x2" in labels_str
    assert "unused" in labels_str

def test_parity_with_get_margeff():
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
    
    # M.dydx("*", at="overall")
    res = M.dydx("*", at="overall")
    # statsmodels get_margeff
    sm_res = model.get_margeff(at="overall", method="dydx")
    
    # Compare x1 and x2
    # res might have 'unused' if we had one, but here just x1, x2
    # Match by label
    for i, label in enumerate(res.labels):
        if label == "dx1":
            idx_sm = 0 # usually
            np.testing.assert_allclose(res.estimate[i], sm_res.margeff[0], atol=1e-6)
            np.testing.assert_allclose(res.se[i], sm_res.margeff_se[0], rtol=1e-3)
        if label == "dx2":
            np.testing.assert_allclose(res.estimate[i], sm_res.margeff[1], atol=1e-6)
            np.testing.assert_allclose(res.se[i], sm_res.margeff_se[1], rtol=1e-3)

if __name__ == "__main__":
    test_multi_variable_equivalence()
    test_star_enumeration()
    test_parity_with_get_margeff()
    print("All multi-variable dydx tests passed!")
