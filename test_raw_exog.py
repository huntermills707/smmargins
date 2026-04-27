import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from marginal_effects import Margins
# import pytest

class MockPytest:
    class raises:
        def __init__(self, exc, match=None):
            self.exc = exc
            self.match = match
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc} but no exception was raised")
            if not issubclass(exc_type, self.exc):
                raise AssertionError(f"Expected {self.exc} but got {exc_type}")
            if self.match and self.match not in str(exc_val):
                raise AssertionError(f"Expected match {self.match!r} but got {exc_val!r}")
            return True

pytest = MockPytest()

def test_raw_exog_equivalence():
    """Test that formula and raw exog fits give identical results for additive models."""
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "x1": rng.normal(size=N),
        "x2": rng.normal(size=N),
    })
    # Additive model: y = 1 + 0.5*x1 + 0.8*x2 + e
    df["y"] = 1 + 0.5*df["x1"] + 0.8*df["x2"] + rng.normal(scale=0.1, size=N)
    
    # 1. Formula fit
    res_f = smf.ols("y ~ x1 + x2", data=df).fit()
    M_f = Margins(res_f)
    me_f = M_f.dydx("x1")
    pred_f = M_f.predict(at="mean")
    
    # 2. Raw fit (passing values)
    X = sm.add_constant(df[["x1", "x2"]].values)
    # This will have exog_names ['const', 'x1', 'x2'] because SM pulls them from the DF
    # when we pass X as a DF, or x1, x2 if we pass X as a value but SM is smart.
    # Let's pass it as a DataFrame to be sure about names.
    X_df = sm.add_constant(df[["x1", "x2"]])
    res_r = sm.OLS(df["y"], X_df).fit()
    
    M_r = Margins(res_r)
    assert M_r._raw_mode is True
    me_r = M_r.dydx("x1")
    pred_r = M_r.predict(at="mean")
    
    print(f"Formula ME: {me_f.estimate[0]:.6f} (SE: {me_f.se[0]:.6f})")
    print(f"Raw ME:     {me_r.estimate[0]:.6f} (SE: {me_r.se[0]:.6f})")
    
    assert np.isclose(me_f.estimate[0], me_r.estimate[0], atol=1e-10)
    assert np.isclose(me_f.se[0], me_r.se[0], atol=1e-10)
    assert np.isclose(pred_f.estimate[0], pred_r.estimate[0], atol=1e-10)

def test_raw_exog_validation():
    """Test validation of variable names and constants in raw mode."""
    rng = np.random.default_rng(42)
    N = 20
    X = sm.add_constant(rng.normal(size=(N, 2)))
    y = X @ np.array([1, 2, 3]) + rng.normal(size=N)
    res = sm.OLS(y, X).fit()
    # exog_names will be ['const', 'x1', 'x2']
    
    M = Margins(res)
    assert M._raw_mode is True
    
    # Bad name
    with pytest.raises(ValueError, match="not found in model's exog columns"):
        M.dydx("nonexistent")
        
    # Constant
    with pytest.raises(ValueError, match="Cannot compute marginal effect for the constant"):
        M.dydx("const")

def test_raw_exog_interaction_limitation():
    """Demonstrate (and document) that raw mode fails to track interactions."""
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "x1": rng.normal(size=N),
        "x2": rng.normal(size=N),
    })
    df["x1x2"] = df["x1"] * df["x2"]
    df["y"] = 1 + 2*df["x1"] + 3*df["x2"] + 4*df["x1x2"] + rng.normal(scale=0.1, size=N)
    
    # 1. Formula fit: y ~ x1 * x2
    res_f = smf.ols("y ~ x1 * x2", data=df).fit()
    M_f = Margins(res_f)
    # E[dy/dx1] = 2 + 4 * mean(x2)
    me_f = M_f.dydx("x1", at="mean")
    
    # 2. Raw fit: X = [1, x1, x2, x1x2]
    X_raw = sm.add_constant(df[["x1", "x2", "x1x2"]])
    res_r = sm.OLS(df["y"], X_raw).fit()
    M_r = Margins(res_r)
    me_r = M_r.dydx("x1", at="mean")
    
    print(f"\nInteraction Limitation Check:")
    print(f"Formula ME (correctly handles interaction): {me_f.estimate[0]:.6f}")
    print(f"Raw ME (incorrectly ignores interaction):  {me_r.estimate[0]:.6f}")
    
    # In raw mode, only the 'x1' column is perturbed. 
    # Its coefficient is ~2. The 'x1x2' column coefficient is ~4.
    # me_r.estimate should be approx 2.0.
    # me_f.estimate should be approx 2.0 + 4.0 * mean(x2).
    
    assert not np.isclose(me_f.estimate[0], me_r.estimate[0])
    assert np.isclose(me_r.estimate[0], res_r.params["x1"], atol=1e-10)

def test_raw_exog_no_names():
    """Test raw mode with numpy arrays and no names (auto-generated x0, x1...)."""
    rng = np.random.default_rng(42)
    N = 1000  # More data for better precision
    X = rng.normal(size=(N, 2))
    y = X @ np.array([2, 3]) + rng.normal(scale=0.1, size=N)
    res = sm.OLS(y, X).fit()
    
    M = Margins(res)
    assert M._raw_mode is True
    # Statsmodels defaults to x1, x2 for no-intercept models with numpy input
    print(f"No-names exog_names: {res.model.exog_names}")
    assert "x1" in res.model.exog_names
    me = M.dydx("x1")
    print(f"Estimate for x1: {me.estimate[0]}")
    assert np.isclose(me.estimate[0], 2, atol=0.1)

if __name__ == "__main__":
    # Run tests manually since pytest might not be available or used in this env's flow
    try:
        test_raw_exog_equivalence()
        print("test_raw_exog_equivalence: PASSED")
        test_raw_exog_validation()
        print("test_raw_exog_validation: PASSED")
        test_raw_exog_interaction_limitation()
        print("test_raw_exog_interaction_limitation: PASSED")
        test_raw_exog_no_names()
        print("test_raw_exog_no_names: PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
