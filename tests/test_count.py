"""
Count-variable marginal effects (count=True) tests.

For integer-valued covariates (e.g., number of children), the marginal effect
is the change in E[y] when x increases by 1 unit: E[f(x+1)] - E[f(x)].
This is a discrete contrast rather than a continuous derivative.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from numpy.testing import assert_allclose
from smmargins import Margins


# ── AME with count=True ────────────────────────────────────────────────────

def test_count_dydx_hand_check():
    """AME for count variable matches hand-rolled mean(predict(X+1) - predict(X)).

    For a Poisson model with identity link (via predict), the count ME is:
        (1/N) * sum_i [predict(x_i + 1) - predict(x_i)]
    This is a direct finite-difference contrast, not a derivative.
    """
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "kids": rng.integers(0, 5, N),
        "income": rng.normal(50, 10, N),
    })
    df["y"] = rng.poisson(np.exp(0.5 * df["kids"] - 0.01 * df["income"]))

    model = smf.poisson("y ~ kids + income", data=df).fit(disp=0)
    M = Margins(model)
    est = M.dydx("kids", count=True)

    # Hand-rolled: mean(predict(X+1) - predict(X))
    df_p = df.copy()
    df_p["kids"] = df_p["kids"] + 1
    p0 = model.predict(df)
    p1 = model.predict(df_p)
    hand_est = np.mean(p1 - p0)
    assert_allclose(est.estimate[0], hand_est, atol=1e-10)


# ── MEM with count=True ────────────────────────────────────────────────────

def test_count_mem_hand_check():
    """MEM (marginal effect at means) for count variable matches hand computation.

    At means: compute prediction at mean(x), then at mean(x)+1, take difference.
    """
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "kids": rng.integers(0, 5, N),
        "income": rng.normal(50, 10, N),
    })
    df["y"] = rng.poisson(np.exp(0.5 * df["kids"] - 0.01 * df["income"]))

    model = smf.poisson("y ~ kids + income", data=df).fit(disp=0)
    M = Margins(model)
    est_mem = M.dydx("kids", at="mean", count=True)

    # Hand-rolled MEM
    df_mean = df.mean(numeric_only=True).to_frame().T
    df_mean_p = df_mean.copy()
    df_mean_p["kids"] = df_mean_p["kids"] + 1
    hand_mem = model.predict(df_mean_p)[0] - model.predict(df_mean)[0]
    assert_allclose(est_mem.estimate[0], hand_mem)


# ── Multi-variable count ───────────────────────────────────────────────────

def test_count_multi_variable():
    """dydx with list of variables and count=True returns results for all vars."""
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "kids": rng.integers(0, 5, N),
        "income": rng.normal(50, 10, N),
    })
    df["y"] = rng.poisson(np.exp(0.5 * df["kids"] - 0.01 * df["income"]))

    model = smf.poisson("y ~ kids + income", data=df).fit(disp=0)
    M = Margins(model)
    est_multi = M.dydx(["kids", "income"], count=True)

    assert len(est_multi.estimate) == 2
    assert "kids (count)" in est_multi.labels[0]
    assert "income (count)" in est_multi.labels[1]
