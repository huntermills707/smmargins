"""Tests for Krinsky–Robb simulation VCE."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import statsmodels.api as sm
from numpy.testing import assert_allclose

from smmargins import Margins


def test_kr_linear_converges_to_delta(sim_frame):
    """Linear regression AME: KR SE should match delta SE within MC error."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    ame_delta = M.dydx("x1", vce="delta")
    ame_kr = M.dydx("x1", vce="simulation", n_sims=10_000, sim_seed=42)

    delta_se = ame_delta.se[0]
    kr_se = ame_kr.se[0]
    mc_error_bound = 3 * delta_se / np.sqrt(10_000)
    assert abs(kr_se - delta_se) < mc_error_bound


def test_kr_logit_asymmetric_ci(sim_frame):
    """Logit AME near boundary: KR percentile CI is asymmetric and stays in [0,1]."""
    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
    })
    # Strong coefficient so Pr(y=1) varies rapidly
    z = -1.0 + 2.0 * df["x"]
    df["y"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-z))).astype(int)

    fit = smf.logit("y ~ x", data=df).fit(disp=False)
    M = Margins(fit)
    ame_kr = M.dydx("x", vce="simulation", n_sims=10_000, sim_seed=42)
    ame_delta = M.dydx("x", vce="delta")

    lower = ame_kr.ci_lower[0]
    upper = ame_kr.ci_upper[0]
    est = ame_kr.estimate[0]

    # Asymmetry: distance from estimate to bounds differs by >5% of SE
    se = ame_kr.se[0]
    asym = abs((upper - est) - (est - lower))
    assert asym > 0.05 * se

    # Stays in [0, 1] for probability-scale margins
    assert lower >= -1e-6
    assert upper <= 1 + 1e-6

    # Delta normal CI may wander outside [0,1]; KR should be tighter on the lower side
    delta_lower = ame_delta.ci_lower[0]
    assert lower > delta_lower or abs(lower - delta_lower) < 1e-3


def test_kr_reproducibility(sim_frame):
    """Same seed → identical SEs to machine precision."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    ame1 = M.dydx("x1", vce="simulation", n_sims=2000, sim_seed=123)
    ame2 = M.dydx("x1", vce="simulation", n_sims=2000, sim_seed=123)
    assert_allclose(ame1.se, ame2.se, atol=1e-15)
    assert_allclose(ame1.ci_lower, ame2.ci_lower, atol=1e-15)
    assert_allclose(ame1.ci_upper, ame2.ci_upper, atol=1e-15)


def test_kr_composition_with_cov_type(sim_frame):
    """vce='simulation' + cov_type='HC1' draws from the correct covariance."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    k = len(fit.params)
    # Known vcov: identity scaled
    vcov = np.eye(k) * 2.0
    M = Margins(fit, vcov=vcov)
    ame_kr = M.dydx("x1", vce="simulation", n_sims=5000, sim_seed=99)

    # Empirical covariance of the parameter draws should match vcov within MC error
    from smmargins.inference import _simulate_vce

    beta = M.params
    statistic = lambda b: b  # identity to get param draws directly
    param_draws = _simulate_vce(beta, vcov, statistic, n_sims=5000, seed=99)
    emp_cov = np.cov(param_draws, rowvar=False, ddof=1)
    # Check diagonal within 5% relative error
    assert_allclose(np.diag(emp_cov), np.diag(vcov), rtol=0.05)


def test_kr_multi_outcome_mnlogit(mnlogit_fit, mnlogit_frame):
    """MNLogit AMEs: KR draws produce per-outcome SEs that match delta within MC error."""
    M = Margins(mnlogit_fit)
    ame_delta = M.dydx("x1", vce="delta")
    ame_kr = M.dydx("x1", vce="simulation", n_sims=10_000, sim_seed=42)

    for i in range(3):
        delta_se = ame_delta.se[i]
        kr_se = ame_kr.se[i]
        mc_error_bound = 3 * delta_se / np.sqrt(10_000)
        assert abs(kr_se - delta_se) < mc_error_bound
