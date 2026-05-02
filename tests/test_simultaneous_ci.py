"""Tests for simultaneous confidence intervals."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from numpy.testing import assert_allclose

from smmargins import Margins


def test_bonferroni_width_exact(sim_frame):
    """Bonferroni CI half-width = pointwise half-width * z(1 - 0.05/(2*m)) / z(1 - 0.05/2)."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    # m = 5 margins (AAP, APM, APR at 3 values... no, use dydx for simplicity)
    # Use predict with atexog to get m=5 margins
    ame = M.dydx(["x1", "x2"], vce="delta", ci_method="pointwise")
    bonf = M.dydx(["x1", "x2"], vce="delta", ci_method="bonferroni")

    m = len(ame.estimate)
    from scipy import stats
    z_pw = stats.norm.ppf(1 - 0.05 / 2)
    z_bf = stats.norm.ppf(1 - 0.05 / (2 * m))

    for i in range(m):
        hw_pw = (ame.ci_upper[i] - ame.ci_lower[i]) / 2
        hw_bf = (bonf.ci_upper[i] - bonf.ci_lower[i]) / 2
        expected_hw_bf = hw_pw * (z_bf / z_pw)
        assert_allclose(hw_bf, expected_hw_bf, atol=1e-10)


def test_sidak_narrower_than_bonferroni(sim_frame):
    """Sidak CIs should be very slightly narrower than Bonferroni."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    bonf = M.dydx(["x1", "x2"], vce="delta", ci_method="bonferroni")
    sidak = M.dydx(["x1", "x2"], vce="delta", ci_method="sidak")

    for i in range(len(bonf.estimate)):
        hw_bonf = (bonf.ci_upper[i] - bonf.ci_lower[i]) / 2
        hw_sidak = (sidak.ci_upper[i] - sidak.ci_lower[i]) / 2
        assert hw_sidak < hw_bonf


def test_sup_t_narrower_than_bonferroni_for_correlated_margins(sim_frame):
    """Sup-t should give visibly narrower CIs than Bonferroni for correlated margins."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    # Predict at adjacent values of x1 -> highly correlated margins
    pred = M.predict(atexog={"x1": [-1, 0, 1]}, vce="simulation",
                     n_sims=5000, sim_seed=1, ci_method="sup-t")
    bonf = M.predict(atexog={"x1": [-1, 0, 1]}, vce="simulation",
                     n_sims=5000, sim_seed=1, ci_method="bonferroni")

    for i in range(len(pred.estimate)):
        hw_sup = (pred.ci_upper[i] - pred.ci_lower[i]) / 2
        hw_bonf = (bonf.ci_upper[i] - bonf.ci_lower[i]) / 2
        assert hw_sup < hw_bonf


def test_sup_t_requires_draws(sim_frame):
    """ci_method='sup-t' with vce='delta' raises ValueError."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    with pytest.raises(ValueError, match="sup-t requires draws"):
        M.dydx("x1", vce="delta", ci_method="sup-t")


@pytest.mark.slow
@pytest.mark.skipif(
    pytest.config.getoption("-m") == "slow" if hasattr(pytest, "config") else False,
    reason="slow test only run when explicitly requested",
)
def test_family_wise_coverage_simulation():
    """Slow simulation: verify family-wise coverage for each ci_method."""
    # This test is very slow; run with pytest -m slow
    rng = np.random.default_rng(123)
    n_sim = 200
    n = 500
    m = 5
    coverage = {"pointwise": 0, "bonferroni": 0, "sidak": 0, "sup-t": 0}

    for _ in range(n_sim):
        df = pd.DataFrame(rng.normal(0, 1, (n, m)), columns=[f"x{i}" for i in range(m)])
        df["y"] = df.sum(axis=1) + rng.normal(0, 1, n)
        fit = smf.ols("y ~ " + " + ".join(df.columns[:-1]), data=df).fit()
        M = Margins(fit)

        # True margins are all 1.0
        true = np.ones(m)

        for method in coverage:
            if method == "sup-t":
                res = M.dydx([f"x{i}" for i in range(m)], vce="simulation",
                             n_sims=1000, sim_seed=rng.integers(0, 1_000_000),
                             ci_method="sup-t")
            else:
                res = M.dydx([f"x{i}" for i in range(m)], vce="delta",
                             ci_method=method)
            covered = np.all((res.ci_lower <= true) & (true <= res.ci_upper))
            coverage[method] += covered

    for method, count in coverage.items():
        rate = count / n_sim
        if method == "pointwise":
            assert rate < 0.90  # should under-cover
        else:
            assert rate >= 0.90  # should be near 0.95
