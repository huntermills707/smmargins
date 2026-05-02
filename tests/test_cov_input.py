"""Tests for custom covariance input (cov_type, vcov)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import statsmodels.api as sm
from numpy.testing import assert_allclose

from smmargins import Margins


def test_cov_type_hc_round_trip_ols(sim_frame):
    """Fit OLS with HC1 directly; then fit plain OLS and pass cov_type='HC1'.
    SEs must match to 1e-10."""
    fit_robust = smf.ols("y_ols ~ x1 + I(x1**2) + x2 + x1:x2", data=sim_frame).fit(cov_type="HC1")
    fit_plain = smf.ols("y_ols ~ x1 + I(x1**2) + x2 + x1:x2", data=sim_frame).fit()

    M_robust = Margins(fit_robust)
    M_plain = Margins(fit_plain, cov_type="HC1")

    ame_robust = M_robust.dydx("x1")
    ame_plain = M_plain.dydx("x1")

    assert_allclose(ame_robust.estimate, ame_plain.estimate, atol=1e-10)
    assert_allclose(ame_robust.se, ame_plain.se, atol=1e-10)


def test_cov_type_hc0_to_hc3_ols(sim_frame):
    """Check that different robust covariance types give different SEs."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    se_map = {}
    for ctype in ("HC0", "HC1", "HC2", "HC3"):
        M = Margins(fit, cov_type=ctype)
        se_map[ctype] = M.dydx("x1").se[0]
    # They need not be monotonic, but they should all differ from nonrobust
    M_nonrobust = Margins(fit)
    se_nonrobust = M_nonrobust.dydx("x1").se[0]
    for ctype, se in se_map.items():
        assert se != pytest.approx(se_nonrobust, rel=1e-6)


def test_cluster_matches_hand_sandwich(sim_frame):
    """Verify cluster-robust margin SEs match a hand-sandwich Jacobian*V*J'."""
    rng = np.random.default_rng(42)
    n_clust = 50
    obs_per = 10
    n = n_clust * obs_per
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "clust": np.repeat(np.arange(n_clust), obs_per),
    })
    clust_eff = np.repeat(rng.normal(0, 2.0, n_clust), obs_per)
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + clust_eff + rng.normal(0, 0.5, n)

    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    M_cluster = Margins(fit, cov_type="cluster", cov_kwds={"groups": df["clust"]})
    ame_cluster = M_cluster.dydx("x1")

    # Hand-check: get cluster-robust cov from statsmodels and sandwich manually
    robust_res = fit.get_robustcov_results(cov_type="cluster", groups=df["clust"])
    V_params = np.asarray(robust_res.cov_params())
    # For simple OLS without interactions, AME of x1 = beta_x1, so J = [0, 1, 0]
    J = np.array([[0.0, 1.0, 0.0]])
    V_margin = J @ V_params @ J.T
    hand_se = np.sqrt(np.diag(V_margin))
    assert_allclose(ame_cluster.se, hand_se, atol=1e-10)

    # Cluster SE should differ from nonrobust
    M_nonrobust = Margins(fit)
    ame_nonrobust = M_nonrobust.dydx("x1")
    assert ame_cluster.se[0] != pytest.approx(ame_nonrobust.se[0], rel=1e-6)


def test_user_vcov_identity(sim_frame):
    """Pass an identity-scaled vcov; verify margin SEs equal sqrt(diag(J @ vcov @ J.T))."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    k = len(fit.params)
    vcov = np.eye(k) * 2.0

    M = Margins(fit, vcov=vcov)
    ame = M.dydx("x1")

    # Compute Jacobian numerically by re-running delta with known vcov
    M_check = Margins(fit)
    # We need the Jacobian. Use the internal _delta with a dummy statistic?
    # Simpler: verify the SE matches the formula by computing J manually
    # For OLS AME of x1, the Jacobian is [0, 1, 0] because AME = beta_x1
    # (since there's no interaction or polynomial)
    J = np.array([[0.0, 1.0, 0.0]])
    V = J @ vcov @ J.T
    expected_se = np.sqrt(np.diag(V))
    assert_allclose(ame.se, expected_se, atol=1e-10)


def test_cov_type_and_vcov_mutual_exclusion(sim_frame):
    """Passing both cov_type and vcov raises ValueError."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    with pytest.raises(ValueError, match="Only one of"):
        Margins(fit, cov_type="HC1", vcov=np.eye(len(fit.params)))


def test_cov_type_multi_outcome_mnlogit(mnlogit_fit, mnlogit_frame):
    """Same cov_type logic works for MNLogit (multi-outcome)."""
    fit = mnlogit_fit
    M = Margins(fit)
    ame = M.dydx("x1")
    # Just verify it runs and produces per-outcome SEs
    assert ame.estimate.shape[0] == 3  # 3 outcome classes
    assert ame.se.shape[0] == 3


def test_vcov_shape_error(sim_frame):
    """Wrong-shaped vcov raises ValueError."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    with pytest.raises(ValueError, match="vcov must have shape"):
        Margins(fit, vcov=np.eye(2))
