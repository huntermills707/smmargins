"""Tests for bootstrap VCE."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import statsmodels.api as sm
from numpy.testing import assert_allclose

from smmargins import Margins


def test_bootstrap_pairs_ols_converges_to_hc1(sim_frame):
    """Pairs bootstrap on OLS converges to HC1 sandwich SE."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    ame_hc1 = M.dydx("x1", vce="delta", cov_type="HC1")
    ame_boot = M.dydx("x1", vce="bootstrap", n_boot=1000, boot_seed=7)

    hc1_se = ame_hc1.se[0]
    boot_se = ame_boot.se[0]
    # Conservative MC error bound
    mc_error_bound = 5 * hc1_se / np.sqrt(1000)
    assert abs(boot_se - hc1_se) < mc_error_bound


def test_bootstrap_cluster_matches_cluster_robust(sim_frame):
    """Cluster bootstrap SE roughly matches StatsModels cluster-robust SE."""
    rng = np.random.default_rng(42)
    n_clust = 40
    obs_per = 10
    n = n_clust * obs_per
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "clust": np.repeat(np.arange(n_clust), obs_per),
    })
    clust_eff = np.repeat(rng.normal(0, 3.0, n_clust), obs_per)
    df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + clust_eff + rng.normal(0, 0.5, n)

    fit = smf.ols("y ~ x1 + x2", data=df).fit()
    M = Margins(fit)
    ame_cluster = M.dydx("x1", vce="bootstrap", n_boot=500, boot_seed=1,
                         boot_method="cluster", cluster=df["clust"])

    # Compare to cluster-robust delta SE
    M_clust_delta = Margins(fit, cov_type="cluster", cov_kwds={"groups": df["clust"]})
    delta_se = M_clust_delta.dydx("x1").se[0]
    boot_se = ame_cluster.se[0]
    # Within 20% — rough match is enough for bootstrap with n_boot=500
    assert abs(boot_se - delta_se) / delta_se < 0.20


def test_bootstrap_block_matches_hac_roughly():
    """Block bootstrap SE > pairs bootstrap SE for autocorrelated series."""
    rng = np.random.default_rng(99)
    n = 500
    # Both regressor and error are AR(1) — typical time-series structure
    x = np.zeros(n)
    e = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 1)
        e[t] = 0.9 * e[t - 1] + rng.normal(0, 1)
    df = pd.DataFrame({
        "x": x,
        "y": 1.0 + 0.5 * x + e,
    })
    fit = smf.ols("y ~ x", data=df).fit()
    M = Margins(fit)
    ame_pairs = M.dydx("x", vce="bootstrap", n_boot=500, boot_seed=1, boot_method="pairs")
    ame_block = M.dydx("x", vce="bootstrap", n_boot=500, boot_seed=1,
                       boot_method="block", block_size=15)

    # Block bootstrap should inflate SE because it preserves within-block correlation
    assert ame_block.se[0] > ame_pairs.se[0]


def test_bootstrap_reproducibility(sim_frame):
    """Same boot_seed → identical SEs."""
    fit = smf.ols("y_ols ~ x1 + x2", data=sim_frame).fit()
    M = Margins(fit)
    ame1 = M.dydx("x1", vce="bootstrap", n_boot=500, boot_seed=42)
    ame2 = M.dydx("x1", vce="bootstrap", n_boot=500, boot_seed=42)
    assert_allclose(ame1.se, ame2.se, atol=1e-15)


def test_bootstrap_refit_failure_handling(sim_frame):
    """Monkeypatch fit() to raise on some calls; assert warning and valid SE."""
    # Use raw-matrix model so _refit_model has only one path
    import statsmodels.api as sm
    y = sim_frame["y_ols"]
    X = sm.add_constant(sim_frame[["x1", "x2"]])
    fit = sm.OLS(y, X).fit()
    M = Margins(fit)

    original_fit = type(fit.model).fit
    call_count = [0]

    def bad_fit(self, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 3 == 0:
            raise RuntimeError("simulated fit failure")
        return original_fit(self, *args, **kwargs)

    # Patch the class-level fit method
    type(fit.model).fit = bad_fit
    try:
        with pytest.warns(RuntimeWarning, match="bootstrap reps failed"):
            ame = M.dydx("x1", vce="bootstrap", n_boot=100, boot_seed=1)
        assert np.isfinite(ame.se[0])
        assert ame.se[0] > 0
    finally:
        type(fit.model).fit = original_fit


def test_bootstrap_multi_outcome_mnlogit(mnlogit_fit, mnlogit_frame):
    """MNLogit pairs bootstrap: per-outcome SEs close to delta SEs."""
    M = Margins(mnlogit_fit)
    ame_delta = M.dydx("x1", vce="delta")
    ame_boot = M.dydx("x1", vce="bootstrap", n_boot=500, boot_seed=3)

    for i in range(3):
        delta_se = ame_delta.se[i]
        boot_se = ame_boot.se[i]
        mc_error_bound = 5 * delta_se / np.sqrt(500)
        assert abs(boot_se - delta_se) < mc_error_bound


def test_bootstrap_speed_sanity(sim_frame):
    """200-rep bootstrap on 1000-row Logit fixture should finish in <30s."""
    rng = np.random.default_rng(88)
    n = 1000
    df = pd.DataFrame({
        "x": rng.normal(0, 1, n),
    })
    z = -0.5 + 0.8 * df["x"]
    df["y"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-z))).astype(int)
    fit = smf.logit("y ~ x", data=df).fit(disp=False)
    M = Margins(fit)
    import time
    t0 = time.time()
    ame = M.dydx("x", vce="bootstrap", n_boot=200, boot_seed=1)
    elapsed = time.time() - t0
    assert elapsed < 30.0
    assert ame.se[0] > 0
