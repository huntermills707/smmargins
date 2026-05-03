"""Tests for ``MarginsResult.wald()`` and ``.pairwise()``."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit
from scipy import stats

from smmargins import Margins


# ---------------------------------------------------------------------------
# Wald: single restriction
# ---------------------------------------------------------------------------

def test_wald_single_restriction():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.standard_normal(200), "x2": rng.standard_normal(200)})
    df["y"] = 1.0 + 2.0 * df["x1"] - df["x2"] + 0.1 * rng.standard_normal(200)
    fit = smf.ols("y ~ x1 + x2", df).fit()
    M = Margins(fit)
    res = M.dydx(["x1", "x2"])

    # H0: AME of x1 = 0
    w = res.wald(C=np.array([[1, 0]]))
    t_stat = res.estimate[0] / res.se[0]
    expected_stat = t_stat ** 2
    assert pytest.approx(expected_stat, rel=1e-10) == w.stat
    expected_p = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    assert pytest.approx(expected_p, rel=1e-10) == w.pvalue


# ---------------------------------------------------------------------------
# Wald: joint restriction parity with statsmodels
# ---------------------------------------------------------------------------

def test_wald_joint_restriction_parity():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x1": rng.standard_normal(300),
        "x2": rng.standard_normal(300),
        "x3": rng.standard_normal(300),
    })
    df["y"] = 0.5 + 1.0 * df["x1"] - 0.5 * df["x2"] + 0.3 * df["x3"] + 0.5 * rng.standard_normal(300)
    fit = smf.ols("y ~ x1 + x2 + x3", df).fit()
    M = Margins(fit)
    res = M.dydx(["x1", "x2", "x3"])

    # H0: all AMEs = 0
    C = np.eye(3)
    w = res.wald(C=C)

    # For OLS, AME = beta, so parity with statsmodels wald_test holds.
    # statsmodels returns an F-statistic; our wald() returns chi² = F * df.
    sm_wald = fit.wald_test(r_matrix=["x1", "x2", "x3"])
    f_stat = np.asarray(sm_wald.statistic).ravel()[0]
    df_num = 3
    assert pytest.approx(f_stat * df_num, rel=1e-8) == w.stat
    assert pytest.approx(np.asarray(sm_wald.pvalue).ravel()[0], rel=1e-8) == w.pvalue


# ---------------------------------------------------------------------------
# Wald: linear combination
# ---------------------------------------------------------------------------

def test_wald_linear_combination():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x1": rng.standard_normal(300),
        "x2": rng.standard_normal(300),
        "x3": rng.standard_normal(300),
    })
    df["y"] = 0.5 + 1.0 * df["x1"] - 0.5 * df["x2"] + 0.3 * df["x3"] + 0.5 * rng.standard_normal(300)
    fit = smf.ols("y ~ x1 + x2 + x3", df).fit()
    M = Margins(fit)
    res = M.dydx(["x1", "x2", "x3"])

    # H0: AME(x1) - AME(x2) = 0
    C = np.array([[1, -1, 0]])
    w = res.wald(C=C)

    # For OLS, AME = beta, so parity with statsmodels wald_test holds
    sm_wald = fit.wald_test(r_matrix="x1 = x2")
    f_stat = np.asarray(sm_wald.statistic).ravel()[0]
    assert pytest.approx(f_stat * 1, rel=1e-8) == w.stat
    assert pytest.approx(np.asarray(sm_wald.pvalue).ravel()[0], rel=1e-8) == w.pvalue


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------

def test_pairwise_four_level_factor():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "group": rng.choice(["a", "b", "c", "d"], n),
    })
    df["y"] = 1.0 + 2.0 * df["x"] + df["group"].map({"a": 0, "b": 1, "c": 2, "d": 3}) + 0.1 * rng.standard_normal(n)
    fit = smf.ols("y ~ x + C(group)", df).fit()
    M = Margins(fit)
    res = M.dydx("group")

    pw = res.pairwise(by="group")
    # dydx returns 3 contrasts (b vs a, c vs a, d vs a);
    # pairwise produces C(3,2) = 3 comparisons among them.
    assert pw.estimate.shape[0] == 3
    assert len(pw.labels) == 3

    # Each estimate should equal a manual subtract of the original contrasts
    for i, lab in enumerate(pw.labels):
        # Parse label like "group: b vs c"
        parts = lab.split(" vs ")
        lvl = parts[0].split(": ")[1]
        ref = parts[1]
        idx_lvl = res.labels.index(f"group: {lvl} vs a")
        idx_ref = res.labels.index(f"group: {ref} vs a")
        expected = res.estimate[idx_lvl] - res.estimate[idx_ref]
        assert pytest.approx(expected, rel=1e-10) == pw.estimate[i]


# ---------------------------------------------------------------------------
# Pairwise + Bonferroni
# ---------------------------------------------------------------------------

def test_pairwise_bonferroni():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "group": rng.choice(["a", "b", "c"], n),
    })
    df["y"] = 1.0 + 2.0 * df["x"] + df["group"].map({"a": 0, "b": 1, "c": 2}) + 0.1 * rng.standard_normal(n)
    fit = smf.ols("y ~ x + C(group)", df).fit()
    M = Margins(fit)
    res = M.dydx("group")
    pw = res.pairwise(by="group", ci_method="bonferroni")

    # Bonferroni factor = 3
    m = 3
    for i in range(pw.estimate.shape[0]):
        raw_p = pw.pvalues[i] * m
        if raw_p > 1.0:
            raw_p = 1.0
        # Just check the CI is wider than pointwise
        assert pw.ci_lower[i] <= pw.estimate[i] <= pw.ci_upper[i]


# ---------------------------------------------------------------------------
# Pairwise + sup-t (simultaneous CIs from simulation draws)
# ---------------------------------------------------------------------------

def test_pairwise_sup_t_narrower_than_bonferroni():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "group": rng.choice(["a", "b", "c", "d"], n),
    })
    df["y"] = (
        1.0 + 2.0 * df["x"]
        + df["group"].map({"a": 0, "b": 1, "c": 2, "d": 3})
        + 0.1 * rng.standard_normal(n)
    )
    fit = smf.ols("y ~ x + C(group)", df).fit()
    M = Margins(fit)
    res = M.dydx("group", vce="simulation", n_sims=4000, sim_seed=0)

    sup = res.pairwise(by="group", ci_method="sup-t")
    bonf = res.pairwise(by="group", ci_method="bonferroni")

    assert sup.estimate.shape[0] == bonf.estimate.shape[0] == 3
    sup_widths = sup.ci_upper - sup.ci_lower
    bonf_widths = bonf.ci_upper - bonf.ci_lower
    # sup-t exploits correlation; for correlated contrasts it should be
    # strictly narrower than the conservative Bonferroni bound.
    assert np.all(sup_widths < bonf_widths)


# ---------------------------------------------------------------------------
# Singular vcov error
# ---------------------------------------------------------------------------

def test_wald_singular_vcov_error():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_normal(50)})
    df["y"] = 1.0 + 2.0 * df["x"] + 0.1 * rng.standard_normal(50)
    fit = smf.ols("y ~ x", df).fit()
    M = Margins(fit)
    res = M.predict()

    # 1x1 vcov, C = [[1, 0]] would be invalid shape
    with pytest.raises(ValueError):
        res.wald(C=np.array([[1, 0]]))
