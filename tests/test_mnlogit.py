"""Tests for multi-outcome support on MNLogit.

Covers parity with statsmodels get_margeff, hand checks, analytic-vs-FD
parity, discrete contrasts, and outcome subsetting.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

from smmargins import Margins


# ---------------------------------------------------------------------------
# Parity vs statsmodels get_margeff
# ---------------------------------------------------------------------------

def _non_const_index(exog_names, name):
    """Return the index of ``name`` among non-constant exog names."""
    filtered = [n for n in exog_names if n not in ("Intercept", "const")]
    return filtered.index(name)


@pytest.mark.parity
@pytest.mark.parametrize("outcome", [0, 1, 2])
def test_mnlogit_ame_vs_get_margeff(mnlogit_fit, outcome):
    """MNLogit AME agrees with get_margeff(at='overall') per outcome."""
    M = Margins(mnlogit_fit)
    ame = M.dydx("x1")
    mfx = mnlogit_fit.get_margeff(at="overall", method="dydx")
    x1_pos = _non_const_index(mnlogit_fit.model.exog_names, "x1")

    assert_allclose(
        ame.estimate[outcome], mfx.margeff[x1_pos, outcome], atol=1e-5
    )
    assert_allclose(
        ame.se[outcome], mfx.margeff_se[x1_pos, outcome], rtol=1e-3
    )


@pytest.mark.parity
@pytest.mark.parametrize("outcome", [0, 1, 2])
def test_mnlogit_mem_vs_get_margeff(mnlogit_fit, outcome):
    """MNLogit MEM agrees with get_margeff(at='mean') per outcome."""
    M = Margins(mnlogit_fit)
    mem = M.dydx("x1", at="mean")
    mfx = mnlogit_fit.get_margeff(at="mean", method="dydx")
    x1_pos = _non_const_index(mnlogit_fit.model.exog_names, "x1")

    assert_allclose(
        mem.estimate[outcome], mfx.margeff[x1_pos, outcome], atol=1e-5
    )
    assert_allclose(
        mem.se[outcome], mfx.margeff_se[x1_pos, outcome], rtol=1e-3
    )


# ---------------------------------------------------------------------------
# Hand checks
# ---------------------------------------------------------------------------

def test_mnlogit_aap_sums_to_one(mnlogit_fit):
    """K AAP values must sum to 1.0 (per-row probabilities sum to 1)."""
    M = Margins(mnlogit_fit)
    aap = M.predict()
    assert_allclose(aap.estimate.sum(), 1.0, atol=1e-10)


def test_mnlogit_discrete_contrast_hand_check(mnlogit_fit, mnlogit_frame):
    """Discrete contrast on a binary covariate matches two-row predict diff."""
    # Create a binary variable in the frame for the discrete contrast test
    d = mnlogit_frame.copy()
    d["d_bin"] = (d["x1"] > d["x1"].median()).astype(int)
    fit = sm.MNLogit(
        d["y_mnl"], sm.add_constant(d[["x1", "x2", "d_bin"]])
    ).fit(disp=False)
    M = Margins(fit)
    disc = M.dydx("d_bin")

    # Hand-compute: average prediction at d_bin=1 minus d_bin=0
    d1 = d.copy(); d1["d_bin"] = 1
    d0 = d.copy(); d0["d_bin"] = 0
    # Build exog manually because sm.add_constant skips when a column is already
    # constant (d_bin==1 or d_bin==0 for all rows).
    N = len(d1)
    X1 = np.column_stack([np.ones(N), d1["x1"], d1["x2"], d1["d_bin"]])
    X0 = np.column_stack([np.ones(N), d0["x1"], d0["x2"], d0["d_bin"]])
    probs1 = fit.model.predict(fit.params, X1)
    probs0 = fit.model.predict(fit.params, X0)
    hand = probs1.mean(axis=0) - probs0.mean(axis=0)

    assert_allclose(disc.estimate, hand, atol=1e-8)


# ---------------------------------------------------------------------------
# Analytic vs FD parity
# ---------------------------------------------------------------------------

def test_mnlogit_analytic_vs_fd_ame(mnlogit_fit):
    """Analytic and FD AME agree on MNLogit."""
    M_an = Margins(mnlogit_fit, analytic=True)
    M_fd = Margins(mnlogit_fit, analytic=False)
    r_an = M_an.dydx("x1")
    r_fd = M_fd.dydx("x1")
    assert_allclose(r_an.estimate, r_fd.estimate, atol=1e-5)
    assert_allclose(r_an.se, r_fd.se, rtol=1e-3)


def test_mnlogit_analytic_vs_fd_predict(mnlogit_fit):
    """Analytic and FD predict agree on MNLogit."""
    M_an = Margins(mnlogit_fit, analytic=True)
    M_fd = Margins(mnlogit_fit, analytic=False)
    r_an = M_an.predict()
    r_fd = M_fd.predict()
    assert_allclose(r_an.estimate, r_fd.estimate, atol=1e-5)
    assert_allclose(r_an.se, r_fd.se, atol=1e-5)


# ---------------------------------------------------------------------------
# outcome= subsetting
# ---------------------------------------------------------------------------

def test_mnlogit_predict_outcome_subset(mnlogit_fit):
    """outcome= slices the full predict result correctly."""
    M = Margins(mnlogit_fit)
    full = M.predict()
    sub = M.predict(outcome=1)

    assert_allclose(sub.estimate, full.estimate[1:2], atol=1e-12)
    assert_allclose(sub.vcov, full.vcov[np.ix_([1], [1])], atol=1e-12)


def test_mnlogit_predict_outcome_multi_subset(mnlogit_fit):
    """outcome= with a list slices correctly and preserves joint vcov."""
    M = Margins(mnlogit_fit)
    full = M.predict()
    sub = M.predict(outcome=[0, 2])

    assert_allclose(sub.estimate, full.estimate[[0, 2]], atol=1e-12)
    assert_allclose(
        sub.vcov, full.vcov[np.ix_([0, 2], [0, 2])], atol=1e-12
    )


def test_mnlogit_dydx_outcome_subset(mnlogit_fit):
    """outcome= on dydx slices to the requested outcome."""
    M = Margins(mnlogit_fit)
    full = M.dydx("x1")
    sub = M.dydx("x1", outcome=2)

    assert_allclose(sub.estimate, full.estimate[2:3], atol=1e-12)


# ---------------------------------------------------------------------------
# MarginsResult.outcome() helper
# ---------------------------------------------------------------------------

def test_mnlogit_outcome_helper_by_index(mnlogit_fit):
    M = Margins(mnlogit_fit)
    full = M.predict()
    sub = full.outcome(1)
    assert_allclose(sub.estimate, full.estimate[1:2])


def test_mnlogit_outcome_helper_by_label(mnlogit_fit):
    M = Margins(mnlogit_fit)
    full = M.predict()
    sub = full.outcome("1")
    assert_allclose(sub.estimate, full.estimate[1:2])


# ---------------------------------------------------------------------------
# Scope guards
