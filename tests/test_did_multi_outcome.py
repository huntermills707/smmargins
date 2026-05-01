"""
DiD on multi-outcome models (MNLogit, OrderedModel).

Covers shape invariants, K-sum-to-zero properties, hand-computed DiD
per outcome, cross-outcome contrasts via ``joint``, ``DiDResult.outcome``,
and a single-outcome regression test.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.testing import assert_allclose

from smmargins import Margins


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mnlogit_did_fit():
    """3-class MNLogit with binary treat and post for DiD testing."""
    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame({
        "treat": rng.integers(0, 2, n),
        "post": rng.integers(0, 2, n),
        "x": rng.normal(0, 1, n),
    })
    eta = np.column_stack([
        np.zeros(n),
        0.3 + 0.5 * df["treat"] - 0.2 * df["post"] + 0.4 * df["x"],
        -0.1 + 0.3 * df["treat"] + 0.1 * df["post"] - 0.2 * df["x"],
    ])
    probs = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    df["y"] = np.array([rng.choice([0, 1, 2], p=p) for p in probs])
    return sm.MNLogit(df["y"], sm.add_constant(df[["treat", "post", "x"]])).fit(disp=False)


# ── shape invariants ────────────────────────────────────────────────────────

def test_did_cells_shape(mnlogit_did_fit):
    """Cells must have shape (4*K,) with outcome index repeating 0..K-1."""
    M = Margins(mnlogit_did_fit)
    res = M.did("treat", "post")
    K = M.n_outcomes
    assert res.cells.estimate.shape == (4 * K,)
    assert res.cells.outcome_index is not None
    expected = np.tile(np.arange(K), 4)
    assert np.array_equal(res.cells.outcome_index, expected)


def test_did_simple_and_did_shapes(mnlogit_did_fit):
    """Simple effects (2*K), DiD (K), joint vcov (3*K, 3*K)."""
    M = Margins(mnlogit_did_fit)
    res = M.did("treat", "post")
    K = M.n_outcomes
    assert res.simple_effects.estimate.shape == (2 * K,)
    assert res.did.estimate.shape == (K,)
    assert res.joint.vcov.shape == (3 * K, 3 * K)


# ── sum-to-one invariants ───────────────────────────────────────────────────

def test_did_cells_sum_to_one_per_profile(mnlogit_did_fit):
    """Each block of K consecutive cells (same treat/post) sums to 1."""
    M = Margins(mnlogit_did_fit)
    res = M.did("treat", "post")
    K = M.n_outcomes
    for block in range(4):
        block_est = res.cells.estimate[block * K:(block + 1) * K]
        assert_allclose(block_est.sum(), 1.0, atol=1e-10)


def test_did_sums_to_zero_across_outcomes(mnlogit_did_fit):
    """DiD across all outcomes sums to ~0 (probabilities sum to 1)."""
    M = Margins(mnlogit_did_fit)
    res = M.did("treat", "post")
    assert_allclose(res.did.estimate.sum(), 0.0, atol=1e-10)


# ── hand-computed DiD per outcome ───────────────────────────────────────────

def test_did_hand_computed_per_outcome(mnlogit_did_fit):
    """DiD_k matches hand computation from design-matrix predictions."""
    M = Margins(mnlogit_did_fit)
    res = M.did("treat", "post")
    K = M.n_outcomes

    # Build the four design matrices and average predictions
    data = M.data.copy()
    cells = []
    for t in [0, 1]:
        for p in [0, 1]:
            d = data.copy()
            d["treat"] = t
            d["post"] = p
            X = M._build_exog(d)
            pred = M._predict(M.params, X)
            cells.append(pred.mean(axis=0))
    # cells order from predict: (0,0), (0,1), (1,0), (1,1)
    hand = cells[0] - cells[1] - cells[2] + cells[3]
    assert_allclose(res.did.estimate, hand, atol=1e-10)


# ── cross-outcome contrast via joint ────────────────────────────────────────

def test_did_cross_outcome_contrast(mnlogit_did_fit):
    """joint.contrast can compare DiD of outcome 1 vs outcome 0."""
    M = Margins(mnlogit_did_fit)
    res = M.did("treat", "post")
    K = M.n_outcomes
    if K < 2:
        pytest.skip("needs at least 2 outcomes")

    # DiD rows are the last K rows of joint
    # Compare DiD of outcome 1 minus DiD of outcome 0
    c = np.zeros(res.joint.estimate.size)
    c[2 * K + 1] = 1.0
    c[2 * K + 0] = -1.0
    diff = res.joint.contrast(c, labels=["DiD_1 - DiD_0"])

    hand_diff = res.did.estimate[1] - res.did.estimate[0]
    assert_allclose(diff.estimate[0], hand_diff, atol=1e-12)

    # SE via delta method
    hand_se = np.sqrt(c @ res.joint.vcov @ c)
    assert_allclose(diff.se[0], hand_se, atol=1e-12)


# ── DiDResult.outcome(k) slicer ────────────────────────────────────────────

def test_did_outcome_slicer(mnlogit_did_fit):
    """DiDResult.outcome(k) returns a single-outcome slice."""
    M = Margins(mnlogit_did_fit)
    res = M.did("treat", "post")
    K = M.n_outcomes

    for k in range(K):
        sub = res.outcome(k)
        assert sub.did.estimate.shape == (1,)
        assert_allclose(sub.did.estimate[0], res.did.estimate[k], atol=1e-12)
        assert sub.cells.estimate.shape == (4,)
        assert sub.simple_effects.estimate.shape == (2,)
        assert sub.joint.vcov.shape == (3, 3)
        # Vcov sub-block match
        did_slice = slice(2 * K + k, 2 * K + k + 1)
        assert_allclose(
            sub.did.vcov,
            res.joint.vcov[did_slice, did_slice],
            atol=1e-12,
        )


def test_did_outcome_slicer_error_on_single_outcome(did_frame):
    """outcome() on a single-outcome DiDResult raises ValueError."""
    ols = smf.ols("y_lin ~ treat + post + treat:post + x", data=did_frame).fit()
    res = Margins(ols).did("treat", "post")
    with pytest.raises(ValueError, match="only valid for multi-outcome"):
        res.outcome(0)


# ── single-outcome regression ───────────────────────────────────────────────

def test_ols_did_regression(did_frame):
    """Single-outcome OLS DiD is byte-identical to interaction coefficient."""
    ols = smf.ols("y_lin ~ treat + post + treat:post + x", data=did_frame).fit()
    M = Margins(ols)
    res = M.did("treat", "post")

    assert_allclose(res.did.estimate[0], ols.params["treat:post"], atol=1e-10)
    assert_allclose(res.did.se[0], ols.bse["treat:post"], rtol=1e-5)
    assert res.cells.estimate.shape == (4,)
    assert res.simple_effects.estimate.shape == (2,)
    assert res.joint.vcov.shape == (3, 3)
