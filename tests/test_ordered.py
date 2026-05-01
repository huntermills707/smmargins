"""Tests for multi-outcome support on OrderedModel (FD-only path).

Covers basic predictions, marginal effects, and hand checks.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

def test_ordered_aap_sums_to_one(ordered_fit):
    """K AAP values must sum to 1.0 for OrderedModel."""
    M = Margins(ordered_fit)
    aap = M.predict()
    assert_allclose(aap.estimate.sum(), 1.0, atol=1e-10)


def test_ordered_ame_returns_k_estimates(ordered_fit):
    """AME on OrderedModel returns one estimate per category."""
    M = Margins(ordered_fit)
    ame = M.dydx("x1")
    assert len(ame.estimate) == 4
    assert len(ame.se) == 4


def test_ordered_mem_returns_k_estimates(ordered_fit):
    """MEM on OrderedModel returns one estimate per category."""
    M = Margins(ordered_fit)
    mem = M.dydx("x1", at="mean")
    assert len(mem.estimate) == 4


def test_ordered_predict_outcome_subset(ordered_fit):
    """outcome= slices the full predict result correctly."""
    M = Margins(ordered_fit)
    full = M.predict()
    sub = M.predict(outcome=2)
    assert_allclose(sub.estimate, full.estimate[2:3], atol=1e-12)


def test_ordered_dydx_outcome_subset(ordered_fit):
    """outcome= on dydx slices to the requested outcome."""
    M = Margins(ordered_fit)
    full = M.dydx("x1")
    sub = M.dydx("x1", outcome=[1, 3])
    assert_allclose(sub.estimate, full.estimate[[1, 3]], atol=1e-12)


# ---------------------------------------------------------------------------
# Scope guards
# ---------------------------------------------------------------------------

def test_ordered_did_smoke(ordered_fit):
    """DiD on OrderedModel returns correct shapes and sums to zero."""
    res = Margins(ordered_fit).did("x1", "x2")
    K = 4
    assert res.cells.estimate.shape == (4 * K,)
    assert res.simple_effects.estimate.shape == (2 * K,)
    assert res.did.estimate.shape == (K,)
    assert res.joint.vcov.shape == (3 * K, 3 * K)
    # Probabilities sum to 1, so DiD across outcomes sums to ~0
    assert_allclose(res.did.estimate.sum(), 0.0, atol=1e-10)


def test_ordered_eyex_smoke(ordered_fit):
    """Elasticity on OrderedModel returns finite (K,) shape."""
    res = Margins(ordered_fit).dydx("x1", method="eyex")
    K = 4
    assert res.estimate.shape == (K,)
    assert np.isfinite(res.estimate).all()
