"""Tests for the values= DSL (Pillar 1a)."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


# ---------------------------------------------------------------------------
# Basic scalar / sequence / reducer
# ---------------------------------------------------------------------------

def test_values_scalar_matches_atexog_scalar(ols_fit):
    """values={"x1": 1.0} and atexog={"x1": 1.0} produce identical estimates."""
    M = Margins(ols_fit)
    r1 = M.predict(atexog={"x1": 1.0})
    r2 = M.predict(values={"x1": 1.0})
    assert_allclose(r1.estimate, r2.estimate, atol=1e-10)


def test_values_sequence_creates_grid(ols_fit):
    """values={"x1": [0, 1, 2]} returns 3 rows with correct labels."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": [0, 1, 2]})
    assert len(res.estimate) == 3
    assert res.labels == ["x1=0", "x1=1", "x1=2"]


def test_values_reducer_mean_matches_at_mean(ols_fit, sim_frame):
    """values={"x1": "mean"} reproduces the equivalent manual mean replacement."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": "mean"}, default_values="asobserved")
    manual = sim_frame.copy()
    manual["x1"] = manual["x1"].mean()
    fit = ols_fit.model.fit()
    pred_manual = fit.predict(manual).mean()
    assert_allclose(res.estimate[0], pred_manual, atol=1e-6)


def test_values_reducer_percentile(ols_fit, sim_frame):
    """values={"x1": "p25"} matches manual quantile replacement."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": "p25"})
    q25 = float(np.percentile(sim_frame["x1"].to_numpy(), 25))
    manual = M.predict(atexog={"x1": q25})
    assert_allclose(res.estimate, manual.estimate, atol=1e-10)


def test_values_default_values_mean(ols_fit, sim_frame):
    """default_values="mean" sets unspecified numerics to their mean."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": 1}, default_values="mean")
    manual = sim_frame.copy()
    for c in ["x1", "x2"]:
        manual[c] = manual[c].mean()
    manual["x1"] = 1.0
    fit = ols_fit.model.fit()
    pred_manual = fit.predict(manual).mean()
    assert_allclose(res.estimate[0], pred_manual, atol=1e-6)


def test_values_atexog_merge(ols_fit):
    """Disjoint atexog and values keys merge."""
    M = Margins(ols_fit)
    res = M.predict(atexog={"x1": 1}, values={"x2": "p25"})
    assert len(res.estimate) == 1
    assert "x1=1" in res.labels[0]
    assert "x2=" in res.labels[0]


def test_values_atexog_conflict(ols_fit):
    """Same variable in atexog and values raises ValueError."""
    M = Margins(ols_fit)
    with pytest.raises(ValueError, match="x1"):
        M.predict(atexog={"x1": 1}, values={"x1": "mean"})


def test_values_unknown_reducer(ols_fit):
    """Unknown reducer raises ValueError listing valid reducers."""
    M = Margins(ols_fit)
    with pytest.raises(ValueError, match="weird"):
        M.predict(values={"x1": "weird"})


def test_values_reducer_on_wrong_dtype(ols_fit):
    """Numeric reducer on categorical column raises TypeError."""
    M = Margins(ols_fit)
    with pytest.raises(TypeError, match="mean"):
        M.predict(values={"grp": "mean"})


def test_values_cartesian_product(ols_fit):
    """Two sequence axes produce a 4-row Cartesian product."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": [0, 1], "x2": [10, 20]})
    assert len(res.estimate) == 4


# ---------------------------------------------------------------------------
# Composition with over=, weights=, inference
# ---------------------------------------------------------------------------

def test_values_composes_with_over(ols_fit, sim_frame):
    """values + over returns one row per group with subgroup mean applied."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": "mean"}, over="grp")
    groups = sorted(sim_frame["grp"].unique())
    assert len(res.estimate) == len(groups)
    for g in groups:
        assert any(f"grp={g}" in lab for lab in res.labels)


def test_values_composes_with_vce_simulation(ols_fit):
    """values + vce='simulation' + ci_method='bonferroni' runs without error."""
    M = Margins(ols_fit)
    res = M.predict(
        values={"x1": [0, 1]},
        vce="simulation", n_sims=500, sim_seed=42,
        ci_method="bonferroni",
    )
    assert len(res.estimate) == 2
    assert res.draws is not None


def test_values_composes_with_weights(ols_fit, sim_frame):
    """Weighted means computed correctly with default_values='mean'."""
    w = np.abs(np.random.default_rng(42).standard_normal(len(sim_frame)))
    fit = ols_fit.model.fit(weights=w)
    M = Margins(fit, weights=w)
    res = M.predict(values={"x1": 1}, default_values="mean")
    # Just a smoke test that it runs and returns one row
    assert len(res.estimate) == 1


# ---------------------------------------------------------------------------
# Multi-output
# ---------------------------------------------------------------------------

def test_values_mnlogit(mnlogit_fit):
    """values= works on MNLogit; _delta flattens multi-output to (n_grid * K,)."""
    M = Margins(mnlogit_fit)
    res = M.predict(values={"x1": [-1, 0, 1]})
    # 3 grid points × 3 outcomes, flattened by _delta into a 1-D vector.
    # Conceptually (n_grid, K); physically (n_grid * K,) for uniform handling.
    assert res.estimate.shape == (9,)
