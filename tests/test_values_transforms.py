"""Tests for generate-style transforms in values= (Pillar 1b)."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins, Expr


# ---------------------------------------------------------------------------
# Callable and Expr
# ---------------------------------------------------------------------------

def test_callable_scales_income(ols_fit, sim_frame):
    """Callable that scales income matches a hand-built frame."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": lambda df: df["x1"] * 1.1})
    manual = sim_frame.copy()
    manual["x1"] = manual["x1"] * 1.1
    fit = ols_fit.model.fit()
    pred_manual = fit.predict(manual).mean()
    assert_allclose(res.estimate[0], pred_manual, atol=1e-6)


def test_expr_equivalent_to_callable(ols_fit, sim_frame):
    """Expr and callable produce identical predictions."""
    M = Margins(ols_fit)
    r1 = M.predict(values={"x1": Expr("x1 * 1.1")})
    r2 = M.predict(values={"x1": lambda df: df["x1"] * 1.1})
    assert_allclose(r1.estimate, r2.estimate, atol=1e-10)


def test_callable_wrong_length_raises(ols_fit):
    """Callable returning wrong length raises ValueError."""
    M = Margins(ols_fit)
    with pytest.raises(ValueError, match="length"):
        M.predict(values={"x1": lambda df: np.array([1.0])})


def test_expr_unknown_column_raises(ols_fit):
    """Expr referencing unknown column raises (let df.eval raise)."""
    M = Margins(ols_fit)
    with pytest.raises(Exception):  # pd.eval or KeyError
        M.predict(values={"x1": Expr("nonexistent_col * 2")})


def test_composition_with_reducer(ols_fit, sim_frame):
    """values={'x1': 'mean', 'x2': Expr('x2 + x1')} evaluates x2 + mean(x1)."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": "mean", "x2": Expr("x2 + x1")})
    manual = sim_frame.copy()
    x1_mean = manual["x1"].mean()
    manual["x1"] = x1_mean
    manual["x2"] = manual["x2"] + x1_mean
    fit = ols_fit.model.fit()
    pred_manual = fit.predict(manual).mean()
    assert_allclose(res.estimate[0], pred_manual, atol=1e-6)


def test_pate_pattern_point_estimates(ols_fit):
    """PATE pattern: predict(treat=1) - predict(treat=0) gives sensible estimates."""
    M = Margins(ols_fit)
    r1 = M.predict(values={"dummy": 1})
    r0 = M.predict(values={"dummy": 0})
    diff = r1.estimate[0] - r0.estimate[0]
    # Not a strong assert — just check it's finite and non-NaN
    assert np.isfinite(diff)


def test_callable_composes_with_bootstrap(ols_fit):
    """Callable runs once per resample under bootstrap without error."""
    M = Margins(ols_fit)
    res = M.predict(
        values={"x1": lambda df: df["x1"] * 1.1},
        vce="bootstrap", n_boot=50, boot_seed=0,
    )
    assert len(res.estimate) == 1
    assert res.draws is not None


# ---------------------------------------------------------------------------
# Multi-output
# ---------------------------------------------------------------------------

def test_callable_expr_mnlogit(mnlogit_fit):
    """Callable / Expr work against MNLogit."""
    M = Margins(mnlogit_fit)
    r1 = M.predict(values={"x1": lambda df: df["x1"] + 1})
    r2 = M.predict(values={"x1": Expr("x1 + 1")})
    assert_allclose(r1.estimate, r2.estimate, atol=1e-10)
