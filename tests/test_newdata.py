"""Tests for the newdata= escape hatch (Pillar 1c)."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def test_newdata_predict_handbuilt(ols_fit):
    """Hand-built two-row frame predict matches hand-computed Xβ."""
    M = Margins(ols_fit)
    df = pd.DataFrame({
        "x1": [0.0, 1.0],
        "x2": [0.0, 0.0],
    })
    res = M.predict(newdata=df)
    b = ols_fit.params
    pred0 = b["Intercept"] + b["x1"] * 0.0 + b["I(x1 ** 2)"] * 0.0 + b["x2"] * 0.0 + b["x1:x2"] * 0.0
    pred1 = b["Intercept"] + b["x1"] * 1.0 + b["I(x1 ** 2)"] * 1.0 + b["x2"] * 0.0 + b["x1:x2"] * 0.0
    assert_allclose(res.estimate[0], (pred0 + pred1) / 2, atol=1e-7)


def test_newdata_dydx(ols_fit):
    """dydx on newdata matches hand-computed f'(Xβ)·β_x1 averaged."""
    M = Margins(ols_fit)
    df = pd.DataFrame({
        "x1": [0.0, 1.0],
        "x2": [0.0, 0.0],
    })
    res = M.dydx("x1", newdata=df)
    # Model is y ~ x1 + I(x1**2) + x2 + x1:x2
    # dy/dx1 = beta_x1 + 2*beta_x1sq*x1 + beta_x1x2*x2
    b = ols_fit.params
    d0 = b["x1"] + 2 * b["I(x1 ** 2)"] * 0.0 + b["x1:x2"] * 0.0
    d1 = b["x1"] + 2 * b["I(x1 ** 2)"] * 1.0 + b["x1:x2"] * 0.0
    assert_allclose(res.estimate[0], (d0 + d1) / 2, atol=1e-7)


# ---------------------------------------------------------------------------
# Schema mismatch
# ---------------------------------------------------------------------------

def test_newdata_schema_mismatch_formula(ols_fit):
    """Missing column in newdata for formula fit raises ValueError."""
    M = Margins(ols_fit)
    bad = pd.DataFrame({"x1": [0.0]})
    with pytest.raises(ValueError, match="missing"):
        M.predict(newdata=bad)


def test_newdata_schema_mismatch_raw():
    """Missing column in newdata for raw-mode fit raises ValueError."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x1": rng.standard_normal(50),
        "x2": rng.standard_normal(50),
    })
    df["y"] = 1.0 + 2.0 * df["x1"] + 0.5 * df["x2"] + rng.standard_normal(50)
    import statsmodels.api as sm
    fit = sm.OLS(df["y"], sm.add_constant(df[["x1", "x2"]])).fit()
    M = Margins(fit, data=df)
    bad = pd.DataFrame({"x1": [0.0]})
    with pytest.raises(ValueError, match="missing"):
        M.predict(newdata=bad)


# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------

def test_newdata_mutual_exclusion_at(ols_fit):
    """newdata + at raises ValueError."""
    M = Margins(ols_fit)
    with pytest.raises(ValueError, match="mutually exclusive"):
        M.predict(newdata=pd.DataFrame({"x1": [0.0], "x2": [0.0]}), at="mean")


def test_newdata_mutual_exclusion_values(ols_fit):
    """newdata + values raises ValueError."""
    M = Margins(ols_fit)
    with pytest.raises(ValueError, match="mutually exclusive"):
        M.predict(newdata=pd.DataFrame({"x1": [0.0], "x2": [0.0]}), values={"x1": 1})


# ---------------------------------------------------------------------------
# Inference composition
# ---------------------------------------------------------------------------

def test_newdata_composes_with_hc1(ols_fit):
    """newdata + cov_type='HC1' produces sensible SEs."""
    M = Margins(ols_fit)
    df = pd.DataFrame({
        "x1": [0.0, 1.0],
        "x2": [0.0, 0.0],
    })
    res = M.predict(newdata=df, cov_type="HC1")
    assert np.all(res.se >= 0)


def test_newdata_composes_with_simulation(ols_fit):
    """newdata + vce='simulation' produces sensible SEs."""
    M = Margins(ols_fit)
    df = pd.DataFrame({
        "x1": [0.0, 1.0],
        "x2": [0.0, 0.0],
    })
    res = M.predict(newdata=df, vce="simulation", n_sims=500, sim_seed=0)
    assert np.all(res.se >= 0)
    assert res.draws is not None


def test_newdata_bootstrap_does_not_resample_newdata(ols_fit, sim_frame):
    """Bootstrap refits on training data, not newdata."""
    M = Margins(ols_fit)
    tiny = pd.DataFrame({
        "x1": [0.0],
        "x2": [0.0],
    })
    # This should NOT fail due to tiny sample — bootstrap resamples training data
    res = M.predict(newdata=tiny, vce="bootstrap", n_boot=20, boot_seed=0)
    assert len(res.estimate) == 1


# ---------------------------------------------------------------------------
# Multi-output
# ---------------------------------------------------------------------------

def test_newdata_mnlogit(mnlogit_fit):
    """newdata works on MNLogit, returns (K,) shape (averaged over rows)."""
    M = Margins(mnlogit_fit)
    df = pd.DataFrame({
        "const": [1.0, 1.0],
        "x1": [0.0, 1.0],
        "x2": [0.0, 0.0],
    })
    res = M.predict(newdata=df)
    assert res.estimate.shape == (3,)  # 1 average * 3 outcomes


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_newdata_empty_raises(ols_fit):
    """Zero-row newdata raises ValueError."""
    M = Margins(ols_fit)
    empty = pd.DataFrame({"x1": pd.Series([], dtype=float), "x2": pd.Series([], dtype=float)})
    with pytest.raises(ValueError, match="at least one row"):
        M.predict(newdata=empty)


def test_newdata_single_row(ols_fit):
    """Single-row newdata works."""
    M = Margins(ols_fit)
    df = pd.DataFrame({"x1": [0.5], "x2": [-0.5]})
    res = M.predict(newdata=df)
    assert len(res.estimate) == 1
