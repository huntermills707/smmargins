"""
Tests for adjusted predictions: AAP, APM, APR.

Covers prediction at different evaluation profiles (overall, mean, median,
zero) and at representative values via atexog, across OLS, Logit, and
Poisson models.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


# ---------------------------------------------------------------------------
# OLS predictions
# ---------------------------------------------------------------------------

def test_ols_apr_at_representative_values(ols_fit, sim_frame):
    """APR first row at x1=-1 matches hand-computed OLS prediction."""
    M = Margins(ols_fit, use_t=True)
    apr = M.predict(atexog={"x1": [-1.0, 0.0, 1.0]})

    b = ols_fit.params
    x1v = -1.0
    pred_hand = (
        b["Intercept"]
        + b["x1"] * x1v
        + b["I(x1 ** 2)"] * x1v ** 2
        + b["x2"] * sim_frame["x2"].mean()
        + b["x1:x2"] * x1v * sim_frame["x2"].mean()
    )
    assert_allclose(apr.estimate[0], pred_hand, atol=1e-7)


def test_ols_apm(ols_fit):
    """APM returns a single estimate labelled 'APM'."""
    M = Margins(ols_fit)
    apm = M.predict(at="mean")
    assert apm.labels == ["APM"]
    assert len(apm.estimate) == 1


def test_ols_aap_label(ols_fit):
    """Default predict() returns a single estimate labelled 'AAP'."""
    M = Margins(ols_fit)
    aap = M.predict()
    assert aap.labels == ["AAP"]


def test_ols_predict_at_median(ols_fit):
    """predict(at='median') returns a single estimate."""
    M = Margins(ols_fit)
    res = M.predict(at="median")
    assert len(res.estimate) == 1
    assert res.labels == ["AP @ median"]


def test_ols_predict_at_zero(ols_fit):
    """predict(at='zero') returns a single estimate."""
    M = Margins(ols_fit)
    res = M.predict(at="zero")
    assert len(res.estimate) == 1
    assert res.labels == ["AP @ zero"]


def test_ols_apr_with_others_at_mean(ols_fit):
    """APR with atexog and at='mean' produces correct row count and labels."""
    M = Margins(ols_fit)
    apr = M.predict(at="mean", atexog={"x1": [-1.0, 0.0, 1.0]})
    assert len(apr.estimate) == 3
    assert apr.labels == ["x1=-1.0", "x1=0.0", "x1=1.0"]


def test_ols_apr_all_three_rows(ols_fit, sim_frame):
    """All three APR rows match hand-computed OLS predictions."""
    M = Margins(ols_fit, use_t=True)
    apr = M.predict(atexog={"x1": [-1.0, 0.0, 1.0]})

    b = ols_fit.params
    for idx, x1v in enumerate([-1.0, 0.0, 1.0]):
        pred_hand = (
            b["Intercept"]
            + b["x1"] * x1v
            + b["I(x1 ** 2)"] * x1v ** 2
            + b["x2"] * sim_frame["x2"].mean()
            + b["x1:x2"] * x1v * sim_frame["x2"].mean()
        )
        assert_allclose(apr.estimate[idx], pred_hand, atol=1e-7)


# ---------------------------------------------------------------------------
# Poisson predictions
# ---------------------------------------------------------------------------

def test_poisson_aap_equals_sample_mean(poisson_fit, sim_frame):
    """For canonical-link Poisson, AAP equals the sample mean of count."""
    MP = Margins(poisson_fit)
    aap = MP.predict()
    assert_allclose(aap.estimate[0], sim_frame["count"].mean(), atol=1e-10)


def test_poisson_aap_hand_gradient(poisson_fit):
    """Poisson AAP SE matches hand-derived delta-method gradient."""
    MP = Margins(poisson_fit)
    aap = MP.predict()
    X = poisson_fit.model.exog
    eta = X @ poisson_fit.params
    mu = np.exp(eta)
    g = (mu[:, None] * X).mean(axis=0)
    V = np.asarray(poisson_fit.cov_params())
    hand_se = np.sqrt(g @ V @ g)
    assert_allclose(aap.se[0], hand_se, rtol=1e-5)


def test_poisson_apm(poisson_fit):
    """Poisson APM returns a single estimate labelled 'APM'."""
    MP = Margins(poisson_fit)
    apm = MP.predict(at="mean")
    assert apm.labels == ["APM"]
    assert len(apm.estimate) == 1


def test_poisson_predict_at_median(poisson_fit):
    """Poisson predict(at='median') returns a single estimate."""
    MP = Margins(poisson_fit)
    res = MP.predict(at="median")
    assert len(res.estimate) == 1


def test_poisson_predict_at_zero(poisson_fit):
    """Poisson predict(at='zero') returns a single estimate."""
    MP = Margins(poisson_fit)
    res = MP.predict(at="zero")
    assert len(res.estimate) == 1


# ---------------------------------------------------------------------------
# Logit predictions
# ---------------------------------------------------------------------------

def test_logit_aap(logit_fit):
    """Logit AAP returns a single estimate between 0 and 1."""
    ML = Margins(logit_fit)
    aap = ML.predict()
    assert aap.labels == ["AAP"]
    assert len(aap.estimate) == 1
    assert 0.0 < aap.estimate[0] < 1.0


def test_logit_apm(logit_fit):
    """Logit APM returns a single estimate labelled 'APM'."""
    ML = Margins(logit_fit)
    apm = ML.predict(at="mean")
    assert apm.labels == ["APM"]
    assert len(apm.estimate) == 1


def test_logit_apr_at_representative_values(logit_fit):
    """Logit APR produces correct row count and bounded probabilities."""
    ML = Margins(logit_fit)
    apr = ML.predict(atexog={"x1": [-1.0, 0.0, 1.0]})
    assert len(apr.estimate) == 3
    assert apr.labels == ["x1=-1.0", "x1=0.0", "x1=1.0"]
    for p in apr.estimate:
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# atexog cartesian-product expansion
# ---------------------------------------------------------------------------

def test_ols_apr_cartesian_two_vars(ols_fit):
    """atexog with two variables expands as a cartesian product."""
    M = Margins(ols_fit)
    apr = M.predict(atexog={"x1": [0.0, 1.0], "x2": [-1.0, 1.0]})
    assert len(apr.estimate) == 4


# ---------------------------------------------------------------------------
# factor_stat modes
# ---------------------------------------------------------------------------

def test_logit_predict_factor_stat_mode(logit_fit):
    """predict(at='mean', factor_stat='mode') uses modal factor level."""
    ML = Margins(logit_fit)
    res = ML.predict(at="mean", factor_stat="mode")
    assert len(res.estimate) == 1
