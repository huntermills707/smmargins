"""Tests for Margins.contrast() (Pillar 1d)."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


# ---------------------------------------------------------------------------
# Point estimates
# ---------------------------------------------------------------------------

def test_contrast_pate_point_estimate(ols_fit):
    """contrast(a={"dummy": 1}, b={"dummy": 0}) estimate[2] equals diff of predicts."""
    M = Margins(ols_fit)
    joint = M.contrast(a={"dummy": 1}, b={"dummy": 0})
    r1 = M.predict(values={"dummy": 1})
    r0 = M.predict(values={"dummy": 0})
    expected_diff = r1.estimate[0] - r0.estimate[0]
    assert_allclose(joint.estimate[2], expected_diff, atol=1e-10)


# ---------------------------------------------------------------------------
# Joint SE
# ---------------------------------------------------------------------------

def test_contrast_joint_se_differs_from_naive(ols_fit):
    """Joint contrast SE != sqrt(SE_A^2 + SE_B^2) when there's covariance."""
    M = Margins(ols_fit)
    joint = M.contrast(a={"dummy": 1}, b={"dummy": 0})
    se_a = joint.se[0]
    se_b = joint.se[1]
    se_diff = joint.se[2]
    naive_se = np.sqrt(se_a ** 2 + se_b ** 2)
    # They should differ because A and B share the same beta uncertainty
    assert not np.isclose(se_diff, naive_se, atol=1e-10)


def test_contrast_joint_se_matches_manual_delta_ols(ols_fit):
    """Manual delta-method Jacobian for OLS contrast matches Margins.contrast."""
    M = Margins(ols_fit)
    joint = M.contrast(a={"dummy": 1}, b={"dummy": 0})

    # Build frames manually
    df_a = ols_fit.model.data.frame.copy()
    df_a["dummy"] = 1
    df_b = ols_fit.model.data.frame.copy()
    df_b["dummy"] = 0
    X_a = M._design.build_exog(df_a)
    X_b = M._design.build_exog(df_b)

    beta = np.asarray(ols_fit.params, dtype=float)
    pred_a = X_a.mean(axis=0) @ beta
    pred_b = X_b.mean(axis=0) @ beta

    J_a = X_a.mean(axis=0).reshape(1, -1)
    J_b = X_b.mean(axis=0).reshape(1, -1)
    J_diff = J_a - J_b
    J = np.vstack([J_a, J_b, J_diff])
    V = J @ M.cov @ J.T

    assert_allclose(joint.estimate, [pred_a, pred_b, pred_a - pred_b], atol=1e-7)
    assert_allclose(joint.vcov, V, atol=1e-7)


# ---------------------------------------------------------------------------
# newdata arms
# ---------------------------------------------------------------------------

def test_contrast_a_newdata_b_newdata(ols_fit):
    """Both arms as newdata frames."""
    M = Margins(ols_fit)
    df_a = pd.DataFrame({"x1": [0.0], "x2": [0.0]})
    df_b = pd.DataFrame({"x1": [1.0], "x2": [0.0]})
    joint = M.contrast(a_newdata=df_a, b_newdata=df_b)
    assert len(joint.estimate) == 3


def test_contrast_mixed_dsl_newdata(ols_fit):
    """a via DSL, b via newdata."""
    M = Margins(ols_fit)
    df_b = pd.DataFrame({"x1": [1.0], "x2": [0.0]})
    joint = M.contrast(a={"x1": 0}, b_newdata=df_b)
    assert len(joint.estimate) == 3


# ---------------------------------------------------------------------------
# Inference composition
# ---------------------------------------------------------------------------

def test_contrast_composes_with_bootstrap(ols_fit):
    """Bootstrap draws used for both arms simultaneously."""
    M = Margins(ols_fit)
    joint = M.contrast(
        a={"dummy": 1}, b={"dummy": 0},
        vce="bootstrap", n_boot=50, boot_seed=0,
    )
    assert joint.draws is not None
    # Draws matrix shape: (n_boot, 3)
    assert joint.draws.shape == (50, 3)


def test_contrast_composes_with_sup_t(ols_fit):
    """ci_method='sup-t' requires draw-based VCE; works with simulation."""
    M = Margins(ols_fit)
    joint = M.contrast(
        a={"dummy": 1}, b={"dummy": 0},
        vce="simulation", n_sims=500, sim_seed=0,
        ci_method="sup-t",
    )
    assert len(joint.estimate) == 3


# ---------------------------------------------------------------------------
# Multi-output
# ---------------------------------------------------------------------------

def test_contrast_mnlogit(mnlogit_fit):
    """Contrast against MNLogit returns the right shape."""
    M = Margins(mnlogit_fit)
    joint = M.contrast(a={"x1": 1}, b={"x1": 0})
    # 3 blocks (A, B, A-B) * 3 outcomes = 9 rows
    assert len(joint.estimate) == 9


# ---------------------------------------------------------------------------
# over=
# ---------------------------------------------------------------------------

def test_contrast_with_over(ols_fit):
    """Contrast within subgroups; result has n_groups * 3 rows."""
    M = Margins(ols_fit)
    joint = M.contrast(a={"dummy": 1}, b={"dummy": 0}, over="grp")
    n_groups = ols_fit.model.data.frame["grp"].nunique()
    assert len(joint.estimate) == n_groups * 3
