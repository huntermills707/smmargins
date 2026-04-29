"""
Difference-in-differences (DiD) and contrast tests.

These tests verify:
1. OLS DiD equals the interaction coefficient (canonical sanity check)
2. Logit DiD on probability scale differs from the interaction coeff (Ai & Norton 2003)
3. Contrast round-trip: did() == predict().contrast() with the DiD vector
4. Joint contrast matrix produces the same estimates and vcov as did()
5. Covariate-adjusted DiD (smoke tests for at=mean and atexog)
"""
import numpy as np
import statsmodels.formula.api as smf
from numpy.testing import assert_allclose
import pytest
from smmargins import Margins


# ── OLS DiD ────────────────────────────────────────────────────────────────

def test_ols_did_equals_interaction_coefficient(did_frame):
    """On OLS with identity link, DiD equals the treat:post coefficient.

    For a linear model y = beta0 + beta1*treat + beta2*post + beta3*(treat:post) + ... + eps,
    the DiD on the response scale is exactly beta3. This is because the identity
    link is linear, so E[y|treat,post] = beta0 + beta1*treat + beta2*post + beta3*(treat*post).
    The four cell means are:
        m00 = beta0,               m01 = beta0 + beta2
        m10 = beta0 + beta1,       m11 = beta0 + beta1 + beta2 + beta3
    DiD = (m11 - m01) - (m10 - m00) = beta3.
    """
    ols = smf.ols("y_lin ~ treat + post + treat:post + x", data=did_frame).fit()
    M = Margins(ols)
    did_res = M.did("treat", "post")

    b_inter = ols.params["treat:post"]
    se_inter = ols.bse["treat:post"]
    assert_allclose(did_res.did.estimate[0], b_inter, atol=1e-8)
    assert_allclose(did_res.did.se[0], se_inter, rtol=1e-5)

    # Simple effect at post=0 equals treat coefficient
    b_treat = ols.params["treat"]
    se_treat = ols.bse["treat"]
    assert_allclose(did_res.simple_effects.estimate[0], b_treat, atol=1e-8)
    assert_allclose(did_res.simple_effects.se[0], se_treat, rtol=1e-5)


# ── Logit DiD ──────────────────────────────────────────────────────────────

def test_logit_did_prob_scale_differs_from_interaction_coeff(did_frame):
    """On logit, DiD on probability scale != interaction coeff (Ai & Norton 2003).

    The coefficient on treat:post is on the log-odds scale. On the probability
    scale, the DiD is a nonlinear function of all parameters and covariate
    profiles -- you cannot read it off the interaction coefficient. This is
    exactly the Ai & Norton (2003) issue.

    We verify by:
    1. Checking that the DiD estimate is NOT close to the log-odds coefficient.
    2. Hand-computing the four cell means and the DiD, confirming exact match.

    References
    ----------
    Ai, C. and Norton, E. C. (2003). Interaction terms in logit and probit
    models. Economics Letters, 80(1), 123-129.
    """
    logit = smf.logit("y_bin ~ treat + post + treat:post + x", data=did_frame).fit(disp=False)
    ML = Margins(logit)
    did_log = ML.did("treat", "post")

    b_int_logit = logit.params["treat:post"]
    # They should NOT be equal
    assert not np.isclose(did_log.did.estimate[0], b_int_logit, atol=0.05)

    # Hand computation of 4 cells by setting treat/post and averaging predictions
    def hand_cell(t, p):
        d = did_frame.copy()
        d["treat"] = t
        d["post"] = p
        return float(logit.model.predict(logit.params, ML._build_exog(d)).mean())

    m00 = hand_cell(0, 0)
    m01 = hand_cell(0, 1)
    m10 = hand_cell(1, 0)
    m11 = hand_cell(1, 1)
    hand_did = (m11 - m01) - (m10 - m00)
    assert_allclose(did_log.did.estimate[0], hand_did, atol=1e-10)


# ── Contrast round-trip ────────────────────────────────────────────────────

def test_contrast_roundtrip_equals_did(did_frame):
    """did() == predict(atexog).contrast([1, -1, -1, 1]).

    The DiD can be obtained directly via did() or by computing the four cell
    predictions and applying the contrast vector [1, -1, -1, 1] (which is
    m00 - m01 - m10 + m11).  Both paths must give identical estimates and SEs.
    """
    logit = smf.logit("y_bin ~ treat + post + treat:post + x", data=did_frame).fit(disp=False)
    ML = Margins(logit)
    did_direct = ML.did("treat", "post")

    cells = ML.predict(atexog={"treat": [0, 1], "post": [0, 1]})
    did_via_contrast = cells.contrast([1, -1, -1, 1], labels=["DiD"])

    assert_allclose(did_via_contrast.estimate[0], did_direct.did.estimate[0], atol=1e-12)
    assert_allclose(did_via_contrast.se[0], did_direct.did.se[0], atol=1e-12)


# ── Joint contrast matrix ──────────────────────────────────────────────────

def test_joint_contrast_matrix(did_frame):
    """The joint contrast from predict().contrast(C) matches did().joint.

    C = [[-1, 0, 1, 0],    -> simple effect of treat at post=0
         [0, -1, 0, 1],    -> simple effect of treat at post=1
         [1, -1, -1, 1]]   -> DiD

    All three contrasts share a joint covariance from the delta method.
    """
    logit = smf.logit("y_bin ~ treat + post + treat:post + x", data=did_frame).fit(disp=False)
    ML = Margins(logit)
    did_direct = ML.did("treat", "post")

    cells = ML.predict(atexog={"treat": [0, 1], "post": [0, 1]})
    C = np.array([[-1, 0, 1, 0], [0, -1, 0, 1], [1, -1, -1, 1]], float)
    joint = cells.contrast(C, labels=["B@p=0", "B@p=1", "DiD"])

    assert_allclose(joint.estimate, did_direct.joint.estimate, atol=1e-12)
    assert_allclose(joint.vcov, did_direct.joint.vcov, atol=1e-12)


# ── DiD with covariate adjustment (smoke tests) ────────────────────────────

def test_did_at_mean(did_frame):
    """DiD with at='mean' should return a single positive SE estimate."""
    logit = smf.logit("y_bin ~ treat + post + treat:post + x", data=did_frame).fit(disp=False)
    ML = Margins(logit)
    res = ML.did("treat", "post", at="mean")
    assert res.did.estimate.size == 1
    assert res.did.se[0] > 0


def test_did_with_atexog(did_frame):
    """DiD with atexog should return a single positive SE estimate."""
    logit = smf.logit("y_bin ~ treat + post + treat:post + x", data=did_frame).fit(disp=False)
    ML = Margins(logit)
    res = ML.did("treat", "post", atexog={"x": 1.5})
    assert res.did.estimate.size == 1
    assert res.did.se[0] > 0
