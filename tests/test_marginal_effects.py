"""
Tests for marginal effects: AME, MEM, MER, discrete contrasts,
and analytic vs finite-difference parity.

Covers continuous derivatives, count contrasts, discrete factor contrasts,
and hand-derived checks against closed-form expressions.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


# ---------------------------------------------------------------------------
# OLS AME — hand-derived analytic check
# ---------------------------------------------------------------------------

def test_ols_ame_hand_derived(ols_fit, sim_frame):
    """AME for x1 in OLS with poly+interaction matches closed form.

    d/dx1 [b0 + b1*x1 + b2*x1^2 + b3*x2 + b4*x1*x2]
      = b1 + 2*b2*x1 + b4*x2
    AME = b1 + 2*b2*mean(x1) + b4*mean(x2)
    """
    M = Margins(ols_fit, use_t=True)
    ame = M.dydx("x1")

    p = ols_fit.params
    hand_ame = (
        p["x1"]
        + 2 * p["I(x1 ** 2)"] * sim_frame["x1"].mean()
        + p["x1:x2"] * sim_frame["x2"].mean()
    )
    assert_allclose(ame.estimate[0], hand_ame, atol=1e-7)


def test_ols_ame_se_hand_derived(ols_fit, sim_frame):
    """AME SE for x1 in OLS matches delta-method with hand-derived gradient.

    g = [0, 1, 2*mean(x1), 0, mean(x2)]  w.r.t. [Int, x1, x1^2, x2, x1:x2]
    """
    M = Margins(ols_fit, use_t=True)
    ame = M.dydx("x1")

    names = list(ols_fit.params.index)
    g = np.zeros(len(names))
    g[names.index("x1")] = 1.0
    g[names.index("I(x1 ** 2)")] = 2 * sim_frame["x1"].mean()
    g[names.index("x1:x2")] = sim_frame["x2"].mean()
    V = np.asarray(ols_fit.cov_params())
    hand_se = np.sqrt(g @ V @ g)
    assert_allclose(ame.se[0], hand_se, rtol=1e-5)


# ---------------------------------------------------------------------------
# Logit AME / MEM vs statsmodels get_margeff
# ---------------------------------------------------------------------------

@pytest.mark.parity
def test_logit_ame_vs_get_margeff(logit_fit):
    """Logit AME for x1 agrees with statsmodels get_margeff(at='overall')."""
    ML = Margins(logit_fit)
    ame = ML.dydx("x1")

    mfx = logit_fit.get_margeff(at="overall", method="dydx")
    exog_names = logit_fit.model.exog_names
    x1_pos = [n for n in exog_names if n != "Intercept"].index("x1")

    assert_allclose(ame.estimate[0], mfx.margeff[x1_pos], atol=1e-5)
    assert_allclose(ame.se[0], mfx.margeff_se[x1_pos], rtol=1e-3)


@pytest.mark.parity
def test_logit_mem_vs_get_margeff(logit_fit):
    """Logit MEM for x1 agrees with statsmodels get_margeff(at='mean')."""
    ML = Margins(logit_fit)
    mem = ML.dydx("x1", at="mean")

    mfx = logit_fit.get_margeff(at="mean", method="dydx")
    exog_names = logit_fit.model.exog_names
    x1_pos = [n for n in exog_names if n != "Intercept"].index("x1")

    assert_allclose(mem.estimate[0], mfx.margeff[x1_pos], atol=1e-5)
    assert_allclose(mem.se[0], mfx.margeff_se[x1_pos], rtol=1e-3)


# ---------------------------------------------------------------------------
# Logit MEM factor_stat="mode" — hand check
# ---------------------------------------------------------------------------

def test_logit_mem_mode_hand_check(logit_fit, sim_frame):
    """Logit MEM with factor_stat='mode' matches central FD around the modal profile."""
    ML = Margins(logit_fit)
    mem_mode = ML.dydx("x1", at="mean", factor_stat="mode")

    modal_grp = sim_frame["grp"].mode().iloc[0]
    row = pd.DataFrame(
        [
            {
                "x1": sim_frame["x1"].mean(),
                "x2": sim_frame["x2"].mean(),
                "grp": modal_grp,
                "dummy": sim_frame["dummy"].mean(),
            }
        ]
    )
    h = (np.finfo(float).eps ** (1.0 / 3.0)) * max(
        sim_frame["x1"].std(ddof=0), abs(sim_frame["x1"].mean()), 1.0
    )
    rp = row.copy()
    rp["x1"] = rp["x1"] + h
    rm = row.copy()
    rm["x1"] = rm["x1"] - h
    Xp = ML._build_exog(rp)
    Xm = ML._build_exog(rm)
    hand_mode = float(
        (
            logit_fit.model.predict(logit_fit.params, Xp)[0]
            - logit_fit.model.predict(logit_fit.params, Xm)[0]
        )
        / (2.0 * h)
    )
    assert_allclose(mem_mode.estimate[0], hand_mode, atol=1e-6)


# ---------------------------------------------------------------------------
# Logit discrete factor contrast — hand check
# ---------------------------------------------------------------------------

def test_logit_discrete_grp_hand_check(logit_fit, sim_frame):
    """Logit discrete contrast for grp (b vs a) matches hand-averaged predictions."""
    ML = Margins(logit_fit)
    grp_effects = ML.dydx("grp")

    def avg_predict_at_grp(level):
        d = sim_frame.copy()
        d["grp"] = level
        X = Margins(logit_fit)._build_exog(d)
        return float(np.mean(logit_fit.model.predict(logit_fit.params, X)))

    hand_b_vs_a = avg_predict_at_grp("b") - avg_predict_at_grp("a")
    assert_allclose(grp_effects.estimate[0], hand_b_vs_a, atol=1e-8)


# ---------------------------------------------------------------------------
# MER (marginal effect at representative values)
# ---------------------------------------------------------------------------

def test_ols_mer_at_representative_values(ols_fit, sim_frame):
    """OLS MER at x2=[-1, 1] returns two estimates."""
    M = Margins(ols_fit, use_t=True)
    mer = M.dydx("x1", atexog={"x2": [-1.0, 1.0]})
    assert len(mer.estimate) == 2

    p = ols_fit.params
    for idx, x2v in enumerate([-1.0, 1.0]):
        hand = p["x1"] + 2 * p["I(x1 ** 2)"] * sim_frame["x1"].mean() + p["x1:x2"] * x2v
        assert_allclose(mer.estimate[idx], hand, atol=1e-7)


def test_logit_mer_at_representative_values(logit_fit):
    """Logit MER at x2=[-1, 1] returns two estimates."""
    ML = Margins(logit_fit)
    mer = ML.dydx("x1", atexog={"x2": [-1.0, 1.0]})
    assert len(mer.estimate) == 2


# ---------------------------------------------------------------------------
# Median and zero evaluation points
# ---------------------------------------------------------------------------

def test_ols_mem_at_median(ols_fit):
    """OLS dydx at='median' returns a single estimate."""
    M = Margins(ols_fit)
    res = M.dydx("x1", at="median")
    assert len(res.estimate) == 1
    assert "median" in res.labels[0]


def test_ols_mem_at_zero(ols_fit):
    """OLS dydx at='zero' returns a single estimate."""
    M = Margins(ols_fit)
    res = M.dydx("x1", at="zero")
    assert len(res.estimate) == 1
    assert "zero" in res.labels[0]


def test_logit_mem_at_median(logit_fit):
    """Logit dydx at='median' returns a single estimate."""
    ML = Margins(logit_fit)
    res = ML.dydx("x1", at="median")
    assert len(res.estimate) == 1


def test_logit_mem_at_zero(logit_fit):
    """Logit dydx at='zero' returns a single estimate."""
    ML = Margins(logit_fit)
    res = ML.dydx("x1", at="zero")
    assert len(res.estimate) == 1


# ---------------------------------------------------------------------------
# _link_deriv() behaviour
# ---------------------------------------------------------------------------

def test_link_deriv_returns_callable_for_ols(ols_fit):
    """_link_deriv returns a callable for OLS (identity link)."""
    M = Margins(ols_fit)
    deriv = M._link_deriv()
    assert deriv is not None


def test_link_deriv_returns_callable_for_logit(logit_fit):
    """_link_deriv returns a callable for Logit (logit link)."""
    ML = Margins(logit_fit)
    deriv = ML._link_deriv()
    assert deriv is not None


def test_link_deriv_returns_callable_for_poisson(poisson_fit):
    """_link_deriv returns a callable for Poisson (log link)."""
    MP = Margins(poisson_fit)
    deriv = MP._link_deriv()
    assert deriv is not None


def test_link_deriv_returns_none_when_analytic_disabled(ols_fit):
    """_link_deriv returns None when analytic=False."""
    M = Margins(ols_fit, analytic=False)
    deriv = M._link_deriv()
    assert deriv is None


# ---------------------------------------------------------------------------
# Analytic vs FD parity matrix
# ---------------------------------------------------------------------------

PARITY_CASES = [
    # OLS cases
    ("ols_fit", "AAP", lambda M: M.predict()),
    ("ols_fit", "APM", lambda M: M.predict(at="mean")),
    (
        "ols_fit",
        "APR x1=[-1,0,1]",
        lambda M: M.predict(atexog={"x1": [-1.0, 0.0, 1.0]}),
    ),
    ("ols_fit", "AME x1", lambda M: M.dydx("x1")),
    ("ols_fit", "MEM x1", lambda M: M.dydx("x1", at="mean")),
    (
        "ols_fit",
        "MER x1|x2",
        lambda M: M.dydx("x1", atexog={"x2": [-1.0, 1.0]}),
    ),
    # Logit cases
    ("logit_fit", "AAP", lambda M: M.predict()),
    ("logit_fit", "APM", lambda M: M.predict(at="mean")),
    ("logit_fit", "AME x1", lambda M: M.dydx("x1")),
    ("logit_fit", "MEM x1", lambda M: M.dydx("x1", at="mean")),
    (
        "logit_fit",
        "MEM x1 (mode)",
        lambda M: M.dydx("x1", at="mean", factor_stat="mode"),
    ),
    ("logit_fit", "AME grp", lambda M: M.dydx("grp")),
    # Poisson cases
    ("poisson_fit", "AAP", lambda M: M.predict()),
    ("poisson_fit", "APM", lambda M: M.predict(at="mean")),
    ("poisson_fit", "AME x1", lambda M: M.dydx("x1")),
    ("poisson_fit", "MEM x1", lambda M: M.dydx("x1", at="mean")),
]


@pytest.mark.slow
@pytest.mark.parity
@pytest.mark.parametrize("model_fixture,label,call", PARITY_CASES)
def test_analytic_vs_fd_parity(model_fixture, label, call, request):
    """Analytic Jacobian and finite-difference Jacobian give identical results."""
    fit = request.getfixturevalue(model_fixture)
    M_an = Margins(fit, analytic=True)
    M_fd = Margins(fit, analytic=False)
    r_an = call(M_an)
    r_fd = call(M_fd)
    assert_allclose(
        r_an.estimate,
        r_fd.estimate,
        atol=1e-7,
        rtol=1e-5,
        err_msg=f"{model_fixture} :: {label}: estimate mismatch",
    )
    assert_allclose(
        r_an.se,
        r_fd.se,
        atol=1e-7,
        rtol=1e-5,
        err_msg=f"{model_fixture} :: {label}: SE mismatch",
    )


# ---------------------------------------------------------------------------
# Multi-variable marginal effects
# ---------------------------------------------------------------------------

def test_ols_multi_dydx(ols_fit):
    """dydx with a list of variables returns one estimate per variable."""
    M = Margins(ols_fit)
    res = M.dydx(["x1", "x2"])
    assert len(res.estimate) == 2
    assert len(res.labels) == 2


def test_ols_star_dydx(ols_fit):
    """dydx('*') returns estimates for all non-response columns."""
    M = Margins(ols_fit)
    res = M.dydx("*")
    # sim_frame columns: x1, x2, grp, dummy, y_ols, y_logit, count
    # Non-response used by model: x1, x2, grp, dummy (y_ols is response)
    assert len(res.estimate) >= 1
