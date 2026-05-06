"""Release 0.5 parity tests.

External parity against Stata/R requires those tools in CI. The tests below
are structural consistency checks; Stata/R parity assertions should be added
once reference outputs are generated.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from smmargins import Margins


def test_values_percentile_parity_against_manual_quantile(ols_fit, sim_frame):
    """values={"x1": "p25"} matches a manual quantile replacement."""
    M = Margins(ols_fit)
    res = M.predict(values={"x1": "p25"})
    q25 = float(np.percentile(sim_frame["x1"].to_numpy(), 25))
    manual = M.predict(atexog={"x1": q25})
    assert_allclose(res.estimate, manual.estimate, atol=1e-10)


def test_newdata_parity_against_training_subset(ols_fit, sim_frame):
    """newdata= on the first 10 rows matches predict() restricted to those rows."""
    M = Margins(ols_fit)
    subset = sim_frame.head(10).copy()
    # newdata predicts are averaged over the provided rows
    res_new = M.predict(newdata=subset[["x1", "x2"]])
    # Restricted AAP on the same rows
    res_aap = Margins(ols_fit, data=subset).predict()
    assert_allclose(res_new.estimate, res_aap.estimate, atol=1e-6)


def test_contrast_parity_against_manual_delta_ols(ols_fit):
    """Margins.contrast on OLS reproduces a hand-built delta-method Jacobian."""
    M = Margins(ols_fit)
    joint = M.contrast(a={"dummy": 1}, b={"dummy": 0})

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


# TODO: Add Stata parity tests once CI has Stata available:
# - values={"x1": "p25"} parity against Stata `margins, atspec(...)`
# - newdata= parity against R `marginaleffects::predictions(model, newdata=df)`
# - Margins.contrast parity against Stata `margins, ... post; lincom`
