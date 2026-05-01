"""Tests for n_outcomes and outcome_labels detection (Phase A.1-A.2)."""

import pytest
from smmargins import Margins


def test_ols_n_outcomes(ols_fit):
    M = Margins(ols_fit)
    assert M.n_outcomes == 1
    assert M.outcome_labels is None


def test_logit_n_outcomes(logit_fit):
    M = Margins(logit_fit)
    assert M.n_outcomes == 1
    assert M.outcome_labels is None


def test_poisson_n_outcomes(poisson_fit):
    M = Margins(poisson_fit)
    assert M.n_outcomes == 1
    assert M.outcome_labels is None


def test_mnlogit_n_outcomes(mnlogit_fit):
    M = Margins(mnlogit_fit)
    assert M.n_outcomes == 3
    assert M.outcome_labels is not None
    assert len(M.outcome_labels) == 3


def test_ordered_n_outcomes(ordered_fit):
    M = Margins(ordered_fit)
    assert M.n_outcomes == 4
    assert M.outcome_labels is not None
    assert len(M.outcome_labels) == 4
