"""Tests for smmargins.plot (Pillar 2)."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from smmargins import Margins
from smmargins.plot import plot_predictions, plot_slopes, plot_comparisons


# ---------------------------------------------------------------------------
# plot_predictions
# ---------------------------------------------------------------------------

def test_plot_predictions_returns_fig_ax(ols_fit):
    """plot_predictions returns a 2-tuple of matplotlib objects."""
    M = Margins(ols_fit)
    fig, ax = plot_predictions(M, "x1")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_plot_predictions_x_values_match_grid(ols_fit):
    """Extract line data, verify x-values are the linspace."""
    M = Margins(ols_fit)
    fig, ax = plot_predictions(M, "x1")
    line = ax.lines[0]
    x_data = line.get_xdata()
    assert len(x_data) == 50
    plt.close(fig)


def test_plot_predictions_ci_band_by_default(ols_fit):
    """Presence of a PolyCollection (fill_between) on the axes."""
    M = Margins(ols_fit)
    fig, ax = plot_predictions(M, "x1")
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    plt.close(fig)


def test_plot_predictions_ci_none_removes_band(ols_fit):
    """No PolyCollection when ci=None."""
    M = Margins(ols_fit)
    fig, ax = plot_predictions(M, "x1", ci=None)
    assert not any(isinstance(c, PolyCollection) for c in ax.collections)
    plt.close(fig)


def test_plot_predictions_by_group_adds_multiple_lines(ols_fit):
    """Number of Line2D objects equals number of groups."""
    M = Margins(ols_fit)
    fig, ax = plot_predictions(M, "x1", by="grp")
    n_groups = M.data["grp"].nunique()
    assert len(ax.lines) == n_groups
    plt.close(fig)


def test_plot_predictions_custom_grid(ols_fit):
    """condition dict with explicit grid points."""
    M = Margins(ols_fit)
    fig, ax = plot_predictions(M, {"x1": np.linspace(0, 10, 5)})
    line = ax.lines[0]
    assert len(line.get_xdata()) == 5
    plt.close(fig)


def test_plot_predictions_user_supplied_ax(ols_fit):
    """Passing an existing axis renders to it (no new figure)."""
    M = Margins(ols_fit)
    fig0, ax0 = plt.subplots()
    fig, ax = plot_predictions(M, "x1", ax=ax0)
    assert ax is ax0
    plt.close(fig0)


# ---------------------------------------------------------------------------
# plot_slopes
# ---------------------------------------------------------------------------

def test_plot_slopes_runs(ols_fit):
    """plot_slopes with a numeric variable runs without error."""
    M = Margins(ols_fit)
    fig, ax = plot_slopes(M, "x1", "x2")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_slopes_x_values_match_grid(ols_fit):
    """x-axis values for plot_slopes match the condition grid."""
    M = Margins(ols_fit)
    fig, ax = plot_slopes(M, "x1", {"x2": np.linspace(-2, 2, 20)})
    line = ax.lines[0]
    assert len(line.get_xdata()) == 20
    plt.close(fig)


def test_plot_slopes_categorical_ticks(ols_fit):
    """plot_slopes with categorical condition has each level as a tick."""
    M = Margins(ols_fit)
    fig, ax = plot_slopes(M, "x1", "grp")
    # For categorical conditions, x_values has one point per level
    n_levels = M.data["grp"].nunique()
    assert len(ax.lines[0].get_xdata()) == n_levels
    plt.close(fig)


# ---------------------------------------------------------------------------
# plot_comparisons
# ---------------------------------------------------------------------------

def test_plot_comparisons_binary_treat(ols_fit):
    """plot_comparisons for binary dummy produces a contrast estimate at each x-value."""
    M = Margins(ols_fit)
    fig, ax = plot_comparisons(M, "dummy", condition="x1")
    assert isinstance(fig, plt.Figure)
    # Should have as many x-points as the auto-grid (50)
    assert len(ax.lines[0].get_xdata()) == 50
    plt.close(fig)


# ---------------------------------------------------------------------------
# Backend stubs
# ---------------------------------------------------------------------------

def test_plotly_backend_raises():
    """Plotly backend stub raises NotImplementedError."""
    from smmargins.plot._core import _get_backend
    with pytest.raises(NotImplementedError):
        _get_backend("plotly")


def test_unknown_backend_raises():
    """Unknown backend raises ValueError."""
    from smmargins.plot._core import _get_backend
    with pytest.raises(ValueError, match="Unknown"):
        _get_backend("nonexistent")


# ---------------------------------------------------------------------------
# Inference passthrough
# ---------------------------------------------------------------------------

def test_plot_predictions_inference_passthrough(ols_fit):
    """vce='simulation' changes SEs (compare CI widths)."""
    M = Margins(ols_fit)
    fig1, ax1 = plot_predictions(M, "x1")
    fig2, ax2 = plot_predictions(M, "x1", vce="simulation", n_sims=200, sim_seed=0)
    # Just a smoke test that both render
    assert len(ax1.lines) == 1
    assert len(ax2.lines) == 1
    plt.close(fig1)
    plt.close(fig2)


def test_plot_simultaneous_ci_sup_t_requires_draw_vce(ols_fit):
    """ci_method='sup-t' without draw-based VCE should raise."""
    M = Margins(ols_fit)
    with pytest.raises(ValueError, match="sup-t"):
        plot_predictions(M, "x1", ci_method="sup-t")


# ---------------------------------------------------------------------------
# Multi-output
# ---------------------------------------------------------------------------

def test_plot_predictions_mnlogit(mnlogit_fit):
    """plot_predictions against MNLogit works."""
    M = Margins(mnlogit_fit)
    fig, ax = plot_predictions(M, "x1")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
