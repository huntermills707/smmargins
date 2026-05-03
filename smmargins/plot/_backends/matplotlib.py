"""Matplotlib backend for smmargins plotting."""

from typing import Any, Tuple

import numpy as np

from .._grid import PlotData
from .._palette import get_color


class MplBackend:
    """Matplotlib rendering backend."""

    def render_predictions(
        self, data: PlotData, ax=None, ci_alpha: float = 0.2, **kwargs
    ) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x = data.x_values
        n_by = data.estimates.shape[1] if data.estimates.ndim > 1 else 1

        for i in range(n_by):
            est = data.estimates[:, i] if n_by > 1 else data.estimates
            lower = data.lower[:, i] if n_by > 1 else data.lower
            upper = data.upper[:, i] if n_by > 1 else data.upper
            color = get_color(i)
            label = data.by_levels[i] if data.by_levels is not None else None
            if ci_alpha > 0:
                ax.fill_between(x, lower, upper, alpha=ci_alpha, color=color)
            ax.plot(x, est, color=color, label=label)

        ax.set_xlabel(data.x_name)
        ax.set_ylabel(data.label)
        if data.by_levels is not None:
            ax.legend(title=data.by_name)
        return fig, ax

    def render_slopes(
        self, data: PlotData, ax=None, **kwargs
    ) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x = data.x_values
        n_by = data.estimates.shape[1] if data.estimates.ndim > 1 else 1
        ci_alpha = kwargs.get("ci_alpha", 0.2)

        for i in range(n_by):
            est = data.estimates[:, i] if n_by > 1 else data.estimates
            lower = data.lower[:, i] if n_by > 1 else data.lower
            upper = data.upper[:, i] if n_by > 1 else data.upper
            color = get_color(i)
            label = data.by_levels[i] if data.by_levels is not None else None
            if ci_alpha > 0:
                ax.fill_between(x, lower, upper, alpha=ci_alpha, color=color)
            ax.plot(x, est, color=color, label=label)

        ax.set_xlabel(data.x_name)
        ax.set_ylabel(data.label)
        if data.by_levels is not None:
            ax.legend(title=data.by_name)
        return fig, ax

    def render_comparisons(
        self, data: PlotData, ax=None, **kwargs
    ) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        x = data.x_values
        n_by = data.estimates.shape[1] if data.estimates.ndim > 1 else 1

        ci_alpha = kwargs.get("ci_alpha", 0.2)
        for i in range(n_by):
            est = data.estimates[:, i] if n_by > 1 else data.estimates
            lower = data.lower[:, i] if n_by > 1 else data.lower
            upper = data.upper[:, i] if n_by > 1 else data.upper
            color = get_color(i)
            label = data.by_levels[i] if data.by_levels is not None else None
            if ci_alpha > 0:
                ax.fill_between(x, lower, upper, alpha=ci_alpha, color=color)
            ax.plot(x, est, color=color, label=label)

        ax.set_xlabel(data.x_name)
        ax.set_ylabel(data.label)
        if data.by_levels is not None:
            ax.legend(title=data.by_name)
        return fig, ax
