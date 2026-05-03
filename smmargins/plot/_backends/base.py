"""Backend protocol/ABC stub."""

from typing import Any, Tuple

from .._grid import PlotData


class BaseBackend:
    """Abstract base for plotting backends."""

    def render_predictions(
        self, data: PlotData, ax=None, ci_alpha: float = 0.2, **kwargs
    ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def render_slopes(
        self, data: PlotData, ax=None, **kwargs
    ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def render_comparisons(
        self, data: PlotData, ax=None, **kwargs
    ) -> Tuple[Any, Any]:
        raise NotImplementedError
