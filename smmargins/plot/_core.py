"""Backend-agnostic plot core and dispatch."""

import importlib
from typing import Any, Optional, Protocol, Tuple

from ._grid import PlotData


class Backend(Protocol):
    def render_predictions(
        self, data: PlotData, ax=None, ci_alpha: float = 0.2, **kwargs
    ) -> Tuple[Any, Any]:
        ...

    def render_slopes(
        self, data: PlotData, ax=None, **kwargs
    ) -> Tuple[Any, Any]:
        ...

    def render_comparisons(
        self, data: PlotData, ax=None, **kwargs
    ) -> Tuple[Any, Any]:
        ...


_BACKENDS = {
    "matplotlib": "smmargins.plot._backends.matplotlib:MplBackend",
}


def _get_backend(name: str) -> Backend:
    if name == "plotly":
        raise NotImplementedError("Plotly backend planned for 0.6; use 'matplotlib' for now.")
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Available: {list(_BACKENDS)}")
    module_path, cls_name = _BACKENDS[name].split(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)()
