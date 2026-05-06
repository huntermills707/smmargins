from .core import Margins
from .data import Expr
from .results import MarginsResult, DiDResult, WaldResult
from .transforms import Transform

__version__ = "0.5.0"

__all__ = [
    "Margins",
    "MarginsResult",
    "DiDResult",
    "WaldResult",
    "Transform",
    "Expr",
    "__version__",
    "plot_predictions",
    "plot_slopes",
    "plot_comparisons",
]


# Lazy imports for plotting so `import smmargins` does not require matplotlib
_plot_module = None


def _get_plot_module():
    global _plot_module
    if _plot_module is None:
        from . import plot as _plot_module
    return _plot_module


def plot_predictions(*args, **kwargs):
    return _get_plot_module().plot_predictions(*args, **kwargs)


def plot_slopes(*args, **kwargs):
    return _get_plot_module().plot_slopes(*args, **kwargs)


def plot_comparisons(*args, **kwargs):
    return _get_plot_module().plot_comparisons(*args, **kwargs)
