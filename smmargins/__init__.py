from .core import Margins
from .results import MarginsResult, DiDResult, WaldResult
from .transforms import Transform

__version__ = "0.4.0"

__all__ = [
    "Margins",
    "MarginsResult",
    "DiDResult",
    "WaldResult",
    "Transform",
    "__version__",
]
