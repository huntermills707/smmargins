from __future__ import annotations

import numpy as np
from typing import Callable, Optional


class Transform:
    """Analytic transformation of the linear predictor.

    Parameters
    ----------
    value : callable
        ``value(eta)`` returns the transformed scale, element-wise.
    grad : callable
        ``grad(eta)`` returns the first derivative ``λ'(eta)``, element-wise.
    hess : callable, optional
        ``hess(eta)`` returns the second derivative ``λ''(eta)``, element-wise.
        Required for ``dydx`` calls; optional for ``predict`` calls.
    name : str, optional
        Human-readable name, used in result labels.

    Notes
    -----
    Custom transforms used in ``Margins.dydx`` **must** supply an analytic
    ``hess``.  Autodiff is deliberately not supported — users who want it
    can wrap their own ``Transform`` with ``jax.grad`` etc.
    """

    def __init__(
        self,
        value: Callable[[np.ndarray], np.ndarray],
        grad: Callable[[np.ndarray], np.ndarray],
        hess: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        name: str = "custom",
    ):
        self.value = value
        self.grad = grad
        self.hess = hess
        self.name = name

    def __repr__(self) -> str:
        return f"Transform(name={self.name!r})"


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------

Identity = Transform(
    value=lambda eta: np.asarray(eta, dtype=float),
    grad=lambda eta: np.ones_like(np.asarray(eta, dtype=float), dtype=float),
    hess=lambda eta: np.zeros_like(np.asarray(eta, dtype=float), dtype=float),
    name="response",
)

Linear = Transform(
    value=lambda eta: np.asarray(eta, dtype=float),
    grad=lambda eta: np.ones_like(np.asarray(eta, dtype=float), dtype=float),
    hess=lambda eta: np.zeros_like(np.asarray(eta, dtype=float), dtype=float),
    name="linear",
)

Exp = Transform(
    value=lambda eta: np.exp(np.asarray(eta, dtype=float)),
    grad=lambda eta: np.exp(np.asarray(eta, dtype=float)),
    hess=lambda eta: np.exp(np.asarray(eta, dtype=float)),
    name="exp",
)

Log = Transform(
    value=lambda eta: np.log(np.asarray(eta, dtype=float)),
    grad=lambda eta: 1.0 / np.asarray(eta, dtype=float),
    hess=lambda eta: -1.0 / (np.asarray(eta, dtype=float) ** 2),
    name="log",
)

Logit = Transform(
    value=lambda eta: _logistic_cdf(np.asarray(eta, dtype=float)),
    grad=lambda eta: _logistic_pdf(np.asarray(eta, dtype=float)),
    hess=lambda eta: _logistic_pdf(np.asarray(eta, dtype=float))
    * (1.0 - 2.0 * _logistic_cdf(np.asarray(eta, dtype=float))),
    name="logit",
)

Probit = Transform(
    value=lambda eta: _norm_cdf(np.asarray(eta, dtype=float)),
    grad=lambda eta: _norm_pdf(np.asarray(eta, dtype=float)),
    hess=lambda eta: -np.asarray(eta, dtype=float) * _norm_pdf(np.asarray(eta, dtype=float)),
    name="probit",
)

OddsRatio = Transform(
    value=lambda eta: np.exp(np.asarray(eta, dtype=float)),
    grad=lambda eta: np.exp(np.asarray(eta, dtype=float)),
    hess=lambda eta: np.exp(np.asarray(eta, dtype=float)),
    name="or",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logistic_cdf(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def _logistic_pdf(eta: np.ndarray) -> np.ndarray:
    e = np.exp(-eta)
    return e / (1.0 + e) ** 2


def _norm_cdf(eta: np.ndarray) -> np.ndarray:
    from scipy import stats
    return stats.norm.cdf(eta)


def _norm_pdf(eta: np.ndarray) -> np.ndarray:
    from scipy import stats
    return stats.norm.pdf(eta)


# ---------------------------------------------------------------------------
# Dispatch table (model-agnostic transforms only)
# ---------------------------------------------------------------------------

_BUILT_IN_SCALES = {
    "response": Identity,
    "linear": Linear,
    "exp": Exp,
    "log": Log,
    "logit": Logit,
    "probit": Probit,
    "or": OddsRatio,
}
