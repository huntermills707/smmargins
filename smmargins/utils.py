from __future__ import annotations

import numpy as np
from typing import Callable, Optional

def _central_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    rel_step: Optional[float] = None,
) -> np.ndarray:
    r"""Jacobian of ``func`` at ``x`` via central differences.

    This is a numerical-analysis primitive used throughout the module
    whenever an analytic derivative is unavailable.

    Parameters
    ----------
    func : callable
        Maps a length-``p`` vector to a scalar or length-``m`` vector.
    x : ndarray of shape (p,)
        Point at which to evaluate the Jacobian.
    rel_step : float, optional
        Relative step size. Default is :math:`\epsilon^{1/3}`, the
        truncation-vs-rounding sweet spot for central differences.

    Returns
    -------
    ndarray of shape (m, p)
        The Jacobian matrix :math:`J_{ij} = \partial f_i / \partial x_j`.

    Notes
    -----
    Uses the central-difference formula

    .. math::

        \frac{\partial f}{\partial x_j}
        \approx \frac{f(x + h e_j) - f(x - h e_j)}{2 h_j},

    where :math:`h_j = \text{rel\_step} \cdot \max(|x_j|, 1)` and
    :math:`e_j` is the j-th unit vector.  The truncation error is
    :math:`O(h^2)` and the round-off error is :math:`O(\epsilon / h)`;
    choosing :math:`h \sim \epsilon^{1/3}` balances the two.

    References
    ----------
    Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*,
    2nd ed., Springer.  Chapter 8 (Calculating Derivatives).
    """
    x = np.asarray(x, dtype=float).ravel()
    p = x.size
    if rel_step is None:
        rel_step = np.finfo(float).eps ** (1.0 / 3.0)
    h = rel_step * np.maximum(np.abs(x), 1.0)

    # Initial call to determine output size m
    f0 = np.atleast_1d(np.asarray(func(x), dtype=float)).ravel()
    m = f0.size
    J = np.empty((m, p))

    for i in range(p):
        xi = x[i]
        hi = h[i]
        
        # Perturb only dimension i
        x_plus = x.copy()
        x_plus[i] = xi + hi
        
        x_minus = x.copy()
        x_minus[i] = xi - hi
        
        fp = np.atleast_1d(np.asarray(func(x_plus), dtype=float)).ravel()
        fm = np.atleast_1d(np.asarray(func(x_minus), dtype=float)).ravel()
        
        # Central difference for each output component
        J[:, i] = (fp - fm) / (2.0 * hi)

    return J


_METHOD_META = {
    "dydx": {"prefix": "d", "stat_name": "dy/dx"},
    "dyex": {"prefix": "d", "stat_name": "dy/ex"},
    "eyex": {"prefix": "e", "stat_name": "ey/ex"},
    "eydx": {"prefix": "e", "stat_name": "ey/dx"},
}
