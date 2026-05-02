from __future__ import annotations

import numpy as np
from typing import Callable, Optional


def _get_param_cov(results, cov_type=None, vcov=None, cov_kwds=None):
    """Resolve the parameter covariance matrix from results or user overrides.

    Precedence:

    1. ``vcov`` if supplied — used directly as ``Var(β̂)``.
    2. ``cov_type`` if supplied — recomputed via, in order,
       ``results.get_robustcov_results(cov_type=...)``,
       ``results.cov_params(cov_type=...)``, or a re-``fit`` of the
       underlying model with ``cov_type=``.
    3. Otherwise, ``results.cov_params()`` as-is (so a robust ``cov_type``
       baked into the original fit is preserved).

    ``cov_type`` and ``vcov`` are mutually exclusive.

    Parameters
    ----------
    results : statsmodels results object
        Must expose ``params`` and ``cov_params()``.
    cov_type : str, optional
        Covariance type to recompute (e.g. ``"HC0"``..``"HC3"``,
        ``"cluster"``, ``"HAC"``).
    vcov : ndarray, optional
        User-supplied :math:`(k, k)` covariance matrix. Mutually
        exclusive with ``cov_type``.
    cov_kwds : dict, optional
        Additional keywords passed to the covariance computation
        (e.g. ``{"groups": ...}`` for cluster, or
        ``{"maxlags": ..., "kernel": ...}`` for HAC).

    Returns
    -------
    ndarray
        The resolved :math:`(k, k)` parameter covariance matrix.

    Raises
    ------
    ValueError
        If both ``cov_type`` and ``vcov`` are supplied, if ``vcov`` has
        the wrong shape, or if every recomputation path for ``cov_type``
        fails.
    """
    if cov_type is not None and vcov is not None:
        raise ValueError("Only one of cov_type or vcov can be provided")
    if vcov is not None:
        V = np.asarray(vcov, dtype=float)
        params_arr = np.asarray(results.params, dtype=float)
        if params_arr.ndim == 2:
            k = params_arr.ravel(order="F").size
        else:
            k = params_arr.size
        if V.shape != (k, k):
            raise ValueError(
                f"vcov must have shape ({k}, {k}), got {V.shape}"
            )
        return V
    if cov_type is None:
        return np.asarray(results.cov_params(), dtype=float)

    kwds = dict(cov_kwds) if cov_kwds is not None else {}

    # Approach 1: get_robustcov_results (OLS, GLM, etc.)
    if hasattr(results, "get_robustcov_results"):
        try:
            robust = results.get_robustcov_results(cov_type=cov_type, **kwds)
            return np.asarray(robust.cov_params(), dtype=float)
        except Exception:
            pass

    # Approach 2: cov_params with cov_type keyword
    if hasattr(results, "cov_params"):
        try:
            return np.asarray(
                results.cov_params(r_matrix=None, cov_type=cov_type, **kwds),
                dtype=float,
            )
        except TypeError:
            pass
        except Exception:
            pass

    # Approach 3: re-fit the model with cov_type. statsmodels' fit() takes
    # robust kwargs nested under ``cov_kwds=``, NOT splatted at top level.
    try:
        model = results.model
        fit_kwargs = {"cov_type": cov_type}
        if kwds:
            fit_kwargs["cov_kwds"] = kwds
        refit = model.fit(disp=False, **fit_kwargs)
        return np.asarray(refit.cov_params(), dtype=float)
    except Exception:
        pass

    raise ValueError(
        f"Unable to compute covariance matrix for cov_type={cov_type!r}. "
        "Try fitting the model with cov_type directly, or pass vcov manually."
    )

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
