"""Plotting subpackage for smmargins."""

from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np

from ..core import Margins
from ._grid import PlotData, build_grid
from ._core import _get_backend


def plot_predictions(
    margins: Margins,
    condition: Union[str, Mapping[str, Any]],
    *,
    by: Optional[str] = None,
    ci: Optional[float] = 0.95,
    ci_method: str = "pointwise",
    backend: str = "matplotlib",
    ax=None,
    outcome: Optional[Union[int, str]] = None,
    **predict_kwargs,
) -> Tuple[Any, Any]:
    """Plot predictions across a covariate.

    Returns (fig, ax) for the matplotlib backend.
    """
    x_name, x_values, by_name, by_levels = build_grid(margins, condition, by=by)

    if by_levels is not None:
        estimates = np.empty((len(x_values), len(by_levels)))
        lower = np.empty((len(x_values), len(by_levels)))
        upper = np.empty((len(x_values), len(by_levels)))
        for j, lvl in enumerate(by_levels):
            # Build values spec with condition and by level
            values = {x_name: x_values.tolist(), by_name: lvl}
            res = margins.predict(
                values=values,
                outcome=outcome,
                ci_method=ci_method,
                **predict_kwargs,
            )
            estimates[:, j] = res.estimate
            lower[:, j] = res.ci_lower
            upper[:, j] = res.ci_upper
    else:
        values = {x_name: x_values.tolist()}
        res = margins.predict(
            values=values,
            outcome=outcome,
            ci_method=ci_method,
            **predict_kwargs,
        )
        # Handle multi-output: reshape from (n_grid*K,) to (n_grid, K)
        n_out = margins.n_outcomes
        if n_out > 1 and outcome is None:
            n_grid = len(x_values)
            estimates = res.estimate.reshape(n_grid, n_out, order="C")
            lower = res.ci_lower.reshape(n_grid, n_out, order="C")
            upper = res.ci_upper.reshape(n_grid, n_out, order="C")
            by_levels = margins.outcome_labels or [str(i) for i in range(n_out)]
            by_name = "outcome"
        else:
            estimates = res.estimate
            lower = res.ci_lower
            upper = res.ci_upper

    data = PlotData(
        x_name=x_name,
        x_values=x_values,
        by_name=by_name,
        by_levels=by_levels,
        estimates=estimates,
        lower=lower,
        upper=upper,
        statistic="prediction",
        label="Predicted y",
    )
    be = _get_backend(backend)
    return be.render_predictions(data, ax=ax, ci_alpha=0.2 if ci is not None else 0.0)


def plot_slopes(
    margins: Margins,
    variables: Union[str, list],
    condition: Union[str, Mapping[str, Any]],
    *,
    by: Optional[str] = None,
    ci: Optional[float] = 0.95,
    ci_method: str = "pointwise",
    backend: str = "matplotlib",
    ax=None,
    outcome: Optional[Union[int, str]] = None,
    **dydx_kwargs,
) -> Tuple[Any, Any]:
    """Plot dy/dx (or other dydx variants) across ``condition``."""
    x_name, x_values, by_name, by_levels = build_grid(margins, condition, by=by)

    if by_levels is not None:
        estimates = np.empty((len(x_values), len(by_levels)))
        lower = np.empty((len(x_values), len(by_levels)))
        upper = np.empty((len(x_values), len(by_levels)))
        for j, lvl in enumerate(by_levels):
            values = {x_name: x_values.tolist(), by_name: lvl}
            res = margins.dydx(
                variables,
                values=values,
                outcome=outcome,
                ci_method=ci_method,
                **dydx_kwargs,
            )
            estimates[:, j] = res.estimate
            lower[:, j] = res.ci_lower
            upper[:, j] = res.ci_upper
    else:
        values = {x_name: x_values.tolist()}
        res = margins.dydx(
            variables,
            values=values,
            outcome=outcome,
            ci_method=ci_method,
            **dydx_kwargs,
        )
        estimates = res.estimate
        lower = res.ci_lower
        upper = res.ci_upper

    data = PlotData(
        x_name=x_name,
        x_values=x_values,
        by_name=by_name,
        by_levels=by_levels,
        estimates=estimates,
        lower=lower,
        upper=upper,
        statistic="dydx",
        label=f"d/dx ({variables})",
    )
    be = _get_backend(backend)
    return be.render_slopes(data, ax=ax, ci_alpha=0.2 if ci is not None else 0.0)


def plot_comparisons(
    margins: Margins,
    variables: Union[str, list],
    condition: Optional[Union[str, Mapping[str, Any]]] = None,
    *,
    by: Optional[str] = None,
    ci: Optional[float] = 0.95,
    ci_method: str = "pointwise",
    backend: str = "matplotlib",
    ax=None,
    outcome: Optional[Union[int, str]] = None,
    **contrast_kwargs,
) -> Tuple[Any, Any]:
    """Plot a contrast (factor levels or a numeric step) across ``condition``."""
    if condition is not None:
        x_name, x_values, by_name, by_levels = build_grid(margins, condition, by=by)
    else:
        x_name = ""
        x_values = np.array([0])
        by_name = None
        by_levels = None

    if isinstance(variables, str):
        var = variables
    else:
        var = variables[0]

    # Determine discrete levels for the variable
    col = margins.data[var]
    from ..plot._grid import _is_numeric
    if _is_numeric(col):
        # Numeric step: compare x+1 vs x
        a_val = lambda v: v + 1
        b_val = lambda v: v
    else:
        levels = sorted(col.dropna().unique().tolist())
        if len(levels) < 2:
            raise ValueError(f"Variable {var!r} has fewer than 2 levels")
        a_val = lambda _: levels[1]
        b_val = lambda _: levels[0]

    def _extract_diff(res):
        # Find the first row whose label starts with "A - B:"
        # This is robust across single/multi-output and over= cases.
        for i, lab in enumerate(res.labels):
            if lab.startswith("A - B:"):
                return res.estimate[i], res.ci_lower[i], res.ci_upper[i]
        raise ValueError("No 'A - B' contrast row found in result")

    if by_levels is not None:
        estimates = np.empty((len(x_values), len(by_levels)))
        lower = np.empty((len(x_values), len(by_levels)))
        upper = np.empty((len(x_values), len(by_levels)))
        for j, lvl in enumerate(by_levels):
            ests = []
            lows = []
            ups = []
            for xv in x_values:
                base = {by_name: lvl} if by_name else {}
                if condition is not None:
                    base[x_name] = xv
                res = margins.contrast(
                    a={var: a_val(xv), **base},
                    b={var: b_val(xv), **base},
                    outcome=outcome,
                    ci_method=ci_method,
                    **contrast_kwargs,
                )
                e, l, u = _extract_diff(res)
                ests.append(e)
                lows.append(l)
                ups.append(u)
            estimates[:, j] = ests
            lower[:, j] = lows
            upper[:, j] = ups
    else:
        ests = []
        lows = []
        ups = []
        for xv in x_values:
            base = {}
            if condition is not None:
                base[x_name] = xv
            res = margins.contrast(
                a={var: a_val(xv), **base},
                b={var: b_val(xv), **base},
                outcome=outcome,
                ci_method=ci_method,
                **contrast_kwargs,
            )
            e, l, u = _extract_diff(res)
            ests.append(e)
            lows.append(l)
            ups.append(u)
        estimates = np.array(ests)
        lower = np.array(lows)
        upper = np.array(ups)

    data = PlotData(
        x_name=x_name,
        x_values=x_values,
        by_name=by_name,
        by_levels=by_levels,
        estimates=estimates,
        lower=lower,
        upper=upper,
        statistic="comparison",
        label=f"Δ {var}",
    )
    be = _get_backend(backend)
    return be.render_comparisons(data, ax=ax, ci_alpha=0.2 if ci is not None else 0.0)
