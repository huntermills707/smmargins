"""
marginal_effects.py
===================

Stata-style ``margins`` for StatsModels, with standard errors from the
delta method.

Reference formulas
------------------

For a fitted model with parameter vector :math:`\\hat\\beta`, estimated
covariance :math:`\\widehat V(\\hat\\beta)`, and a (possibly vector-valued)
statistic :math:`g(\\beta)` (e.g. an adjusted prediction or a marginal
effect), the delta method gives

    Var[g(\\hat\\beta)] ≈ G \\, \\widehat V(\\hat\\beta) \\, G',

where :math:`G = \\partial g / \\partial \\beta` evaluated at
:math:`\\hat\\beta`. That is exactly what Stata's ``margins`` does; see
https://www.stata.com/support/faqs/statistics/compute-standard-errors-with-margins/.

This module supports every statistic discussed in Richard Williams'
*Using the margins command* notes (Margins01):

    * **AAP**  Average Adjusted Prediction
    * **APM**  Adjusted Prediction at Means
    * **APR**  Adjusted Predictions at Representative values
    * **AME**  Average Marginal Effect
    * **MEM**  Marginal Effect at Means
    * **MER**  Marginal Effect at Representative values
    * Discrete changes for factor / binary variables

Design
------

Rather than hand-differentiating each model's linear predictor / link
combination, we build every statistic as a *function of* :math:`\\beta`
using StatsModels' own ``model.predict(params, exog)`` (which handles
the inverse link). We then take the Jacobian of that function by central
finite differences, and apply the delta method with ``results.cov_params()``.

Patsy does the heavy lifting of propagating perturbations through
interactions, polynomials, splines, and categorical encodings: when the
user says "perturb ``x1``", we perturb the *column of the data frame*
and let ``patsy.dmatrix(design_info, ...)`` rebuild the design matrix.
This way ``I(x1**2)``, ``x1:x2``, ``C(group)``, ``bs(x1, df=4)`` all
update correctly and automatically.
"""

from __future__ import annotations

import itertools
from typing import Callable, Optional, Union, Sequence, Mapping, List

import numpy as np
import pandas as pd
import patsy
from scipy import stats


# ---------------------------------------------------------------------------
# Numerical differentiation
# ---------------------------------------------------------------------------

def _central_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    rel_step: Optional[float] = None,
) -> np.ndarray:
    """Jacobian of ``func`` at ``x`` via central differences.

    Parameters
    ----------
    func : callable
        Maps a length-``p`` vector to a scalar or length-``m`` vector.
    x : ndarray of shape (p,)
    rel_step : float, optional
        Relative step. Default ``eps**(1/3)`` (good for central differences).

    Returns
    -------
    ndarray of shape (m, p).
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


# ---------------------------------------------------------------------------
# Marginal-effect method registry
# ---------------------------------------------------------------------------
#
# ``method`` selects the per-observation transform applied to the raw
# derivative dy/dx_i before averaging:
#
#   dydx : dy/dx_i                       (level change in y per unit x)
#   dyex : dy/dx_i * x_i                 (semi-elasticity: dy / d(ln x))
#   eyex : dy/dx_i * x_i / y_i           (full elasticity)
#   eydx : dy/dx_i / y_i                 (semi-elasticity: d(ln y) / dx)
#
# Stat-name strings match Stata's ``margins`` and statsmodels'
# ``get_margeff`` column headers.

_METHOD_META = {
    "dydx": {"prefix": "d", "stat_name": "dy/dx"},
    "dyex": {"prefix": "d", "stat_name": "dy/ex"},
    "eyex": {"prefix": "e", "stat_name": "ey/ex"},
    "eydx": {"prefix": "e", "stat_name": "ey/dx"},
}


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

class MarginsResult:
    """Container for margin estimates with delta-method standard errors.

    Attributes
    ----------
    estimate : ndarray
    se       : ndarray
    labels   : list[str]
    vcov     : ndarray
        Full delta-method covariance of the estimates (useful for joint tests).
    level    : float
    df       : int or None
        Residual degrees of freedom; if set, uses t-distribution for
        p-values and CIs. Otherwise uses N(0,1).
    """

    def __init__(
        self,
        estimate: np.ndarray,
        vcov: np.ndarray,
        labels: Optional[Sequence[str]] = None,
        level: float = 0.95,
        df: Optional[int] = None,
        stat_name: str = "margin",
    ):
        self.estimate = np.atleast_1d(np.asarray(estimate, dtype=float))
        self.vcov = np.atleast_2d(np.asarray(vcov, dtype=float))
        self.se = np.sqrt(np.clip(np.diag(self.vcov), 0, None))
        n = self.estimate.size
        if labels is None:
            labels = [f"m{i+1}" for i in range(n)]
        self.labels = list(labels)
        self.level = level
        self.df = df
        self.stat_name = stat_name

        with np.errstate(divide="ignore", invalid="ignore"):
            self.zstat = np.where(self.se > 0, self.estimate / self.se, np.nan)
        alpha = 1.0 - level
        if df is None:
            self.pvalue = 2.0 * (1.0 - stats.norm.cdf(np.abs(self.zstat)))
            crit = stats.norm.ppf(1.0 - alpha / 2.0)
            self._test_name = "z"
        else:
            self.pvalue = 2.0 * (1.0 - stats.t.cdf(np.abs(self.zstat), df=df))
            crit = stats.t.ppf(1.0 - alpha / 2.0, df=df)
            self._test_name = "t"
        self.ci_lower = self.estimate - crit * self.se
        self.ci_upper = self.estimate + crit * self.se

    def contrast(
        self,
        c: Union[Sequence[float], np.ndarray],
        labels: Optional[Sequence[str]] = None,
        name: str = "contrast",
    ) -> "MarginsResult":
        """Form linear contrasts of the estimates, with delta-method SEs.

        If the estimates have joint covariance :math:`V_m`, any linear
        combination :math:`C m` has covariance :math:`C V_m C^\\top`, and
        :math:`V_m` was already built from the delta method on
        :math:`\\beta`, so this is exact under the same approximation
        (no extra differentiation).

        Parameters
        ----------
        c : 1-D array of length n, or 2-D array of shape (k, n)
            A single contrast vector or a matrix whose rows are contrasts.
            ``n`` must equal ``len(self.estimate)``.
        labels : sequence of str, optional
            Row labels. Defaults to ``["c1", "c2", ...]``.
        name : str
            Column name for the statistic in the summary table.

        Examples
        --------
        Simple effect of `treat` at `post=1`, from a 4-cell
        (treat × post) prediction ordered (0,0), (0,1), (1,0), (1,1)::

            cells.contrast([0, -1, 0, 1], labels=["treat effect | post=1"])

        All three contrasts at once, preserving their joint covariance::

            cells.contrast(
                [[-1, 0, 1, 0],     # treat effect at post=0
                 [ 0,-1, 0, 1],     # treat effect at post=1
                 [ 1,-1,-1, 1]],    # DiD
                labels=["simple @post=0", "simple @post=1", "DiD"],
            )
        """
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C.reshape(1, -1)
        if C.shape[1] != self.estimate.size:
            raise ValueError(
                f"contrast has {C.shape[1]} columns, but there are "
                f"{self.estimate.size} estimates to combine"
            )
        if labels is None:
            labels = (
                ["contrast"] if C.shape[0] == 1
                else [f"c{i+1}" for i in range(C.shape[0])]
            )
        if len(labels) != C.shape[0]:
            raise ValueError("labels length must match number of contrasts")

        est = C @ self.estimate
        vcov = C @ self.vcov @ C.T
        return MarginsResult(
            estimate=est, vcov=vcov, labels=labels,
            level=self.level, df=self.df, stat_name=name,
        )

    def summary(self) -> pd.DataFrame:
        tn = self._test_name
        pct = int(round(self.level * 100))
        return pd.DataFrame(
            {
                self.stat_name: self.estimate,
                "std.err": self.se,
                tn: self.zstat,
                f"P>|{tn}|": self.pvalue,
                f"[{pct}% CI lo]": self.ci_lower,
                f"[{pct}% CI hi]": self.ci_upper,
            },
            index=self.labels,
        )

    def __repr__(self) -> str:
        return self.summary().to_string(float_format=lambda v: f"{v: .6f}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Margins:
    """Compute adjusted predictions and marginal effects for a StatsModels fit.

    Parameters
    ----------
    results : statsmodels results object
        Must expose ``params``, ``cov_params()``, and have
        ``model.predict(params, exog)`` available (OLS, GLM, GLM-like,
        Logit, Probit, Poisson, NegBin, GEE, mixed, ... all qualify).
    data : pandas.DataFrame, optional
        Original fitting data. If ``None``, we try ``model.data.frame``.
    level : float
        Confidence level for intervals (default 0.95).
    use_t : bool
        If True and ``results.df_resid`` is available, use t-distribution.
    analytic : bool
        If True (default), use an analytic outer Jacobian
        :math:`\\partial g/\\partial\\beta` whenever the model exposes a link
        derivative (any GLM via ``family.link.inverse_deriv``, plus
        ``OLS``/``WLS``/``GLS`` via the identity link). Falls back to central
        finite differences otherwise. Set to False to force FD everywhere.

    Notes
    -----
    **Formula vs. Raw Exog Mode**

    If the model was fit using a formula (e.g. ``smf.ols("y ~ x1 + x2", data)``),
    ``Margins`` uses the model's ``DesignInfo`` to rebuild the design matrix.
    This ensures that interactions (``x1:x2``) and transformations (``log(x1)``)
    are correctly updated when a single variable is perturbed for marginal
    effects.

    If the model was fit using raw matrices (e.g. ``sm.OLS(y, X).fit()``),
    ``Margins`` operates in "raw mode". In this mode, only the literal column
    corresponding to the variable is perturbed. Interactions or pre-computed
    transformations in the design matrix will *not* be automatically updated.
    To correctly handle such models, it is recommended to fit using formulas.
    """

    def __init__(
        self,
        results,
        data: Optional[pd.DataFrame] = None,
        level: float = 0.95,
        use_t: bool = False,
        analytic: bool = True,
    ):
        self.results = results
        self.model = results.model
        self.params = np.asarray(results.params, dtype=float).ravel()
        self.cov = np.asarray(results.cov_params(), dtype=float)
        if self.cov.shape != (self.params.size, self.params.size):
            raise ValueError(
                f"cov_params has shape {self.cov.shape}, expected "
                f"({self.params.size}, {self.params.size})"
            )
        self.level = level
        self.df = getattr(results, "df_resid", None) if use_t else None
        self.analytic = analytic

        self.design_info = self._try_get_design_info()
        self._raw_mode = self.design_info is None

        if data is None:
            data = self._try_get_data()
        if data is None and self._raw_mode:
            # Synthesize data from model.exog
            exog = np.asarray(self.model.exog)
            names = list(getattr(self.model, "exog_names", None) or
                         [f"x{i}" for i in range(exog.shape[1])])
            data = pd.DataFrame(exog, columns=names)

        if data is None:
            raise ValueError(
                "Could not retrieve the original data frame from the model; "
                "please pass ``data=`` explicitly."
            )
        self.data = data.copy()

        if self._raw_mode:
            # Sanity check alignment
            n_exog = self.model.exog.shape[1]
            n_params = self.params.size
            if n_exog != n_params:
                raise ValueError(
                    f"Model has {n_exog} exog columns but {n_params} parameters. "
                    "Cannot use raw mode."
                )
            if len(self.model.exog_names) != n_params:
                raise ValueError(
                    f"Model has {len(self.model.exog_names)} exog_names but "
                    f"{n_params} parameters. Cannot use raw mode."
                )

        # Cache training offset/exposure for _predict broadcasting on smaller
        # designs (MEM/APM/etc.). Without this, statsmodels' predict can't
        # align a length-N stored offset against a 1-row exog and silently
        # drops the offset. We collapse offset + log(exposure) into a single
        # log-offset and pass it as ``offset=`` to predict — that way we
        # don't have to care which storage convention the model uses
        # (statsmodels stores ``model.exposure`` as ``log(exposure)`` for
        # GLM, raw for discrete models, etc.).
        self._n_train = (self.model.exog.shape[0]
                         if getattr(self.model, "exog", None) is not None else None)
        self._mean_log_offset_value = self._compute_mean_log_offset()

    def _compute_mean_log_offset(self) -> Optional[float]:
        """Mean of (offset + log(exposure)) across the training sample, or None."""
        oe = getattr(self.model, "_offset_exposure", None)
        if oe is not None:
            a = np.asarray(oe, dtype=float).ravel()
            if a.size > 0 and np.any(a):
                return float(np.mean(a))
            return None
        # Fall back: combine offset and exposure manually. statsmodels stores
        # ``exposure`` as already log-transformed for GLM; discrete models
        # apply log internally too. Either way ``model.exposure`` is in log
        # space when present.
        parts = []
        for attr in ("offset", "exposure"):
            v = getattr(self.model, attr, None)
            if v is None:
                continue
            a = np.asarray(v, dtype=float).ravel()
            if a.size > 0 and np.any(a):
                parts.append(a)
        if not parts:
            return None
        try:
            return float(np.mean(sum(parts)))
        except ValueError:
            return None

    # ---- model / data introspection ----

    def _try_get_data(self):
        try:
            return self.model.data.frame
        except AttributeError:
            return None

    def _try_get_design_info(self):
        exog = getattr(self.model.data, "orig_exog", None)
        if exog is not None and hasattr(exog, "design_info"):
            return exog.design_info
        # Some model wrappers stash it differently:
        di = getattr(self.model.data, "design_info", None)
        if di is not None:
            return di
        return None

    # ---- building design matrices ----

    def _build_exog(self, frame: pd.DataFrame) -> np.ndarray:
        """Build a numeric design matrix from a data frame.

        If the model was fit with a formula, this uses patsy's DesignInfo
        to rebuild the matrix (handling interactions, etc.). If fit with
        raw matrices, it selects columns matching ``model.exog_names``.
        """
        if self._raw_mode:
            cols = list(self.model.exog_names)
            return np.asarray(frame[cols].to_numpy(dtype=float))
        dm = patsy.dmatrix(self.design_info, frame, return_type="matrix")
        return np.asarray(dm)

    # ---- core prediction on the response scale ----

    def _predict(self, params: np.ndarray, exog: np.ndarray) -> np.ndarray:
        """Return E[Y | X] (response scale) using the model's own predict.

        Many StatsModels results accept a ``params`` override to
        ``model.predict``; for GLMs this applies the inverse link. When
        the model carries offset/exposure but the design has a different
        row count than training (e.g. APM/MEM single rows), we broadcast
        the *mean* training offset/exposure to the new design — otherwise
        statsmodels can't align the stored length-N offset against the
        smaller exog and silently drops it.
        """
        exog_arr = np.asarray(exog)
        n_exog = exog_arr.shape[0]
        kwargs = {}
        if (self._n_train is not None
                and n_exog != self._n_train
                and self._mean_log_offset_value is not None):
            kwargs["offset"] = np.full(n_exog, self._mean_log_offset_value)
        try:
            return np.asarray(self.model.predict(params, exog, **kwargs))
        except (TypeError, ValueError, KeyError):
            eta = exog_arr @ np.asarray(params)
            if "offset" in kwargs:
                eta = eta + kwargs["offset"]
            fam = getattr(self.model, "family", None)
            if fam is not None:
                return np.asarray(fam.link.inverse(eta))
            return eta

    # ---- analytic outer-Jacobian support ----

    def _link_deriv(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Callable returning :math:`f'(\\eta)`, or ``None`` if FD is required.

        Eligible when the model exposes ``family.link.inverse_deriv`` (every
        stock GLM family) or is a linear-regression model (identity link).
        Bails out — returning ``None`` so the caller falls back to FD —
        whenever an offset or exposure is present, since then
        :math:`\\eta \\neq X\\beta` and our chain rule would need an
        offset-aware path we don't currently provide.
        """
        if not self.analytic:
            return None
        # Offset/exposure changes the linear-predictor formula.
        off = getattr(self.model, "_offset_exposure", None)
        if off is not None and np.any(np.asarray(off)):
            return None
        for attr in ("offset", "exposure"):
            v = getattr(self.model, attr, None)
            if v is not None and np.any(np.asarray(v)):
                return None

        fam = getattr(self.model, "family", None)
        if fam is not None:
            link = getattr(fam, "link", None)
            if link is not None and callable(getattr(link, "inverse_deriv", None)):
                return link.inverse_deriv
            return None

        try:
            from statsmodels.regression.linear_model import RegressionModel
        except ImportError:
            return None
        if isinstance(self.model, RegressionModel):
            return lambda eta: np.ones_like(np.asarray(eta, dtype=float))
        return None

    @staticmethod
    def _grad_mean_predict(
        X: np.ndarray,
        beta: np.ndarray,
        fprime: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """:math:`\\partial/\\partial\\beta` of :math:`(1/n)\\sum_i f(x_i^\\top\\beta)`.

        Returns shape ``(p,)``. Building block for every analytic Jacobian
        in this module: every statistic we compute is a linear combination
        of mean-predictions on different design matrices, so each row of
        the analytic ``J`` is a linear combination of these gradients.
        """
        X = np.asarray(X, dtype=float)
        eta = X @ beta
        fp = np.asarray(fprime(eta), dtype=float).ravel()
        return (fp[:, None] * X).sum(axis=0) / X.shape[0]

    # ---- utilities for building "at" frames ----

    @staticmethod
    def _expand_at(
        base: pd.DataFrame, at: Optional[Mapping[str, object]]
    ) -> tuple[List[pd.DataFrame], List[str]]:
        """Cartesian-product expansion of an ``at`` specification.

        ``at`` maps variable names to either scalars or lists of scalars.
        Returns (frames, labels) where each frame is ``base`` with the
        chosen values broadcast.
        """
        if not at:
            return [base.copy()], [""]
        keys, vals = [], []
        for k, v in at.items():
            if np.isscalar(v) or isinstance(v, (str, bytes)):
                vals.append([v])
            else:
                vals.append(list(v))
            keys.append(k)
        frames, labels = [], []
        for combo in itertools.product(*vals):
            f = base.copy()
            for k, val in zip(keys, combo):
                f[k] = val
            frames.append(f)
            labels.append(", ".join(f"{k}={val}" for k, val in zip(keys, combo)))
        return frames, labels

    def _single_row(self, at: str, factor_stat: str) -> pd.DataFrame:
        """Build a one-row representative frame in data space.

        Numeric columns get the value implied by ``at``
        (mean / median / 0). Factor / categorical columns get the value
        implied by ``factor_stat`` (modal level for ``"mode"``, the patsy
        reference level for ``"zero"``).

        ``factor_stat="mean"`` is not handled here — the caller routes
        that through a design-matrix-collapse path so that factor dummies
        end up at their observed proportions, which has no representation
        in data space.
        """
        row = {}
        for c in self.data.columns:
            col = self.data[c]
            is_numeric = (
                pd.api.types.is_numeric_dtype(col)
                and not pd.api.types.is_bool_dtype(col)
            )
            if is_numeric:
                if at == "mean":
                    row[c] = col.mean()
                elif at == "median":
                    row[c] = col.median()
                elif at == "zero":
                    row[c] = 0.0
                else:
                    raise ValueError(
                        f"_single_row called with at={at!r}; only "
                        f"'mean'/'median'/'zero' are valid here."
                    )
            elif factor_stat == "mode":
                try:
                    row[c] = col.mode().iloc[0]
                except Exception:
                    row[c] = col.iloc[0]
            elif factor_stat == "zero":
                row[c] = self._factor_reference_level(c)
            else:
                raise ValueError(
                    f"_single_row called with factor_stat={factor_stat!r}; "
                    f"the 'mean' path uses design-matrix collapse instead."
                )
        return pd.DataFrame([row])

    def _factor_reference_level(self, col: str) -> object:
        """Return the patsy reference level for a categorical column.

        Probes one-row designs with ``col`` set to each observed level
        (other variables held at simple defaults) and picks the level
        whose 1-row design has the minimum L1 norm — i.e. the one that
        contributes nothing to the linear predictor under default
        contrasts. This matches the level that ``get_margeff(at='zero')``
        implicitly picks when it sets the design row to zero.

        Falls back to the alphabetically first observed level when not
        in formula mode or when probing fails (custom Link subclasses,
        non-standard ``DesignInfo``, etc.).
        """
        levels = self.data[col].dropna().unique().tolist()
        try:
            levels = sorted(levels)
        except TypeError:
            pass
        if not levels:
            return self.data[col].iloc[0]
        if self._raw_mode or self.design_info is None or len(levels) < 2:
            return levels[0]
        probe = pd.DataFrame([{
            c: self._probe_default(c) for c in self.data.columns
        }])
        best_lvl, best_norm = levels[0], None
        for lvl in levels:
            p = probe.copy()
            p[col] = lvl
            try:
                X = self._build_exog(p)
            except Exception:
                continue
            norm = float(np.abs(np.asarray(X)).sum())
            if best_norm is None or norm < best_norm:
                best_norm = norm
                best_lvl = lvl
        return best_lvl

    def _probe_default(self, col: str):
        """Default probe value: 0 for numerics, first sorted level otherwise."""
        s = self.data[col]
        if (pd.api.types.is_numeric_dtype(s)
                and not pd.api.types.is_bool_dtype(s)):
            return 0.0
        u = s.dropna().unique().tolist()
        if not u:
            return s.iloc[0]
        try:
            return sorted(u)[0]
        except TypeError:
            return u[0]

    def _resolve_base(self, at: str, factor_stat: str) -> tuple[pd.DataFrame, bool]:
        """Return ``(base_frame, collapse)``.

        ``collapse=True`` means the caller should apply
        ``X.mean(axis=0, keepdims=True)`` to the design matrix after
        building it — this is how factor dummies become their observed
        proportions (Stata's ``atmeans`` semantics).

        ``collapse=False`` means ``base_frame`` is already a single
        representative row, with both numeric and factor columns
        materialized in data space.
        """
        if at == "overall":
            return self.data, False

        if factor_stat == "mean":
            # Modify numerics in data space; leave factors observed and
            # let X.mean later pick up their proportions.
            f = self.data.copy()
            for c in f.columns:
                col = f[c]
                is_numeric = (
                    pd.api.types.is_numeric_dtype(col)
                    and not pd.api.types.is_bool_dtype(col)
                )
                if not is_numeric:
                    continue
                if at == "median":
                    f[c] = col.median()
                elif at == "zero":
                    f[c] = 0.0
                # at == "mean": X.mean handles it without touching the data.
            return f, True

        return self._single_row(at, factor_stat), False

    def _collapse_numerics_to_mean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace observed (row-varying) numeric columns with their training mean.

        Used by the count and discrete dydx paths under ``collapse=True``
        to give a single representative profile in *data space*, so that
        derived columns like ``I(x**2)`` evaluate to ``mean(x)**2`` rather
        than ``mean(x**2)``. Columns that are already constant in the
        frame (e.g. fixed by ``atexog``) are left alone.
        """
        out = frame.copy()
        for c in self.data.columns:
            if c not in out.columns:
                continue
            col = self.data[c]
            if not (pd.api.types.is_numeric_dtype(col)
                    and not pd.api.types.is_bool_dtype(col)):
                continue
            if out[c].nunique(dropna=False) <= 1:
                continue
            out[c] = col.mean()
        return out

    @staticmethod
    def _check_at(at: str) -> None:
        if at not in ("overall", "mean", "median", "zero"):
            raise ValueError(
                f"at must be 'overall', 'mean', 'median', or 'zero', "
                f"got {at!r}"
            )

    @staticmethod
    def _check_factor_stat(factor_stat: str) -> None:
        if factor_stat not in ("mean", "mode", "zero"):
            raise ValueError(
                f"factor_stat must be 'mean', 'mode', or 'zero', "
                f"got {factor_stat!r}"
            )

    @staticmethod
    def _default_factor_stat(at: str) -> str:
        """Pick a sensible factor_stat when the user didn't supply one."""
        return {
            "overall": "mean",  # ignored — no collapse happens
            "mean":    "mean",  # Stata default: design-matrix proportions
            "median":  "mode",  # median is undefined for categoricals
            "zero":    "zero",  # reference level — matches at='zero' semantics
        }[at]

    # ---- the delta-method worker ----

    def _delta(
        self,
        statistic: Callable[[np.ndarray], np.ndarray],
        labels: Sequence[str],
        stat_name: str,
        jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> MarginsResult:
        beta = self.params
        est_val = statistic(beta)
        est = np.atleast_1d(np.asarray(est_val, dtype=float)).ravel()

        if jac is not None:
            J = np.atleast_2d(np.asarray(jac(beta), dtype=float))
        else:
            # We must pass the user-provided 'statistic' directly to the
            # numerical differentiator.
            J = _central_jacobian(statistic, beta)

        V = J @ self.cov @ J.T
        return MarginsResult(
            estimate=est, vcov=V, labels=labels,
            level=self.level, df=self.df, stat_name=stat_name,
        )

    # ======================================================================
    # Public API: adjusted predictions
    # ======================================================================

    def predict(
        self,
        at: str = "overall",
        atexog: Optional[Mapping[str, object]] = None,
        factor_stat: Optional[str] = None,
    ) -> MarginsResult:
        """Adjusted predictions (expected outcome on the response scale).

        Parameters
        ----------
        at : {"overall", "mean", "median", "zero"}, default "overall"
            Profile at which to evaluate. Mirrors statsmodels'
            ``get_margeff(at=...)``.

            - ``"overall"`` (default): average over the data — Average
              Adjusted Prediction (AAP).
            - ``"mean"``: evaluate at the mean of each covariate. With
              ``factor_stat="mean"`` (the default for this case), this
              collapses the design matrix column-wise so factor dummies
              become their observed proportions — Stata's
              ``margins, atmeans`` and statsmodels' ``get_margeff(at='mean')``.
            - ``"median"``: evaluate at the column-wise median of each
              numeric covariate. Factors default to their modal level.
            - ``"zero"``: evaluate at zero for all numeric covariates.
              Factors default to their reference (first observed) level —
              matching statsmodels' ``get_margeff(at='zero')``.
        atexog : mapping, optional
            Variable name -> scalar or list. Each variable is held fixed
            at the given value(s); all others go to their ``at``-determined
            values. Lists are expanded as a cartesian product. Same role
            as statsmodels' ``atexog`` (but keyed by name, not column
            index — strictly an upgrade in UX).
        factor_stat : {"mean", "mode", "zero"}, optional
            How factors are handled when ``at != "overall"``. If ``None``
            (default), chosen by ``at``: ``"mean"`` for ``at="mean"``,
            ``"mode"`` for ``at="median"``, ``"zero"`` for ``at="zero"``.

            - ``"mean"``: factors become their observed proportions
              (design-matrix collapse). The Stata default. Only fully
              meaningful with ``at="mean"`` but also defined for
              ``"median"``/``"zero"`` (numerics held at the requested
              point, factors at observed proportions).
            - ``"mode"``: factors at their modal level — a "typical
              individual" rather than a fictional fractional one.
            - ``"zero"``: factors at their reference (first observed)
              level.

        Returns
        -------
        MarginsResult

        Notes
        -----
        ============================  ============================================
        Statistic                     Call
        ============================  ============================================
        AAP  (avg adj. prediction)    ``predict()``
        APM  (pred. at means)         ``predict(at="mean")``
        APR  (pred. at rep. values)   ``predict(atexog={"x1": [0,1,2]})``
        APR with others at means      ``predict(at="mean", atexog={"x1": [0,1,2]})``
        ============================  ============================================
        """
        self._check_at(at)
        if factor_stat is None:
            factor_stat = self._default_factor_stat(at)
        self._check_factor_stat(factor_stat)

        base, collapse = self._resolve_base(at, factor_stat)
        frames, at_labels = self._expand_at(base, atexog)

        stat_name = "prediction"
        if atexog is None:
            if at == "overall":
                labels = ["AAP"]
            elif at == "mean":
                labels = ["APM"]
            else:
                labels = [f"AP @ {at}"]
        else:
            labels = at_labels

        Xs = []
        for f in frames:
            X = self._build_exog(f)
            if collapse:
                X = X.mean(axis=0, keepdims=True)
            Xs.append(X)

        def statistic(beta: np.ndarray) -> np.ndarray:
            out = np.empty(len(Xs))
            for i, X in enumerate(Xs):
                out[i] = float(np.mean(self._predict(beta, X)))
            return out

        fprime = self._link_deriv()
        if fprime is not None:
            def jac(beta: np.ndarray) -> np.ndarray:
                J = np.empty((len(Xs), beta.size))
                for i, X in enumerate(Xs):
                    J[i, :] = self._grad_mean_predict(X, beta, fprime)
                return J
        else:
            jac = None

        return self._delta(statistic, labels=labels, stat_name=stat_name, jac=jac)

    # ======================================================================
    # Public API: marginal effects
    # ======================================================================

    def dydx(
        self,
        variable: Union[str, List[str]],
        at: str = "overall",
        atexog: Optional[Mapping[str, object]] = None,
        discrete: Optional[bool] = None,
        count: bool = False,
        step: Optional[float] = None,
        reference: Optional[object] = None,
        factor_stat: Optional[str] = None,
        method: str = "dydx",
    ) -> MarginsResult:
        """Marginal effect of ``variable`` on the response.

        Parameters
        ----------
        variable : str, list of str, or "*"
            Column name(s) in the fitting data frame. If ``"*"``, use all
            non-response columns in the data frame.
        at, atexog, factor_stat : see :meth:`predict`.
        discrete : bool, optional
            If True, use discrete changes (level vs reference).
            If False, use a numerical derivative.
            If None (default), auto-detect per-variable:
              - non-numeric or boolean dtype  -> discrete
              - numeric with ≤ 2 unique values -> discrete
              - else                           -> continuous derivative.
        count : bool, default False
            If True, treat as a numeric variable with a unit increment (x -> x+1).
            Useful for integer-valued covariates.
        step : float, optional
            Step size for the continuous derivative. Default is
            ``(eps**(1/3)) * max(std(x), 1)`` (central-difference sweet spot).
        reference : scalar, optional
            Reference level for discrete effects; defaults to the smallest.
        method : {"dydx", "dyex", "eyex", "eydx"}, default ``"dydx"``
            What to compute, per observation, before averaging:

            - ``"dydx"`` : :math:`\\partial y_i / \\partial x_{ij}`
              (level change). The default; only path with an analytic
              outer Jacobian.
            - ``"dyex"`` : :math:`(\\partial y_i / \\partial x_{ij})\\,x_i`
              (semi-elasticity, :math:`dy/d(\\ln x)`).
            - ``"eyex"`` : :math:`(\\partial y_i / \\partial x_{ij})\\,x_i / y_i`
              (full elasticity).
            - ``"eydx"`` : :math:`(\\partial y_i / \\partial x_{ij}) / y_i`
              (semi-elasticity, :math:`d(\\ln y)/dx`).

            Only valid for continuous variables; raises if the variable
            is (auto- or explicitly) discrete and ``method != "dydx"``.
            Elasticity methods always go through finite differences;
            ``analytic=True`` on the constructor only affects ``"dydx"``.

        Returns
        -------
        MarginsResult

        Notes
        -----
        ============================  =====================================
        Statistic                     Call
        ============================  =====================================
        AME (average marginal eff.)   ``dydx("x1")``
        MEM (marginal eff. at means)  ``dydx("x1", at="mean")``
        MER (eff. at rep. values)     ``dydx("x1", atexog={"x2": [0, 1]})``
        At medians                    ``dydx("x1", at="median")``
        At zero                       ``dydx("x1", at="zero")``
        Multi-variable                ``dydx(["x1", "x2"])``
        All RHS columns               ``dydx("*")``
        With unit increment           ``dydx("kids", count=True)``
        ============================  =====================================
        """
        if method not in _METHOD_META:
            raise ValueError(
                f"method must be one of {sorted(_METHOD_META)}, got {method!r}"
            )
        self._check_at(at)
        if factor_stat is None:
            factor_stat = self._default_factor_stat(at)
        self._check_factor_stat(factor_stat)

        if variable == "*":
            # All columns minus the response. This may include columns not
            # used in the model, which matches requested v1 behavior.
            variable = self.data.columns.difference(
                [self.model.endog_names]
            ).tolist()
        if isinstance(variable, str):
            variables = [variable]
        else:
            variables = list(variable)

        if self._raw_mode:
            valid = list(self.model.exog_names)
            for v in variables:
                if v not in valid:
                    raise ValueError(
                        f"Variable {v!r} not found in model's exog columns. "
                        f"Valid names: {valid}"
                    )
                if v in ("const", "Intercept") and discrete is None:
                    raise ValueError(
                        f"Cannot compute marginal effect for the constant {v!r}. "
                        "If you intended a discrete change, set discrete=True."
                    )

        base, collapse = self._resolve_base(at, factor_stat)
        frames, at_labels = self._expand_at(base, atexog)

        if count:
            if discrete is True:
                raise ValueError(
                    "count=True is itself a discrete contrast; do not also pass "
                    "discrete=True"
                )
            if method != "dydx":
                raise ValueError("count=True is only valid with method='dydx'")

        parts = []
        for v in variables:
            col = self.data[v]
            if count:
                if not pd.api.types.is_numeric_dtype(col):
                    raise ValueError(
                        f"count=True requires a numeric variable; {v!r} has "
                        f"dtype {col.dtype}"
                    )
                if not pd.api.types.is_integer_dtype(col):
                    import warnings
                    warnings.warn(
                        f"count=True applied to a float column ({v!r}); "
                        "the contrast x -> x+1 may not be physically meaningful.",
                        RuntimeWarning
                    )
                res_parts = self._dydx_count_components(
                    v, frames, at_labels, collapse=collapse
                )
                parts.append(res_parts)
                continue

            v_discrete = discrete
            if v_discrete is None:
                if (pd.api.types.is_bool_dtype(col)
                        or not pd.api.types.is_numeric_dtype(col)):
                    v_discrete = True
                elif pd.api.types.is_numeric_dtype(col) and col.dropna().nunique() <= 2:
                    v_discrete = True
                else:
                    v_discrete = False

            if v_discrete and method != "dydx":
                raise ValueError(
                    f"method={method!r} requires a continuous variable; "
                    f"got discrete=True for {v!r}. "
                    f"Pass discrete=False to override."
                )

            if v_discrete:
                res_parts = self._dydx_discrete_components(
                    v, frames, at_labels, reference=reference,
                    collapse=collapse,
                )
                parts.append(res_parts)
            else:
                res_parts = self._dydx_continuous_components(
                    v, frames, at_labels, step=step, at=at,
                    collapse=collapse, method=method,
                )
                parts.append(res_parts)

        def make_stacked_stat(p_list, m_obj):
            def stacked_stat(beta: np.ndarray) -> np.ndarray:
                res = []
                for fn, _, _, _ in p_list:
                    # fn is the statistic function for one variable. 
                    # We MUST pass beta to it.
                    val = np.atleast_1d(np.asarray(fn(beta), dtype=float)).ravel()
                    res.append(val)
                return np.concatenate(res)
            return stacked_stat

        def make_stacked_jac(p_list):
            # All-or-nothing on analytic Jacobian. Mixing analytic and FD blocks
            # in one delta call is not supported in v1.
            if any(p[1] is None for p in p_list):
                return None

            def stacked_jac(beta: np.ndarray) -> np.ndarray:
                res = []
                for _, j, _, _ in p_list:
                    # j is the jacobian function for one variable.
                    # We MUST pass beta to it.
                    val = np.atleast_2d(np.asarray(j(beta), dtype=float))
                    res.append(val)
                return np.vstack(res)
            return stacked_jac

        statistic = make_stacked_stat(parts, self)
        jac_fn = make_stacked_jac(parts)

        labels = []
        for _, _, p_labels, _ in parts:
            labels.extend(p_labels)

        names = {p[3] for p in parts}
        stat_name = names.pop() if len(names) == 1 else "mixed"

        return self._delta(statistic, labels=labels, stat_name=stat_name, jac=jac_fn)

    # ----- continuous case (numerical derivative) ---------------------------

    def _dydx_continuous_components(
        self,
        variable: str,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        step: Optional[float],
        at: str,
        collapse: bool = False,
        method: str = "dydx",
    ):
        if step is None:
            sd = float(self.data[variable].std(ddof=0))
            scale = max(sd, abs(float(self.data[variable].mean())), 1.0)
            step = (np.finfo(float).eps ** (1.0 / 3.0)) * scale
        h = float(step)

        # For ``eyex``/``eydx`` we need ``y_i = f(eta_i)`` evaluated at
        # the *unperturbed* design row, so we keep an extra ``Xs_orig``.
        need_y = method in ("eyex", "eydx")

        Xs_plus: List[np.ndarray] = []
        Xs_minus: List[np.ndarray] = []
        Xs_orig: List[np.ndarray] = []
        x_vals: List[np.ndarray] = []
        for frame in frames:
            x_col = np.asarray(frame[variable], dtype=float)
            f_plus = frame.copy()
            f_minus = frame.copy()
            f_plus[variable] = x_col + h
            f_minus[variable] = x_col - h
            Xp = self._build_exog(f_plus)
            Xm = self._build_exog(f_minus)
            Xo = self._build_exog(frame) if need_y else None
            if collapse:
                Xp = Xp.mean(axis=0, keepdims=True)
                Xm = Xm.mean(axis=0, keepdims=True)
                if Xo is not None:
                    Xo = Xo.mean(axis=0, keepdims=True)
                x_col = np.array([x_col.mean()])
            Xs_plus.append(Xp)
            Xs_minus.append(Xm)
            if Xo is not None:
                Xs_orig.append(Xo)
            x_vals.append(x_col)

        def statistic(beta: np.ndarray) -> np.ndarray:
            out = np.empty(len(Xs_plus))
            for i in range(len(Xs_plus)):
                Xp, Xm = Xs_plus[i], Xs_minus[i]
                dydx_i = (self._predict(beta, Xp) - self._predict(beta, Xm)) / (2.0 * h)
                if method == "dydx":
                    contrib = dydx_i
                elif method == "dyex":
                    contrib = dydx_i * x_vals[i]
                else:  # eyex / eydx
                    y_i = self._predict(beta, Xs_orig[i])
                    if method == "eyex":
                        contrib = dydx_i * x_vals[i] / y_i
                    else:  # eydx
                        contrib = dydx_i / y_i
                out[i] = np.mean(contrib)
            return out

        # Analytic Jacobian only for the level case ("dydx"); the three
        # elasticities go through FD via _central_jacobian. Mixing FD
        # for the inner ∂y/∂β with analytic for the elasticity scaling
        # would require quotient-rule code with no numerical benefit
        # over straight FD, so we keep the analytic path narrow.
        jac = None
        if method == "dydx" and self.analytic:
            fprime = self._link_deriv()
            if fprime is not None:
                def jac(beta: np.ndarray) -> np.ndarray:
                    J = np.empty((len(Xs_plus), beta.size))
                    for i, (Xp, Xm) in enumerate(zip(Xs_plus, Xs_minus)):
                        gp = self._grad_mean_predict(Xp, beta, fprime)
                        gm = self._grad_mean_predict(Xm, beta, fprime)
                        J[i, :] = (gp - gm) / (2.0 * h)
                    return J

        meta = _METHOD_META[method]
        prefix, stat_name = meta["prefix"], meta["stat_name"]
        suffix_map = {
            "overall": "",
            "mean":    " (at means)",
            "median":  " (at medians)",
            "zero":    " (at zero)",
        }
        if len(frames) == 1 and at_labels[0] == "":
            labels = [f"{prefix}{variable}{suffix_map[at]}"]
        else:
            labels = [f"{prefix}{variable} | {lab}" if lab else f"{prefix}{variable}"
                      for lab in at_labels]
        return (statistic, jac, labels, stat_name)

    def _dydx_continuous(
        self,
        variable: str,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        step: Optional[float],
        at: str,
        collapse: bool = False,
        method: str = "dydx",
    ) -> MarginsResult:
        stat, jac, labels, stat_name = self._dydx_continuous_components(
            variable, frames, at_labels, step, at, collapse, method
        )
        return self._delta(stat, labels=labels, stat_name=stat_name, jac=jac)

    # ----- count case (unit increment x -> x+1) ----------------------------

    def _dydx_count_components(
        self,
        variable: str,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        collapse: bool = False,
    ):
        # Stata-style MEM-count semantics: when ``collapse`` is on
        # (factor_stat="mean"), the contrast should be evaluated as
        # f(η at the mean profile, x+1) − f(η at the mean profile, x),
        # i.e. derived columns like ``I(x**2)`` use ``mean(x)**2`` —
        # not ``mean(x**2)``. The "perturb each row, then column-mean"
        # approach drops that h² term only for h→0 (the FD continuous
        # path); for the unit increment of count it leaves a residual
        # of order Var(x)·β. So pre-collapse non-fixed numerics here.
        if collapse:
            frames = [self._collapse_numerics_to_mean(fr) for fr in frames]

        Xs_p = []
        Xs_m = []
        labels = []

        for frame, at_lab in zip(frames, at_labels):
            # x -> x + 1
            f_p = frame.copy()
            f_p[variable] = f_p[variable] + 1
            f_m = frame.copy()

            Xp = self._build_exog(f_p)
            Xm = self._build_exog(f_m)

            if collapse:
                Xp = Xp.mean(axis=0, keepdims=True)
                Xm = Xm.mean(axis=0, keepdims=True)

            Xs_p.append(Xp)
            Xs_m.append(Xm)

            base = f"{variable} (count)"
            labels.append(f"{base} | {at_lab}" if at_lab else base)

        def statistic(beta: np.ndarray) -> np.ndarray:
            out = np.empty(len(Xs_p))
            for i, (Xp, Xm) in enumerate(zip(Xs_p, Xs_m)):
                out[i] = np.mean(self._predict(beta, Xp)) - np.mean(self._predict(beta, Xm))
            return out

        jac = None
        if self.analytic:
            fprime = self._link_deriv()
            if fprime is not None:
                def jac(beta: np.ndarray) -> np.ndarray:
                    J = np.empty((len(Xs_p), beta.size))
                    for i, (Xp, Xm) in enumerate(zip(Xs_p, Xs_m)):
                        gp = self._grad_mean_predict(Xp, beta, fprime)
                        gm = self._grad_mean_predict(Xm, beta, fprime)
                        J[i, :] = gp - gm
                    return J

        return (statistic, jac, labels, "dy/dx")

    # ----- discrete case (contrast vs reference level) ---------------------

    def _dydx_discrete_components(
        self,
        variable: str,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        reference: Optional[object],
        collapse: bool = False,
    ):
        # See ``_dydx_count_components`` for why non-fixed numerics are
        # collapsed to their training mean here under ``collapse``.
        if collapse:
            frames = [self._collapse_numerics_to_mean(fr) for fr in frames]

        levels = self.data[variable].dropna().unique().tolist()
        try:
            levels = sorted(levels)
        except TypeError:
            pass
        if len(levels) < 2:
            raise ValueError(f"variable {variable!r} has <2 unique levels")
        if reference is None:
            reference = levels[0]
        others = [lv for lv in levels if lv != reference]

        Xs_lvl: List[np.ndarray] = []
        Xs_ref: List[np.ndarray] = []
        labels: List[str] = []
        for frame, at_lab in zip(frames, at_labels):
            f_ref = frame.copy(); f_ref[variable] = reference
            Xref = self._build_exog(f_ref)
            if collapse:
                Xref = Xref.mean(axis=0, keepdims=True)
            for lvl in others:
                f_lvl = frame.copy(); f_lvl[variable] = lvl
                Xl = self._build_exog(f_lvl)
                if collapse:
                    Xl = Xl.mean(axis=0, keepdims=True)
                Xs_lvl.append(Xl)
                Xs_ref.append(Xref)
                base = f"{variable}: {lvl} vs {reference}"
                labels.append(f"{base} | {at_lab}" if at_lab else base)

        def statistic(beta: np.ndarray) -> np.ndarray:
            out = np.empty(len(Xs_lvl))
            for i, (Xl, Xr) in enumerate(zip(Xs_lvl, Xs_ref)):
                out[i] = np.mean(self._predict(beta, Xl)) - np.mean(self._predict(beta, Xr))
            return out

        jac = None
        if self.analytic:
            fprime = self._link_deriv()
            if fprime is not None:
                def jac(beta: np.ndarray) -> np.ndarray:
                    J = np.empty((len(Xs_lvl), beta.size))
                    for i, (Xl, Xr) in enumerate(zip(Xs_lvl, Xs_ref)):
                        gl = self._grad_mean_predict(Xl, beta, fprime)
                        gr = self._grad_mean_predict(Xr, beta, fprime)
                        J[i, :] = gl - gr
                    return J

        return (statistic, jac, labels, "contrast")

    def _dydx_discrete(
        self,
        variable: str,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        reference: Optional[object],
        collapse: bool = False,
    ) -> MarginsResult:
        stat, jac, labels, stat_name = self._dydx_discrete_components(
            variable, frames, at_labels, reference, collapse
        )
        return self._delta(stat, labels=labels, stat_name=stat_name, jac=jac)

    # ======================================================================
    # Public API: difference-in-differences
    # ======================================================================

    def did(
        self,
        group: str,
        condition: str,
        group_levels: Optional[Sequence] = None,
        condition_levels: Optional[Sequence] = None,
        at: str = "overall",
        atexog: Optional[Mapping[str, object]] = None,
        factor_stat: Optional[str] = None,
    ) -> "DiDResult":
        """Difference-in-differences on the response scale.

        Sets up a 2×2 grid (``group`` × ``condition``), computes adjusted
        predictions for all four cells (averaging over other covariates
        per ``at`` / ``atexog``), and returns the cell means, both simple
        effects, and the DiD — all sharing the same delta-method joint
        covariance.

        Parameters
        ----------
        group, condition : str
            Names of the two binary-ish factors. The "DiD" is the
            difference in the group-effect between the two condition
            levels, on the response scale.
        group_levels, condition_levels : sequence of length 2, optional
            Which two levels to use (reference first, treated second).
            Default: the two smallest observed levels.
        at, atexog, factor_stat : see :meth:`predict`.

        Returns
        -------
        DiDResult
            Has ``.cells`` (4 predictions), ``.simple_effects`` (2
            group-effects, one per condition level), and ``.did`` (the
            1-row DiD). All three are ``MarginsResult`` instances.

        Notes
        -----
        On a *linear* model with identity link, the DiD here equals the
        coefficient on the ``group:condition`` interaction. On a
        nonlinear model (logit, probit, Poisson…), it does **not** —
        that's exactly the Ai & Norton (2003) issue, and why a
        response-scale DiD with a delta-method SE is the right thing to
        report.
        """
        g = (list(group_levels) if group_levels is not None
             else self._default_two_levels(group))
        c = (list(condition_levels) if condition_levels is not None
             else self._default_two_levels(condition))
        if len(g) != 2 or len(c) != 2:
            raise ValueError("DiD requires exactly 2 levels for each factor.")

        atexog_full = dict(atexog) if atexog else {}
        for k, v in atexog_full.items():
            # Lists/arrays would expand the cell grid past 4 and break
            # the hard-coded contrast matrix below. Surface a clear error
            # rather than letting ``cells.contrast`` fail downstream.
            if (not np.isscalar(v)) and not isinstance(v, (str, bytes)):
                raise ValueError(
                    f"did() requires scalar atexog values; got list/array "
                    f"for {k!r}. Loop over values yourself if you need a "
                    f"DiD per profile."
                )
        # Insert group first, condition second -> itertools.product iterates
        # with condition varying fastest: (g0,c0),(g0,c1),(g1,c0),(g1,c1)
        atexog_full[group] = g
        atexog_full[condition] = c
        cells = self.predict(at=at, atexog=atexog_full, factor_stat=factor_stat)

        # Rewrite the cell labels to the compact "(group=A, condition=0)" form
        cells.labels = [
            f"{group}={gv}, {condition}={cv}"
            for gv in g for cv in c
        ]

        # Contrast matrix:
        #   row 0: simple effect of group at condition=c0  -> -m0 + m2
        #   row 1: simple effect of group at condition=c1  -> -m1 + m3
        #   row 2: DiD = (m3 - m1) - (m2 - m0) = m0 - m1 - m2 + m3
        C_all = np.array([
            [-1.0, 0.0, 1.0, 0.0],
            [ 0.0,-1.0, 0.0, 1.0],
            [ 1.0,-1.0,-1.0, 1.0],
        ])
        all_contrasts = cells.contrast(
            C_all,
            labels=[
                f"{group}: {g[1]} vs {g[0]} | {condition}={c[0]}",
                f"{group}: {g[1]} vs {g[0]} | {condition}={c[1]}",
                f"DiD: {group}({g[1]}-{g[0]}) × {condition}({c[1]}-{c[0]})",
            ],
            name="estimate",
        )
        # Slice back apart so users can grab each piece
        simple = MarginsResult(
            estimate=all_contrasts.estimate[:2],
            vcov=all_contrasts.vcov[:2, :2],
            labels=all_contrasts.labels[:2],
            level=self.level, df=self.df, stat_name="simple effect",
        )
        did_ = MarginsResult(
            estimate=all_contrasts.estimate[2:3],
            vcov=all_contrasts.vcov[2:3, 2:3],
            labels=[all_contrasts.labels[2]],
            level=self.level, df=self.df, stat_name="DiD",
        )
        return DiDResult(cells=cells, simple_effects=simple, did=did_,
                         joint=all_contrasts)

    def _default_two_levels(self, col: str) -> list:
        vals = self.data[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except TypeError:
            pass
        if len(vals) < 2:
            raise ValueError(f"{col!r} has fewer than 2 unique values")
        if len(vals) > 2:
            # Take smallest and largest, to be explicit
            vals = [vals[0], vals[-1]]
        return vals


# ---------------------------------------------------------------------------
# DiDResult — small container bundling cells / simple effects / DiD
# ---------------------------------------------------------------------------

class DiDResult:
    """Bundle of results from :meth:`Margins.did`.

    Attributes
    ----------
    cells : MarginsResult
        The four adjusted predictions.
    simple_effects : MarginsResult
        Group effect evaluated at each level of the condition.
    did : MarginsResult
        The single difference-in-differences estimate.
    joint : MarginsResult
        All three contrasts (2 simple effects + DiD) sharing joint vcov,
        useful if you want to jointly test e.g. ``simple_effects = did = 0``.
    """

    def __init__(self, cells: MarginsResult, simple_effects: MarginsResult,
                 did: MarginsResult, joint: MarginsResult):
        self.cells = cells
        self.simple_effects = simple_effects
        self.did = did
        self.joint = joint

    def summary(self) -> pd.DataFrame:
        """Return the full DiD summary as one concatenated DataFrame."""
        parts = [
            self.cells.summary().assign(**{"": "cell"}),
            self.simple_effects.summary().assign(**{"": "simple"}),
            self.did.summary().assign(**{"": "DiD"}),
        ]
        return pd.concat(parts)

    def __repr__(self) -> str:
        def banner(title: str) -> str:
            return "\n" + title + "\n" + "-" * len(title)
        return "\n".join([
            banner("Cell predictions"),
            repr(self.cells),
            banner("Simple effects"),
            repr(self.simple_effects),
            banner("Difference-in-differences"),
            repr(self.did),
        ])
