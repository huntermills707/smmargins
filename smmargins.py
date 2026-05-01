r"""
Stata-style ``margins`` for StatsModels with delta-method standard errors.

For a fitted model with parameter vector :math:`\hat\beta`, estimated
covariance :math:`\widehat V(\hat\beta)`, and a (possibly vector-valued)
statistic :math:`g(\beta)`, the delta method gives

.. math::

    \widehat{\operatorname{Var}}\bigl[g(\hat\beta)\bigr]
    \;\approx\; G\,\widehat V\,G^\top,
    \qquad
    G = \left.\frac{\partial g}{\partial \beta}\right|_{\hat\beta}.

Statistics
----------

This module supports every statistic discussed in Richard Williams'
*Using the margins command* notes (Margins01):

=========  ========================================================
AAP        :math:`(1/n)\sum_i f(x_i^\top\beta)`  (avg adj. prediction)
APM        :math:`f(\bar x^\top\beta)`  (prediction at means)
APR        AAP with some :math:`x` fixed at representative values
AME        :math:`(1/n)\sum_i \partial f(x_i^\top\beta)/\partial x_{ij}`
MEM        :math:`\partial f(\bar x^\top\beta)/\partial x_j`  (ME at means)
MER        AME with some :math:`x` held at representative values
=========  ========================================================

where :math:`f` is the response-scale map (identity for OLS, inverse
link for GLMs, etc.).

Patsy integration
-----------------

Rather than hand-differentiating each model's link / linear-predictor
combination, we build every statistic as a function of :math:`\beta`
using StatsModels' own ``model.predict(params, exog)``.  Patsy does the
heavy lifting of propagating perturbations through interactions,
polynomials, splines, and categorical encodings: when the user says
"perturb ``x1``", we perturb the *column of the data frame* and let
``patsy.dmatrix(design_info, ...)`` rebuild the design matrix.  This
way ``I(x1**2)``, ``x1:x2``, ``C(group)``, ``bs(x1, df=4)`` all update
correctly and automatically.

Examples
--------
Fit a logit, then ask for the average marginal effect of ``age`` and the
adjusted-prediction profile across ``age`` for each sex::

    import numpy as np, pandas as pd
    import statsmodels.formula.api as smf
    from smmargins import Margins

    fit = smf.logit("voted ~ age + income + C(educ) + female + age:female",
                    data=df).fit()
    M = Margins(fit)

    M.dydx("age")                                  # AME of age
    M.dydx("age", at="mean")                       # MEM
    M.dydx("age", atexog={"female": [0, 1]})       # MER, by sex
    M.predict(atexog={"age": list(range(20, 91, 10)),
                      "female": [0, 1]})           # plottable table

For a 2x2 difference-in-differences on the response (probability) scale::

    res = M.did("group", "preexist_Y",
                group_levels=["A", "B"], condition_levels=[0, 1],
                atexog={"age": 60, "female": 0})
    print(res)            # cells, simple effects, DiD
    res.did.estimate      # the DiD point estimate
    res.cells.vcov        # 4x4 joint covariance of the cell predictions

See ``demo_margins.py`` and ``demo_did.py`` for end-to-end walkthroughs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
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


# ---------------------------------------------------------------------------
# Marginal-effect method registry
# ---------------------------------------------------------------------------
#
# Registry mapping method names to their column-prefix and statistic-name.
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
    r"""Container for margin estimates with delta-method standard errors.

    Holds the point estimate, standard error, confidence interval,
    p-value, and the full delta-method covariance matrix of a vector of
    margin statistics (e.g. adjusted predictions or marginal effects).

    For a fitted model with parameter vector :math:`\hat\beta`, estimated
    covariance :math:`\widehat V(\hat\beta)`, and a (possibly
    vector-valued) statistic :math:`g(\beta)`, the delta method gives

    .. math::

        \widehat{\mathrm{Var}}[g(\hat\beta)] \approx
        G \, \widehat V \, G^\top,

    where :math:`G = \partial g / \partial \beta|_{\hat\beta}`.

    Parameters
    ----------
    estimate : ndarray
        Point estimates.
    vcov : ndarray
        Full delta-method covariance of the estimates (useful for joint
        tests).
    labels : sequence of str, optional
        Row labels. Defaults to ``["m1", "m2", ...]``.
    level : float
        Confidence level for intervals (default 0.95).
    df : int or None
        Residual degrees of freedom; if set, uses t-distribution for
        p-values and CIs. Otherwise uses N(0,1).
    stat_name : str
        Column name for the statistic in :meth:`summary`.

    Attributes
    ----------
    estimate : ndarray
    se       : ndarray
    labels   : list[str]
    vcov     : ndarray
        Full delta-method covariance of the estimates (useful for joint
        tests).
    level    : float
    df       : int or None
    ci_lower : ndarray
    ci_upper : ndarray
    pvalue   : ndarray
    zstat    : ndarray
    _test_name : str
        ``"z"`` or ``"t"`` depending on whether the normal or
        t-distribution is used.

    Examples
    --------
    Most users obtain a ``MarginsResult`` by calling ``Margins.predict`` or
    ``Margins.dydx`` rather than constructing one directly. Once you have
    a result, the typical workflow is to inspect ``.summary()`` or pull
    fields off it::

        res = M.dydx(["x1", "x2"])
        res.summary()                # tidy table of estimates / SE / CI
        res.estimate                 # ndarray of point estimates
        res.vcov                     # joint delta-method covariance

    Forming linear contrasts of the estimates uses the *joint* covariance
    that is already on the result, so no additional differentiation is
    required. For example, the difference between the AMEs of ``x1`` and
    ``x2``::

        res.contrast([1.0, -1.0], labels=["x1 - x2"])

    Constructing one directly (mostly useful in tests)::

        >>> import numpy as np
        >>> from smmargins import MarginsResult
        >>> est = np.array([1.0, 2.0])
        >>> vcov = np.array([[0.25, 0.10],
        ...                  [0.10, 0.16]])
        >>> res = MarginsResult(est, vcov, labels=["m1", "m2"])
        >>> res.se.round(2).tolist()
        [0.5, 0.4]
        >>> res.contrast([1.0, -1.0], labels=["m1 - m2"]).estimate.tolist()
        [-1.0]

    See Also
    --------
    Margins.predict
    Margins.dydx
    Margins.did
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
        r"""Form linear contrasts of the estimates, with delta-method SEs.

        If the estimates have joint covariance :math:`V_m`, any linear
        combination :math:`C m` has covariance :math:`C V_m C^\top`, and
        :math:`V_m` was already built from the delta method on
        :math:`\beta`, so this is exact under the same approximation
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

        Returns
        -------
        MarginsResult

        Raises
        ------
        ValueError
            If the number of contrast columns does not match the number of
            estimates, or if ``labels`` length does not match the number of
            contrasts.

        Examples
        --------
        Suppose ``cells`` holds the four adjusted predictions of a 2x2
        ``treat`` x ``post`` DiD, ordered as ``(0,0)``, ``(0,1)``,
        ``(1,0)``, ``(1,1)``. The simple effect of ``treat`` at
        ``post=1``, with delta-method SE::

            cells.contrast([0, -1, 0, 1], labels=["treat | post=1"])

        Stack three related contrasts so that you keep their joint
        covariance (useful for joint Wald tests)::

            joint = cells.contrast(
                [[-1, 0, 1, 0],     # simple effect of treat at post=0
                 [ 0,-1, 0, 1],     # simple effect of treat at post=1
                 [ 1,-1,-1, 1]],    # DiD
                labels=["simple @post=0", "simple @post=1", "DiD"],
            )
            joint.summary()
            joint.vcov              # 3x3 — covariances across the three rows

        A runnable smoke test from synthetic estimates::

            >>> import numpy as np
            >>> from smmargins import MarginsResult
            >>> cells = MarginsResult(
            ...     estimate=np.array([0.10, 0.20, 0.18, 0.40]),
            ...     vcov=0.001 * np.eye(4),
            ...     labels=["t=0,p=0", "t=0,p=1", "t=1,p=0", "t=1,p=1"],
            ... )
            >>> did = cells.contrast([1, -1, -1, 1], labels=["DiD"])
            >>> float(did.estimate[0])
            0.12
            >>> round(float(did.se[0]), 4)
            0.0632

        References
        ----------
        [1] StataCorp. *Stata User's Guide*, Section
           ``[R] margins, contrast``.
        [2] Williams, R. (2012). Using the margins command to estimate
           and interpret adjusted predictions and marginal effects.
           *Stata Journal*, 12(2), 308–331.

        See Also
        --------
        MarginsResult.summary
        Margins.did
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
        r"""Return a summary DataFrame with estimates, SEs, and confidence intervals.

        Returns
        -------
        pandas.DataFrame
            One row per estimate, columns:
            ``stat_name``, ``std.err``, ``z``/``t``, ``P>|z|``/``P>|t|``,
            and the lower and upper confidence bounds.

        Examples
        --------
        >>> import numpy as np
        >>> from smmargins import MarginsResult
        >>> tbl = MarginsResult(
        ...     estimate=np.array([2.0]),
        ...     vcov=np.array([[0.04]]),
        ...     labels=["b"],
        ... ).summary()
        >>> list(tbl.columns)
        ['margin', 'std.err', 'z', 'P>|z|', '[95% CI lo]', '[95% CI hi]']
        >>> float(tbl.loc["b", "std.err"])
        0.2
        >>> float(tbl.loc["b", "z"])
        10.0

        For results that came from :class:`Margins`, ``summary()`` is what
        ``__repr__`` prints — so ``print(M.dydx("x1"))`` already gives you
        the same table.
        """
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
        """Return a string representation of the summary table."""
        return self.summary().to_string(float_format=lambda v: f"{v: .6f}")


# ---------------------------------------------------------------------------
# Evaluation profile
# ---------------------------------------------------------------------------

@dataclass
class _Profile:
    """Evaluation profile: how to turn a data-space frame into a design matrix.

    The two flags express the semantic split that the ``at`` /
    ``factor_stat`` combinations encode, applied at different phases.

    Attributes
    ----------
    frame : pd.DataFrame
        The data-space frame used as the evaluation base.
    collapse_design : bool, default False
        If True, average the design matrix column-wise after building,
        so factor dummies become their observed proportions (Stata
        ``atmeans``).
    collapse_numerics : bool, default False
        If True, replace row-varying numeric columns in the data-space
        frame with their training mean *before* perturbation. Needed
        for count and discrete contrasts under MEM so derived columns
        like ``I(x**2)`` evaluate to ``mean(x)**2``.

    Notes
    -----
    ``collapse_numerics`` (pre-perturbation) is applied by the caller
    (e.g. ``_design_pairs``) *before* perturbing the frame, so that
    the perturbation is not overwritten by the mean-substitution.
    ``collapse_design`` (post-build) is applied by ``materialize``
    after ``_build_exog``.
    """

    frame: pd.DataFrame
    collapse_design: bool = False
    collapse_numerics: bool = False

    def prepare_frame(self, frame: pd.DataFrame, margins: "Margins") -> pd.DataFrame:
        """Apply numerics-collapse to a frame when configured.

        Parameters
        ----------
        frame : pd.DataFrame
            Data-space frame to prepare.
        margins : Margins
            Margins instance providing ``_collapse_numerics_to_mean``.

        Returns
        -------
        pd.DataFrame
            The prepared frame (collapsed or unchanged).
        """
        if self.collapse_numerics:
            return margins._collapse_numerics_to_mean(frame)
        return frame

    def materialize(self, frame: pd.DataFrame, margins: "Margins") -> np.ndarray:
        """Run the full pipeline (numerics collapse \u2192 build \u2192 design collapse).

        Parameters
        ----------
        frame : pd.DataFrame
            Data-space frame to materialize.
        margins : Margins
            Margins instance providing ``_build_exog`` and
            ``_collapse_numerics_to_mean``.

        Returns
        -------
        ndarray
            Final design matrix ready for prediction.

        Notes
        -----
        Use this when there is no perturbation step in between, e.g.
        ``predict`` or the unperturbed reference design in elasticities.
        Paired-contrast callers need to interleave a perturb step between
        the two phases \u2014 they should call ``prepare_frame`` themselves,
        perturb, then build / ``collapse_design`` manually (or use
        ``_design_pairs``).
        """
        frame = self.prepare_frame(frame, margins)
        X = margins._build_exog(frame)
        if self.collapse_design:
            X = X.mean(axis=0, keepdims=True)
        return X


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Margins:
    r"""Compute adjusted predictions and marginal effects for a StatsModels fit.

    Rather than hand-differentiating each model's linear predictor / link
    combination, every statistic is built as a *function of*
    :math:`\beta` using ``model.predict(params, exog)`` (which handles
    the inverse link). The Jacobian of that function is taken by central
    finite differences, and the delta method is applied with
    ``results.cov_params()``.

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
        :math:`\partial g/\partial\beta` whenever the model exposes a link
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

    **Delta method**

    For a statistic :math:`g(\beta)` with Jacobian
    :math:`G = \partial g / \partial \beta|_{\hat\beta}`, the delta
    method approximates

    .. math::

        \widehat{\mathrm{Var}}[g(\hat\beta)] \approx
        G \, \widehat V \, G^\top .

    Examples
    --------
    A logit with a categorical and an interaction, which would be awkward
    to differentiate by hand::

        import statsmodels.formula.api as smf
        from smmargins import Margins

        fit = smf.logit(
            "voted ~ age + income + C(educ) + female + age:female",
            data=df,
        ).fit()
        M = Margins(fit)

    Adjusted predictions::

        M.predict()                                # AAP
        M.predict(at="mean")                       # APM
        M.predict(atexog={"age": [25, 45, 65]})    # APR

    Marginal effects on the response (probability) scale::

        M.dydx("age")                              # AME
        M.dydx("age", at="mean")                   # MEM
        M.dydx("age", atexog={"female": [0, 1]})   # MER, by sex
        M.dydx("educ", reference="college")        # discrete contrasts
        M.dydx("kids", count=True)                 # x -> x+1 for integers
        M.dydx("age", method="eyex")               # full elasticity

    Most calls return a :class:`MarginsResult` whose ``__repr__`` prints
    a tidy table of estimates, SEs, z- (or t-) statistics, p-values, and
    confidence intervals.

    A small runnable smoke test::

        >>> import numpy as np, pandas as pd, statsmodels.formula.api as smf
        >>> from smmargins import Margins
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({
        ...     "x": rng.standard_normal(200),
        ...     "g": rng.choice(["A", "B"], 200),
        ... })
        >>> df["y"] = 1.0 + 2.0 * df["x"] + (df["g"] == "B") + rng.standard_normal(200)
        >>> fit = smf.ols("y ~ x + C(g)", df).fit()
        >>> M = Margins(fit)
        >>> aap = M.predict()
        >>> ame = M.dydx("x")
        >>> aap.estimate.shape, ame.estimate.shape
        ((1,), (1,))
        >>> bool(abs(ame.estimate[0] - 2.0) < 0.2)        # close to truth
        True

    References
    ----------
    [1] StataCorp. FAQ: How are the standard errors computed with
       margins https://www.stata.com/support/faqs/statistics/compute-standard-errors-with-margins/
    [2] Williams, R. (2012). Using the margins command to estimate and
       interpret adjusted predictions and marginal effects.
       Stata Journal, 12(2), 308–331
       https://www3.nd.edu/~rwilliam/stats/Margins01.pdf
    [3] Ai, C., & Norton, E. C. (2003). Interaction terms in logit and
       probit models. Economics Letters, 80(1), 123–129
       https://doi.org/10.1016/S0165-1765(03)00032-6

    See Also
    --------
    MarginsResult
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
            # Multi-outcome models have more parameters than exog columns
            expected_params = n_exog
            if hasattr(self.model, "J"):
                # MNLogit: params shape (p, K-1) -> p*(K-1) flat
                expected_params = n_exog * (self.model.J - 1)
            elif hasattr(self.model, "k_extra"):
                # OrderedModel: p + (K-1) thresholds
                expected_params = n_exog + self.model.k_extra
            if n_params != expected_params:
                raise ValueError(
                    f"Model has {n_exog} exog columns but {n_params} parameters "
                    f"(expected {expected_params}). Cannot use raw mode."
                )
            # exog_names may include threshold parameters for OrderedModel,
            # so only validate the first n_exog names match the data columns.
            exog_names = list(self.model.exog_names)
            if len(exog_names) < n_exog:
                raise ValueError(
                    f"Model has {len(exog_names)} exog_names but "
                    f"{n_exog} exog columns. Cannot use raw mode."
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

        # Detect number of outcome classes for multi-outcome models
        self._n_outcomes = self._detect_n_outcomes()
        self._outcome_labels = self._detect_outcome_labels()

    def _detect_n_outcomes(self) -> int:
        """Detect the number of outcome classes (K) for the fitted model.

        Returns
        -------
        int
            Number of outcome classes. 1 for single-outcome models.
        """
        # MNLogit
        if hasattr(self.model, "J"):
            return int(self.model.J)
        # OrderedModel
        if hasattr(self.model, "k_extra"):
            return int(self.model.k_extra) + 1
        # Last resort: probe with a 1-row prediction
        try:
            exog = getattr(self.model, "exog", None)
            if exog is not None and exog.shape[0] > 0:
                probe = exog[:1]
                pred = self.model.predict(self.results.params, probe)
                pred_arr = np.asarray(pred)
                if pred_arr.ndim == 2:
                    return pred_arr.shape[1]
        except Exception:
            pass
        return 1

    def _detect_outcome_labels(self) -> Optional[List[str]]:
        """Detect outcome class labels for multi-outcome models.

        Returns
        -------
        list of str or None
            Labels for each outcome class, or None for single-outcome models.
        """
        if self._n_outcomes == 1:
            return None
        # MNLogit
        ynames = getattr(self.model, "_ynames_map", None)
        if ynames is not None:
            # ynames_map maps class index -> label
            labels = [str(ynames.get(i, i)) for i in range(self._n_outcomes)]
            return labels
        # Try to infer from endog
        endog = getattr(self.model, "endog", None)
        if endog is not None:
            try:
                uniq = np.unique(endog)
                if len(uniq) == self._n_outcomes:
                    return [str(u) for u in uniq]
            except Exception:
                pass
        return [str(i) for i in range(self._n_outcomes)]

    @property
    def n_outcomes(self) -> int:
        """Number of outcome classes (K) for the fitted model."""
        return self._n_outcomes

    @property
    def outcome_labels(self) -> Optional[List[str]]:
        """Outcome class labels for multi-outcome models, or None."""
        return self._outcome_labels

    def _compute_mean_log_offset(self) -> Optional[float]:
        """Mean of (offset + log(exposure)) across the training sample.

        Returns
        -------
        float or None
            The mean log-offset when offset/exposure is present and
            non-zero; ``None`` otherwise.

        Notes
        -----
        This value is used for offset/exposure broadcasting when the
        design matrix has fewer rows than the training sample (e.g.
        APM/MEM single-row evaluations). Without it, statsmodels
        cannot align a length-N stored offset against a 1-row exog
        and silently drops the offset term.
        """
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
        """Attempt to retrieve the original fitting data frame from the model.

        Returns
        -------
        pd.DataFrame or None
            The original data frame when available; ``None`` otherwise.
        """
        try:
            return self.model.data.frame
        except AttributeError:
            pass
        # Some models (e.g. MNLogit, OrderedModel) store the data in orig_exog
        orig_exog = getattr(self.model.data, "orig_exog", None)
        if orig_exog is not None and hasattr(orig_exog, "columns"):
            return pd.DataFrame(orig_exog)
        return None

    def _try_get_design_info(self):
        """Retrieve the patsy DesignInfo from the model, if available.

        Returns
        -------
        DesignInfo or None
            The patsy ``DesignInfo`` object when the model was fit with a
            formula; ``None`` in raw-exog mode.

        Notes
        -----
        In formula mode, the DesignInfo encodes how the data frame maps
        to the design matrix (interactions, transformations, categorical
        encodings). In raw mode, ``Margins`` falls back to column-name
        matching against ``model.exog_names``.
        """
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

        Parameters
        ----------
        frame : pd.DataFrame
            Data-space frame containing the required columns.

        Returns
        -------
        ndarray
            Numeric design matrix suitable for ``model.predict``.

        Notes
        -----
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

        Parameters
        ----------
        params : ndarray of shape (p,)
            Parameter vector override.
        exog : ndarray of shape (n, p)
            Design matrix.

        Returns
        -------
        ndarray of shape (n,)
            Predicted values on the response scale.

        Notes
        -----
        Many StatsModels results accept a ``params`` override to
        ``model.predict``; for GLMs this applies the inverse link. When
        the model carries offset/exposure but the design has a different
        row count than training (e.g. APM/MEM single rows), we broadcast
        the *mean* training offset/exposure to the new design \u2014 otherwise
        statsmodels cannot align the stored length-N offset against the
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
        r"""Return the link derivative :math:`f'(\eta)`, or ``None``.

        Returns a callable ``fprime(eta)`` when an analytic derivative
        of the inverse link is available; otherwise ``None``, signalling
        that the caller should fall back to finite differences.

        Returns
        -------
        callable or None
            ``fprime(eta)`` returning :math:`f'(\eta)` element-wise, or
            ``None`` if the model does not expose an analytic derivative.

        Notes
        -----
        Eligible when the model exposes ``family.link.inverse_deriv`` (every
        stock GLM family) or is a linear-regression model (identity link).
        Bails out returning ``None`` whenever an offset or exposure is
        present, since then :math:`\eta \neq X\beta` and our chain rule
        would need an offset-aware path we do not currently provide.
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
        r"""Gradient of the mean prediction with respect to parameters.

        Computes
        :math:`\partial/\partial\beta` of
        :math:`(1/n)\sum_i f(x_i^\top\beta)`, returning shape ``(p,)``.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Design matrix.
        beta : ndarray of shape (p,)
            Parameter vector.
        fprime : callable
            Function returning :math:`f'(\eta)` for each linear
            predictor value.

        Returns
        -------
        ndarray of shape (p,)
            Gradient vector.

        Notes
        -----
        Building block for every analytic Jacobian in this module: every
        statistic we compute is a linear combination of mean-predictions
        on different design matrices, so each row of the analytic ``J``
        is a linear combination of these gradients.
        """
        X = np.asarray(X, dtype=float)
        eta = X @ beta
        fp = np.asarray(fprime(eta), dtype=float).ravel()
        return (fp[:, None] * X).sum(axis=0) / X.shape[0]

    # ---- shared paired-contrast machinery ------------------------------------
    #
    # Every dydx variant — continuous (level), count, discrete — is a list of
    # contrasts ``E[f(X_a)] - E[f(X_b)]`` over pairs of design matrices, often
    # multiplied by a constant. ``_design_pairs`` builds those pairs from a
    # callable that returns the ``(a, b)`` data-space frames for one input
    # frame; ``_paired_contrast`` turns the list of pairs into a
    # ``(statistic, jac)`` tuple suitable for ``_delta``.

    def _design_pairs(
        self,
        profile: _Profile,
        frames: List[pd.DataFrame],
        perturb: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """Build paired design matrices for a perturbation-based contrast.

        For each input frame, applies ``profile.prepare_frame``, then
        the perturbation callable, then ``_build_exog``, and optionally
        design-matrix collapse.

        Parameters
        ----------
        profile : _Profile
            Evaluation profile controlling numerics/design collapse.
        frames : list of pd.DataFrame
            Base frames (already expanded via ``_expand_at``).
        perturb : callable
            Function ``(pd.DataFrame) -> (fa, fb)`` producing the two
            perturbed data-space frames for the contrast.

        Returns
        -------
        list of tuple
            List of ``(Xa, Xb)`` design-matrix pairs.

        Notes
        -----
        Numerics-collapse runs *before* perturb, so the perturbation
        (e.g. ``variable + 1``) isn't overwritten by the mean-substitution.
        """
        # Numerics-collapse runs *before* perturb, so the perturbation
        # (e.g. ``variable + 1``) isn't overwritten by the mean-substitution.
        out: List[tuple[np.ndarray, np.ndarray]] = []
        for fr in frames:
            prepared = profile.prepare_frame(fr, self)
            fa, fb = perturb(prepared)
            Xa = self._build_exog(fa)
            Xb = self._build_exog(fb)
            if profile.collapse_design:
                Xa = Xa.mean(axis=0, keepdims=True)
                Xb = Xb.mean(axis=0, keepdims=True)
            out.append((Xa, Xb))
        return out

    def _paired_contrast(
        self,
        pairs: Sequence[tuple[np.ndarray, np.ndarray]],
        scale: float = 1.0,
    ) -> tuple[Callable[[np.ndarray], np.ndarray],
               Optional[Callable[[np.ndarray], np.ndarray]]]:
        """Compute ``E[f(X_a)] - E[f(X_b)]`` per pair, scaled by ``scale``.

        Parameters
        ----------
        pairs : sequence of tuple
            Each element is ``(Xa, Xb)``, a pair of design matrices.
        scale : float, default 1.0
            Multiplicative constant applied to each contrast.

        Returns
        -------
        tuple
            ``(statistic, jac)``. ``jac`` is None when no analytic
            link derivative is available — the caller falls back to FD via
            ``_central_jacobian``.
        """
        n = len(pairs)

        def statistic(beta: np.ndarray) -> np.ndarray:
            out = np.empty(n)
            for i, (Xa, Xb) in enumerate(pairs):
                ya = float(np.mean(self._predict(beta, Xa)))
                yb = float(np.mean(self._predict(beta, Xb)))
                out[i] = (ya - yb) * scale
            return out

        fprime = self._link_deriv()
        if fprime is None:
            return statistic, None

        def jac(beta: np.ndarray) -> np.ndarray:
            J = np.empty((n, beta.size))
            for i, (Xa, Xb) in enumerate(pairs):
                ga = self._grad_mean_predict(Xa, beta, fprime)
                gb = self._grad_mean_predict(Xb, beta, fprime)
                J[i, :] = (ga - gb) * scale
            return J

        return statistic, jac

    # ---- utilities for building "at" frames ----

    @staticmethod
    def _expand_at(
        base: pd.DataFrame, at: Optional[Mapping[str, object]]
    ) -> tuple[List[pd.DataFrame], List[str]]:
        """Cartesian-product expansion of an ``at`` specification.

        Parameters
        ----------
        base : pd.DataFrame
            Base frame (e.g. from ``_single_row`` or the full data).
        at : mapping, optional
            Variable name -> scalar or list of scalars.

        Returns
        -------
        tuple
            ``(frames, labels)`` where each frame is ``base`` with the
            chosen values broadcast, and ``labels`` describe the
            ``atexog`` combination.

        Notes
        -----
        ``at`` maps variable names to either scalars or lists of scalars.
        Lists are expanded as a cartesian product over all variables.
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

        Parameters
        ----------
        at : {"mean", "median", "zero"}
            Evaluation point for numeric covariates.
        factor_stat : {"mode", "zero"}
            How to set factor / categorical columns.

        Returns
        -------
        pd.DataFrame
            One-row data frame with all columns set to their
            representative values.

        Notes
        -----
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

        Parameters
        ----------
        col : str
            Column name of a categorical variable in the data frame.

        Returns
        -------
        object
            The reference level \u2014 the one that contributes nothing to the
            linear predictor under default treatment contrasts.

        Notes
        -----
        Probes one-row designs with ``col`` set to each observed level
        (other variables held at simple defaults) and picks the level
        whose 1-row design has the minimum L1 norm \u2014 i.e. the one that
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
        """Return a default probe value for a column.

        Parameters
        ----------
        col : str
            Column name in ``self.data``.

        Returns
        -------
        object
            ``0.0`` for numeric columns; the alphabetically first observed
            level otherwise.
        """
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

    def _resolve_base(self, at: str, factor_stat: str) -> _Profile:
        """Return the evaluation profile for ``(at, factor_stat)``.

        Parameters
        ----------
        at : {"overall", "mean", "median", "zero"}
            Evaluation point for numeric covariates.
        factor_stat : {"mean", "mode", "zero"}
            How to handle factor / categorical variables.

        Returns
        -------
        _Profile
            Evaluation profile with ``collapse_design`` and
            ``collapse_numerics`` flags set appropriately.

        Notes
        -----
        ``profile.collapse_design`` is True iff factors should end up at
        their observed proportions (Stata ``atmeans`` semantics).
        ``profile.collapse_numerics`` is always False here \u2014 the count
        and discrete contrast paths flip it on themselves before
        materializing, since they need a single design row in data space
        rather than the design-column averages that the continuous FD
        path relies on.
        """
        if at == "overall":
            return _Profile(frame=self.data, collapse_design=False)

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
            return _Profile(frame=f, collapse_design=True)

        return _Profile(frame=self._single_row(at, factor_stat),
                        collapse_design=False)

    def _collapse_numerics_to_mean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace observed (row-varying) numeric columns with their training mean.

        Parameters
        ----------
        frame : pd.DataFrame
            Data-space frame, possibly containing ``atexog`` overrides.

        Returns
        -------
        pd.DataFrame
            Copy with row-varying numeric columns replaced by their
            training-sample mean.

        Notes
        -----
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
        """Validate ``at`` argument, raising if unknown."""
        if at not in ("overall", "mean", "median", "zero"):
            raise ValueError(
                f"at must be 'overall', 'mean', 'median', or 'zero', "
                f"got {at!r}"
            )

    @staticmethod
    def _check_factor_stat(factor_stat: str) -> None:
        """Validate ``factor_stat`` argument, raising if unknown."""
        if factor_stat not in ("mean", "mode", "zero"):
            raise ValueError(
                f"factor_stat must be 'mean', 'mode', or 'zero', "
                f"got {factor_stat!r}"
            )

    @staticmethod
    def _default_factor_stat(at: str) -> str:
        """Pick a sensible ``factor_stat`` when the user did not supply one.

        Maps ``at`` values to their natural factor handling:
        ``overall`` -> ``mean``, ``mean`` -> ``mean``, ``median`` -> ``mode``,
        ``zero`` -> ``zero``.
        """
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
        r"""Apply the delta method to a vector-valued statistic.

        Computes :math:`\hat\theta = g(\hat\beta)` and its delta-method
        covariance :math:`\widehat{\operatorname{Var}}(\hat\theta) = J V J^\top`,
        where :math:`J = \partial g / \partial \beta` is either supplied
        analytically or obtained by central finite differences.

        Parameters
        ----------
        statistic : callable
            Function mapping a parameter vector :math:`\beta` to a vector
            of statistic values.
        labels : sequence of str
            Row labels for each component of the statistic.
        stat_name : str
            Column name to use in the summary table.
        jac : callable, optional
            Function returning the analytic Jacobian matrix
            :math:`\partial g / \partial \beta`. If ``None``, the Jacobian
            is computed numerically via ``_central_jacobian``.

        Returns
        -------
        MarginsResult
            Estimates with delta-method standard errors and confidence
            intervals.

        Notes
        -----
        This is the core inference primitive shared by every public API
        method (``predict``, ``dydx``, ``did``).
        """
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

    # ======================================================================}
    # Public API: adjusted predictions
    # ======================================================================}

    def predict(
            self,
            at: str = "overall",
            atexog: Optional[Mapping[str, object]] = None,
            factor_stat: Optional[str] = None,
        ) -> MarginsResult:
        r"""Compute adjusted predictions (expected outcome on the response scale).

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

        Examples
        --------
        Adjusted-prediction profile across ``age`` for each sex::

            M.predict(atexog={"age": list(range(20, 91, 10)),
                              "female": [0, 1]})

        APM (Stata's ``margins, atmeans``) versus AAP::

            M.predict(at="mean")    # factor dummies as observed proportions
            M.predict()             # AAP — usually the more defensible default

        Hold ``age`` at three policy-relevant values, average everything else
        over the sample (Stata's ``margins, at(age=(25 45 65))``)::

            M.predict(atexog={"age": [25, 45, 65]})

        A runnable smoke test on a small linear model::

            >>> import numpy as np, pandas as pd, statsmodels.formula.api as smf
            >>> from smmargins import Margins
            >>> rng = np.random.default_rng(0)
            >>> df = pd.DataFrame({
            ...     "x1": rng.standard_normal(50),
            ...     "x2": rng.standard_normal(50),
            ... })
            >>> df["y"] = 1.0 + 2.0 * df["x1"] - df["x2"] + 0.1 * rng.standard_normal(50)
            >>> fit = smf.ols("y ~ x1 + x2", df).fit()
            >>> M = Margins(fit)
            >>> M.predict(atexog={"x1": [0, 1]}).estimate.shape
            (2,)
            >>> bool(M.predict(atexog={"x1": [0, 1]}).estimate[1]
            ...      > M.predict(atexog={"x1": [0]}).estimate[0])
            True

        See Also
        --------
        Margins.dydx
        Margins.did
        """
        self._check_at(at)
        if factor_stat is None:
            factor_stat = self._default_factor_stat(at)
        self._check_factor_stat(factor_stat)

        profile = self._resolve_base(at, factor_stat)
        frames, at_labels = self._expand_at(profile.frame, atexog)

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

        Xs = [profile.materialize(f, self) for f in frames]

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

    # ======================================================================}
    # Public API: marginal effects
    # ======================================================================}

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
        r"""Marginal effect of ``variable`` on the response.

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

            - ``"dydx"`` : :math:`\partial y_i / \partial x_{ij}`
              (level change). The default; only path with an analytic
              outer Jacobian.
            - ``"dyex"`` : :math:`(\partial y_i / \partial x_{ij})\,x_i`
              (semi-elasticity, :math:`dy/d(\ln x)`).
            - ``"eyex"`` : :math:`(\partial y_i / \partial x_{ij})\,x_i / y_i`
              (full elasticity).
            - ``"eydx"`` : :math:`(\partial y_i / \partial x_{ij}) / y_i`
              (semi-elasticity, :math:`d(\ln y)/dx`).

            Only valid for continuous variables; raises if the variable
            is (auto- or explicitly) discrete and ``method != "dydx"``.
            Elasticity methods always go through finite differences;
            ``analytic=True`` on the constructor only affects ``"dydx"``.

        Returns
        -------
        MarginsResult

        Raises
        ------
        ValueError
            If ``count=True`` and ``discrete=True`` are both passed, if
            ``count=True`` with ``method != "dydx"``, or if ``method``
            is not one of the four allowed strings.

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

        The Average Marginal Effect (AME) is

        .. math::

            \frac{1}{n}\sum_i \frac{\partial f(x_i^\top\beta)}{\partial x_{ij}},

        where :math:`f` is the inverse link function. The Marginal Effect
        at Means (MEM) replaces each :math:`x_i` by its sample mean (or
        the chosen ``at`` profile) before taking the derivative.

        Examples
        --------
        AME, MEM, and MER for a continuous covariate::

            M.dydx("age")                              # AME
            M.dydx("age", at="mean")                   # MEM
            M.dydx("age", atexog={"female": [0, 1]})   # MER — AME by sex

        Multiple variables at once (one row per variable, joint covariance
        on the result so ``contrast()`` can mix them safely)::

            M.dydx(["age", "income"])
            M.dydx("*")                                # everything

        Discrete cases. Booleans, strings, and numerics with <=2 unique
        values are auto-detected; you can override with ``discrete=`` or
        ``count=``::

            M.dydx("female")                           # 0/1, auto-discrete
            M.dydx("educ", reference="college")        # contrasts vs reference
            M.dydx("kids", count=True)                 # unit increment x -> x+1

        Elasticities (continuous variables only)::

            M.dydx("income", method="eyex")            # ey/ex (full elasticity)
            M.dydx("income", method="dyex")            # dy/ex
            M.dydx("income", method="eydx")            # ey/dx

        A runnable smoke test on a small linear model where the AME of
        ``x1`` is exactly :math:`\beta_1`::

            >>> import numpy as np, pandas as pd, statsmodels.formula.api as smf
            >>> from smmargins import Margins
            >>> rng = np.random.default_rng(0)
            >>> df = pd.DataFrame({
            ...     "x1": rng.standard_normal(200),
            ...     "x2": rng.standard_normal(200),
            ... })
            >>> df["y"] = 1.0 + 2.0 * df["x1"] - df["x2"] + 0.1 * rng.standard_normal(200)
            >>> fit = smf.ols("y ~ x1 + x2", df).fit()
            >>> M = Margins(fit)
            >>> ame = M.dydx("x1")
            >>> ame.estimate.shape
            (1,)
            >>> bool(abs(ame.estimate[0] - 2.0) < 0.05)
            True
            >>> M.dydx(["x1", "x2"]).estimate.shape
            (2,)

        See Also
        --------
        Margins.predict
        Margins.did
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

        profile = self._resolve_base(at, factor_stat)
        frames, at_labels = self._expand_at(profile.frame, atexog)

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
                    v, profile, frames, at_labels,
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
                    v, profile, frames, at_labels, reference=reference,
                )
                parts.append(res_parts)
            else:
                res_parts = self._dydx_continuous_components(
                    v, profile, frames, at_labels, step=step, at=at,
                    method=method,
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

    @staticmethod
    def _continuous_label_suffix(at: str) -> str:
        """Human-readable suffix for an evaluation profile."""
        return {"overall": "", "mean": " (at means)",
                "median": " (at medians)", "zero": " (at zero)"}[at]

    def _continuous_labels(
        self, variable: str, method: str, at: str, at_labels: List[str],
        n_frames: int,
    ) -> List[str]:
        """Build column labels for a continuous marginal effect.

        Labels combine the method prefix (``d`` for ``dydx``/``dyex``,
        ``e`` for ``eyex``/``eydx``), the variable name, the evaluation
        profile suffix, and any ``atexog`` label.
        """
        prefix = _METHOD_META[method]["prefix"]
        if n_frames == 1 and at_labels[0] == "":
            return [f"{prefix}{variable}{self._continuous_label_suffix(at)}"]
        return [f"{prefix}{variable} | {lab}" if lab else f"{prefix}{variable}"
                for lab in at_labels]

    def _dydx_continuous_components(
        self,
        variable: str,
        profile: _Profile,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        step: Optional[float],
        at: str,
        method: str = "dydx",
    ):
        r"""Build statistic and Jacobian for a continuous derivative marginal effect.

        Uses a central finite-difference step to approximate
        :math:`\partial y / \partial x`.  For ``method="dydx"`` the paired
        contrast machinery can be used with an analytic Jacobian; elasticity
        branches (``dyex``, ``eyex``, ``eydx``) require per-row scaling and
        fall back to finite differences for the outer Jacobian.

        Parameters
        ----------
        variable : str
            Column name of the continuous covariate.
        profile : _Profile
            Evaluation profile controlling numerics/design collapse.
        frames : list of pd.DataFrame
            Base frames (already expanded via ``_expand_at``).
        at_labels : list of str
            Labels for each ``atexog`` combination.
        step : float, optional
            Finite-difference step size. Defaults to
            :math:`\epsilon^{1/3} \cdot \max(\text{sd}(x), |\bar x|, 1)`.
        at : str
            Profile name (``overall``, ``mean``, ``median``, ``zero``).
        method : {"dydx", "dyex", "eyex", "eydx"}, default "dydx"
            Transform applied to the raw derivative before averaging.

        Returns
        -------
        tuple
            ``(statistic, jac, labels, stat_name)`` suitable for ``_delta``.

        Notes
        -----
        ``method="dydx"`` is the only path that admits an analytic outer
        Jacobian (via ``_paired_contrast``).  Elasticity methods need
        per-row :math:`x_i` and :math:`y_i` values, so the Jacobian is
        always obtained by finite differences.
        """
        if step is None:
            sd = float(self.data[variable].std(ddof=0))
            scale = max(sd, abs(float(self.data[variable].mean())), 1.0)
            step = (np.finfo(float).eps ** (1.0 / 3.0)) * scale
        h = float(step)

        labels = self._continuous_labels(variable, method, at, at_labels, len(frames))
        stat_name = _METHOD_META[method]["stat_name"]

        def perturb(fr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            x_col = np.asarray(fr[variable], dtype=float)
            fp = fr.copy(); fp[variable] = x_col + h
            fm = fr.copy(); fm[variable] = x_col - h
            return fp, fm

        if method == "dydx":
            # Pure paired contrast scaled by 1/(2h) — analytic jac available.
            pairs = self._design_pairs(profile, frames, perturb)
            statistic, jac = self._paired_contrast(pairs, scale=1.0 / (2.0 * h))
            return (statistic, jac, labels, stat_name)

        # Elasticity branches need per-row x_i (and per-row y_i for eyex/eydx),
        # so they don't fit the simple paired-mean schema. Build the same
        # design pairs, but keep the original (unperturbed) design and the
        # original x column around for the per-row scaling.
        need_y = method in ("eyex", "eydx")
        Xs_plus: List[np.ndarray] = []
        Xs_minus: List[np.ndarray] = []
        Xs_orig: List[np.ndarray] = []
        x_vals: List[np.ndarray] = []
        for fr in frames:
            x_col = np.asarray(fr[variable], dtype=float)
            fp, fm = perturb(fr)
            Xs_plus.append(profile.materialize(fp, self))
            Xs_minus.append(profile.materialize(fm, self))
            if need_y:
                Xs_orig.append(profile.materialize(fr, self))
            if profile.collapse_design:
                x_col = np.array([x_col.mean()])
            x_vals.append(x_col)

        def statistic(beta: np.ndarray) -> np.ndarray:
            out = np.empty(len(Xs_plus))
            for i in range(len(Xs_plus)):
                dydx_i = (self._predict(beta, Xs_plus[i])
                          - self._predict(beta, Xs_minus[i])) / (2.0 * h)
                if method == "dyex":
                    contrib = dydx_i * x_vals[i]
                else:  # eyex / eydx
                    y_i = self._predict(beta, Xs_orig[i])
                    if method == "eyex":
                        contrib = dydx_i * x_vals[i] / y_i
                    else:  # eydx
                        contrib = dydx_i / y_i
                out[i] = np.mean(contrib)
            return out

        # Analytic outer jac would require quotient-rule plumbing for the
        # per-row 1/y_i; FD is fine.
        return (statistic, None, labels, stat_name)

    # ----- count case (unit increment x -> x+1) ----------------------------

    def _dydx_count_components(
        self,
        variable: str,
        profile: _Profile,
        frames: List[pd.DataFrame],
        at_labels: List[str],
    ):
        r"""Build statistic and Jacobian for a count (unit increment) marginal effect.

        Computes :math:`E[f(X \mid x+1)] - E[f(X \mid x)]` via paired
        contrasts, treating ``variable`` as an integer-valued covariate.

        Parameters
        ----------
        variable : str
            Column name of the integer-valued covariate.
        profile : _Profile
            Evaluation profile controlling design collapse.
        frames : list of pd.DataFrame
            Base frames (already expanded via ``_expand_at``).
        at_labels : list of str
            Labels for each ``atexog`` combination.

        Returns
        -------
        tuple
            ``(statistic, jac, labels, stat_name)`` suitable for ``_delta``.

        Notes
        -----
        When ``collapse_design`` is active, we also ``collapse_numerics`` so
        derived columns like ``I(x**2)`` evaluate to ``mean(x)**2`` rather
        than ``mean(x**2)``. The continuous-FD path does not need this
        because its :math:`O(h^2)` truncation error vanishes; the unit
        increment does not.
        """
        # Under ``collapse_design`` we also collapse_numerics, so that
        # I(x**2)-style derived columns evaluate to mean(x)**2 rather than
        # mean(x**2) — i.e. the contrast becomes f(η at mean+1) − f(η at
        # mean), Stata-style. The continuous-FD path doesn't need this
        # because its O(h²) residual vanishes; the unit increment doesn't.
        profile = _Profile(
            frame=profile.frame,
            collapse_design=profile.collapse_design,
            collapse_numerics=profile.collapse_design,
        )

        def perturb(fr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            fp = fr.copy(); fp[variable] = fp[variable] + 1
            return fp, fr

        pairs = self._design_pairs(profile, frames, perturb)
        statistic, jac = self._paired_contrast(pairs)
        base = f"{variable} (count)"
        labels = [f"{base} | {lab}" if lab else base for lab in at_labels]
        return (statistic, jac, labels, "dy/dx")

    # ----- discrete case (contrast vs reference level) ---------------------

    def _dydx_discrete_components(
        self,
        variable: str,
        profile: _Profile,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        reference: Optional[object],
    ):
        r"""Build statistic and Jacobian for discrete contrasts of a factor variable.

        For each non-reference level, compute
        :math:`E[f(X \mid \text{level})] - E[f(X \mid \text{reference})]`
        via paired contrasts.

        Parameters
        ----------
        variable : str
            Factor column name in the data frame.
        profile : _Profile
            Evaluation profile controlling design collapse.
        frames : list of pd.DataFrame
            Base frames (already expanded via ``_expand_at``).
        at_labels : list of str
            Labels for each ``atexog`` combination.
        reference : object, optional
            Reference level to contrast against. Defaults to the smallest
            observed level.

        Returns
        -------
        tuple
            ``(statistic, jac, labels, stat_name)`` suitable for ``_delta``.

        Notes
        -----
        Enables ``collapse_numerics`` when ``collapse_design`` is active so
        that derived columns (e.g. ``I(x**2)``) evaluate at the mean before
        the level substitution, matching Stata semantics.
        """
        # See ``_dydx_count_components`` for why we collapse numerics here.
        profile = _Profile(
            frame=profile.frame,
            collapse_design=profile.collapse_design,
            collapse_numerics=profile.collapse_design,
        )

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

        # One pair per (frame × non-reference level). Build manually rather
        # than via ``_design_pairs``'s single-perturb callable. ``materialize``
        # is safe here because ``variable`` is the discrete level being set;
        # numerics-collapse runs first and doesn't touch it.
        pairs: List[tuple[np.ndarray, np.ndarray]] = []
        labels: List[str] = []
        for frame, at_lab in zip(frames, at_labels):
            f_ref = frame.copy(); f_ref[variable] = reference
            Xref = profile.materialize(f_ref, self)
            for lvl in others:
                f_lvl = frame.copy(); f_lvl[variable] = lvl
                pairs.append((profile.materialize(f_lvl, self), Xref))
                base = f"{variable}: {lvl} vs {reference}"
                labels.append(f"{base} | {at_lab}" if at_lab else base)

        statistic, jac = self._paired_contrast(pairs)
        return (statistic, jac, labels, "contrast")

    # ======================================================================}
    # Public API: difference-in-differences
    # ======================================================================}

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
        r"""Difference-in-differences on the response scale.

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

        Examples
        --------
        Healthcare-style 2x2: does the rate of condition X differ between
        groups A/B, and does that gap depend on a preexisting condition
        Y? Average over the rest of the covariate distribution::

            M = Margins(fit)
            res = M.did("group", "preexist_Y",
                        group_levels=["A", "B"], condition_levels=[0, 1])
            print(res)                # 4 cells, 2 simple effects, 1 DiD
            res.did.estimate          # the DiD point estimate
            res.did.ci_lower          # 95% lower CI
            res.cells.vcov            # 4x4 joint covariance of the cells

        Same DiD but at one specific patient profile::

            M.did("group", "preexist_Y",
                  group_levels=["A", "B"], condition_levels=[0, 1],
                  atexog={"age": 60, "female": 0})

        Sanity check on a linear model — the DiD on the response scale
        should equal the coefficient on the ``treat:post`` interaction
        (it does **not** for nonlinear links; see the Notes)::

            >>> import numpy as np, pandas as pd, statsmodels.formula.api as smf
            >>> from smmargins import Margins
            >>> rng = np.random.default_rng(0)
            >>> n = 400
            >>> df = pd.DataFrame({
            ...     "treat": rng.integers(0, 2, n),
            ...     "post":  rng.integers(0, 2, n),
            ... })
            >>> df["y"] = (1.0 + 0.5 * df["treat"] + 0.3 * df["post"]
            ...            + 0.7 * df["treat"] * df["post"]
            ...            + 0.1 * rng.standard_normal(n))
            >>> fit = smf.ols("y ~ treat * post", df).fit()
            >>> res = Margins(fit).did("treat", "post")
            >>> res.cells.estimate.shape
            (4,)
            >>> bool(abs(res.did.estimate[0]
            ...          - fit.params["treat:post"]) < 1e-10)
            True

        References
        ----------
        [1] Ai, C., & Norton, E. C. (2003). Interaction terms in logit
           and probit models. *Economics Letters*, 80(1), 123–129.
           https://doi.org/10.1016/S0165-1765(03)00032-6
        [2] Williams, R. (2012). Using the margins command to estimate
           and interpret adjusted predictions and marginal effects.
           *Stata Journal*, 12(2), 308–331.

        See Also
        --------
        MarginsResult.contrast
        DiDResult
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
        """Return the two most extreme unique levels of a column.

        Parameters
        ----------
        col : str
            Column name in ``self.data``.

        Returns
        -------
        list
            A two-element list ``[smallest, largest]`` of observed levels.

        Raises
        ------
        ValueError
            If the column has fewer than 2 unique values.
        """
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
    r"""Bundle of results from :meth:`Margins.did`.

    Holds the four cell predictions, the two simple effects, the
    difference-in-differences estimate, and a joint result that contains
    all three contrasts with their shared covariance.

    Parameters
    ----------
    cells : MarginsResult
        The four adjusted predictions.
    simple_effects : MarginsResult
        Group effect evaluated at each level of the condition.
    did : MarginsResult
        The single difference-in-differences estimate.
    joint : MarginsResult
        All three contrasts (2 simple effects + DiD) sharing joint vcov,
        useful if you want to jointly test e.g.
        ``simple_effects = did = 0``.

    Attributes
    ----------
    cells : MarginsResult
    simple_effects : MarginsResult
    did : MarginsResult
    joint : MarginsResult

    Examples
    --------
    Use the bundle to grab whichever piece you want::

        res = M.did("group", "preexist_Y",
                    group_levels=["A", "B"], condition_levels=[0, 1])
        print(res)                          # full report (cells + effects + DiD)
        res.cells.summary()                 # 4-row table for plotting
        res.simple_effects.estimate         # 2 group-effects, one per Y level
        res.did.estimate                    # the single DiD on the response scale
        res.joint.vcov                      # 3x3 covariance for joint Wald tests

    Runnable smoke test::

        >>> import numpy as np, pandas as pd, statsmodels.formula.api as smf
        >>> from smmargins import Margins
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({
        ...     "treat": rng.integers(0, 2, 200),
        ...     "post":  rng.integers(0, 2, 200),
        ... })
        >>> df["y"] = (df["treat"] + df["post"] + 0.5 * df["treat"] * df["post"]
        ...            + rng.standard_normal(200))
        >>> res = Margins(smf.ols("y ~ treat * post", df).fit()).did("treat", "post")
        >>> res.cells.estimate.shape
        (4,)
        >>> res.simple_effects.estimate.shape
        (2,)
        >>> res.did.estimate.shape
        (1,)
        >>> res.joint.vcov.shape
        (3, 3)
    """

    def __init__(self, cells: MarginsResult, simple_effects: MarginsResult,
                 did: MarginsResult, joint: MarginsResult):
        self.cells = cells
        self.simple_effects = simple_effects
        self.did = did
        self.joint = joint

    def summary(self) -> pd.DataFrame:
        r"""Return the full DiD summary as one concatenated DataFrame.

        Returns
        -------
        pandas.DataFrame
            Vertically concatenated summaries of ``cells``, ``simple_effects``,
            and ``did``, with an extra column indicating the block
            (``"cell"``, ``"simple"``, ``"DiD"``).

        Examples
        --------
        >>> import numpy as np, pandas as pd, statsmodels.formula.api as smf
        >>> from smmargins import Margins
        >>> np.random.seed(0)
        >>> df = pd.DataFrame({
        ...     "y": np.random.randn(10),
        ...     "treat": np.repeat([0, 1], 5),
        ...     "post": np.tile([0, 1], 5),
        ... })
        >>> fit = smf.ols("y ~ treat * post", df).fit()
        >>> did = Margins(fit).did("treat", "post")
        >>> isinstance(did.summary(), pd.DataFrame)
        True
        >>> len(did.summary())
        7
        """
        parts = [
            self.cells.summary().assign(**{"": "cell"}),
            self.simple_effects.summary().assign(**{"": "simple"}),
            self.did.summary().assign(**{"": "DiD"}),
        ]
        return pd.concat(parts)

    def __repr__(self) -> str:
        """Return a formatted string with cell predictions, simple effects, and DiD."""
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
