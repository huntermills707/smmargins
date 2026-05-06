from __future__ import annotations

import itertools
import warnings
from typing import Callable, Optional, Union, Sequence, Mapping, List

import numpy as np
import pandas as pd
import patsy

from .results import MarginsResult, DiDResult
from .data import _Profile
from .utils import _central_jacobian, _METHOD_META, _get_param_cov, check_at, check_factor_stat, default_factor_stat
from .inference import _simulate_vce, _bootstrap_vce, _summarize_draws
from .transforms import Transform

from ._design import DesignResolver
from ._engine import PredictionEngine
from ._derivs import DerivativeEngine


class Margins:
    r"""Compute adjusted predictions and marginal effects for a StatsModels fit.

    Every statistic is built as a *function of* :math:`\beta` using
    ``model.predict(params, exog)`` (which handles the inverse link).
    The default delta-method VCE applies ``J Var(\betâ) J^\top`` where
    ``J`` is taken analytically when the link supports it (``family.link
    .inverse_deriv`` or the identity link of OLS/WLS/GLS) and by central
    finite differences otherwise. Krinsky–Robb and bootstrap VCEs are
    available via ``vce=`` on :meth:`predict`, :meth:`dydx`, and
    :meth:`did`.

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
    cov_type : str, optional
        Recompute the parameter covariance under this scheme (e.g.
        ``"HC0"``..``"HC3"``, ``"cluster"``, ``"HAC"``). Passed through
        to ``results.get_robustcov_results`` (or equivalent). If neither
        ``cov_type`` nor ``vcov`` is set, ``results.cov_params()`` is used
        as-is — so a ``results`` object that was fit with a robust
        ``cov_type`` keeps its sandwich. Setting ``cov_type`` here
        overrides whatever is on ``results``. The same kwargs are also
        accepted on :meth:`predict`, :meth:`dydx`, and :meth:`did` for
        per-call overrides.
    vcov : ndarray, optional
        User-supplied :math:`(k, k)` parameter covariance, used directly
        as ``Var(\betâ)``. Mutually exclusive with ``cov_type`` —
        passing both raises ``ValueError``.
    cov_kwds : dict, optional
        Extra keyword arguments forwarded to the covariance computation.
        Most importantly, ``{"groups": ...}`` for ``cov_type="cluster"``.

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

    **Inference options**

    Each of :meth:`predict`, :meth:`dydx`, and :meth:`did` accepts a
    common block of inference kwargs:

    - ``vce``: ``"delta"`` (default), ``"simulation"`` (Krinsky–Robb,
      ``n_sims`` draws of :math:`\beta_s \sim N(\hat\beta, \hat V)`),
      or ``"bootstrap"`` (refit on resampled rows; supports
      ``boot_method="pairs"|"cluster"|"block"``).
    - ``cov_type``/``vcov``/``cov_kwds``: same meaning as on the
      constructor, but applied per-call and overriding what's on
      ``self``.
    - ``ci_method``: ``"pointwise"`` (default), ``"bonferroni"``,
      ``"sidak"``, or ``"sup-t"``. The first three are post-hoc
      adjustments to the critical value; ``"sup-t"`` consumes the
      simulation/bootstrap draw matrix and so requires
      ``vce != "delta"``.

    The point estimate reported is always the analytic
    :math:`g(\hat\beta)` from the original fit, even under simulation
    or bootstrap — draws contribute only to SEs and intervals.

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

    Robust SEs and alternative VCEs::

        M.dydx("age", cov_type="HC3")              # heteroskedastic-robust
        M.dydx("age", cov_type="cluster",
               cov_kwds={"groups": df["clust"]})    # cluster-robust
        M.dydx("age", vce="simulation",
               n_sims=2000, sim_seed=0)            # Krinsky–Robb
        M.dydx("age", vce="bootstrap",
               n_boot=1000, boot_seed=0)           # pairs bootstrap

    Multiple-comparison adjustments for a family of margins::

        M.dydx(["age", "income"], ci_method="bonferroni")
        M.predict(atexog={"age": [25, 45, 65]},
                  vce="simulation", n_sims=2000,
                  ci_method="sup-t")

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
        cov_type: Optional[str] = None,
        vcov: Optional[np.ndarray] = None,
        cov_kwds: Optional[dict] = None,
        weights: Optional[Union[np.ndarray, Sequence[float]]] = None,
        weight_type: str = "sampling",
    ):
        self.results = results
        self.model = results.model
        params_arr = np.asarray(results.params, dtype=float)
        if params_arr.ndim == 2:
            # MNLogit and similar store params as (p, K-1) matrices.
            # statsmodels' flat vector and cov_params use Fortran (column-major)
            # order, so we must match that.
            self.params = params_arr.ravel(order="F")
        else:
            self.params = params_arr.ravel()
        self.cov = _get_param_cov(results, cov_type=cov_type, vcov=vcov, cov_kwds=cov_kwds)
        if self.cov.shape != (self.params.size, self.params.size):
            raise ValueError(
                f"cov_params has shape {self.cov.shape}, expected "
                f"({self.params.size}, {self.params.size})"
            )
        self.level = level
        self.df = getattr(results, "df_resid", None) if use_t else None
        self.analytic = analytic

        self._design = DesignResolver.from_results(results, data)
        self._raw_mode = self._design.raw_mode

        if self._design.frame is None and self._raw_mode:
            # Synthesize data from model.exog
            exog = np.asarray(self.model.exog)
            names = list(getattr(self.model, "exog_names", None) or
                         [f"x{i}" for i in range(exog.shape[1])])
            data = pd.DataFrame(exog, columns=names)
            self._design = DesignResolver.from_results(results, data)

        if self._design.frame is None:
            raise ValueError(
                "Could not retrieve the original data frame from the model; "
                "please pass ``data=`` explicitly."
            )
        self.data = self._design.frame

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

        self._engine = PredictionEngine.from_results(
            results, self._design,
            weights=weights, weight_type=weight_type,
            analytic=self.analytic,
        )

        self.weights = self._engine.weights
        self.weight_type = self._engine.weight_type

        self._derivs = DerivativeEngine(
            design=self._design, engine=self._engine,
        )

    @property
    def n_outcomes(self) -> int:
        """Number of outcome classes (K) for the fitted model."""
        return self._engine.n_outcomes

    @property
    def outcome_labels(self) -> Optional[List[str]]:
        """Outcome class labels for multi-outcome models, or None."""
        return self._engine.outcome_labels

    def _build_exog(self, frame: pd.DataFrame) -> np.ndarray:
        return self._design.build_exog(frame)

    def _predict(self, params: np.ndarray, exog: np.ndarray) -> np.ndarray:
        return self._engine.predict(params, exog)

    def _link_deriv(self):
        return self._engine.link_deriv()


    # ---- weights -----------------------------------------------------------


    # ---- model / data introspection ----


    # ---- building design matrices ----


    # ---- core prediction on the response scale ----


    # ---- analytic outer-Jacobian support ----



    # ---- scale resolution ---------------------------------------------------


    # ---- linear predictor and scaled prediction ------------------------------



    # ---- the inference worker ----

    def _delta(
        self,
        statistic: Callable[[np.ndarray], np.ndarray],
        labels: Sequence[str],
        stat_name: str,
        jac: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        outcome: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        vce: str = "delta",
        cov_type: Optional[str] = None,
        vcov: Optional[np.ndarray] = None,
        cov_kwds: Optional[dict] = None,
        n_sims: int = 2000,
        sim_seed: Optional[int] = None,
        n_boot: int = 1000,
        boot_seed: Optional[int] = None,
        boot_method: str = "pairs",
        cluster: Optional[np.ndarray] = None,
        block_size: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        ci_method: str = "pointwise",
        ci_alpha: float = 0.05,
        _margin_factory: Optional[Callable] = None,
    ) -> MarginsResult:
        r"""Apply inference to a vector-valued statistic.

        Supports delta-method, Krinsky–Robb simulation, and bootstrap VCE.
        """
        if vce not in ("delta", "simulation", "bootstrap"):
            raise ValueError(
                f"vce must be 'delta', 'simulation', or 'bootstrap', got {vce!r}"
            )
        if ci_method not in ("pointwise", "bonferroni", "sidak", "sup-t"):
            raise ValueError(
                f"ci_method must be 'pointwise', 'bonferroni', 'sidak', or 'sup-t', "
                f"got {ci_method!r}"
            )
        if ci_method == "sup-t" and vce == "delta":
            raise ValueError(
                "sup-t requires draws (use vce='simulation' or 'bootstrap')"
            )

        beta = self.params
        est_val = statistic(beta)
        est_arr = np.asarray(est_val, dtype=float)

        outcome_index: Optional[np.ndarray] = None
        outcome_labels_out: Optional[List[str]] = None

        if est_arr.ndim == 2 and est_arr.shape[1] > 1:
            m, K = est_arr.shape
            est = est_arr.ravel()
            expanded_labels: List[str] = []
            for lab in labels:
                for k in range(K):
                    suffix = (
                        self.outcome_labels[k]
                        if self.outcome_labels is not None
                        else str(k)
                    )
                    expanded_labels.append(f"{lab} ({suffix})")
            labels = expanded_labels
            outcome_index = np.tile(np.arange(K), m)
            outcome_labels_out = self.outcome_labels
        else:
            est = est_arr.ravel()

        # Apply outcome subsetting if requested
        mask = None
        if outcome is not None and self.n_outcomes > 1 and outcome_index is not None:
            keys = [outcome] if isinstance(outcome, (int, str)) else list(outcome)
            keep_indices: set[int] = set()
            for key in keys:
                if isinstance(key, str):
                    if self.outcome_labels is None or key not in self.outcome_labels:
                        raise ValueError(f"Outcome {key!r} not found")
                    keep_indices.add(self.outcome_labels.index(key))
                else:
                    idx = int(key)
                    if self.outcome_labels is not None and not (
                        0 <= idx < len(self.outcome_labels)
                    ):
                        raise ValueError(f"Outcome index {idx} out of range")
                    keep_indices.add(idx)
            mask = np.isin(outcome_index, list(keep_indices))
            if not np.any(mask):
                raise ValueError(f"No rows found for outcome(s) {keys}")
            est = est[mask]
            labels = [labels[i] for i in np.where(mask)[0]]
            outcome_index = outcome_index[mask]

        level = 1 - ci_alpha

        if vce == "delta":
            if jac is not None:
                J = np.atleast_2d(np.asarray(jac(beta), dtype=float))
            else:
                J = _central_jacobian(statistic, beta)
            if mask is not None:
                J = J[mask, :]
            if cov_type is not None or vcov is not None or cov_kwds is not None:
                param_cov = _get_param_cov(
                    self.results, cov_type=cov_type, vcov=vcov, cov_kwds=cov_kwds
                )
            else:
                param_cov = self.cov
            V = J @ param_cov @ J.T
            return MarginsResult(
                estimate=est, vcov=V, labels=labels,
                level=level, df=self.df, stat_name=stat_name,
                outcome_labels=outcome_labels_out,
                outcome_index=outcome_index,
                ci_method=ci_method,
            )

        elif vce == "simulation":
            param_cov = _get_param_cov(
                self.results, cov_type=cov_type, vcov=vcov, cov_kwds=cov_kwds
            )
            draws = _simulate_vce(
                beta, param_cov, statistic, n_sims=n_sims, seed=sim_seed
            )
            if mask is not None:
                draws = draws[:, mask]
            V, _ = _summarize_draws(draws, ddof=1)
            return MarginsResult(
                estimate=est, vcov=V, labels=labels,
                level=level, df=self.df, stat_name=stat_name,
                outcome_labels=outcome_labels_out,
                outcome_index=outcome_index,
                ci_method=ci_method,
                draws=draws,
            )

        elif vce == "bootstrap":
            if _margin_factory is None:
                raise ValueError("bootstrap requires _margin_factory")
            draws = _bootstrap_vce(
                self.results,
                _margin_factory,
                n_boot=n_boot,
                seed=boot_seed,
                method=boot_method,
                cluster=cluster,
                block_size=block_size,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            V, _ = _summarize_draws(draws, ddof=1)
            return MarginsResult(
                estimate=est, vcov=V, labels=labels,
                level=level, df=self.df, stat_name=stat_name,
                outcome_labels=outcome_labels_out,
                outcome_index=outcome_index,
                ci_method=ci_method,
                draws=draws,
            )

        # Unreachable
        raise RuntimeError(f"Unhandled vce={vce!r}")

    # ======================================================================}
    # Public API: adjusted predictions
    # ======================================================================}

    def predict(
            self,
            at: str = "overall",
            atexog: Optional[Mapping[str, object]] = None,
            factor_stat: Optional[str] = None,
            over: Optional[Union[str, List[str]]] = None,
            scale: Union[str, Transform] = "response",
            outcome: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
            vce: str = "delta",
            cov_type: Optional[str] = None,
            vcov: Optional[np.ndarray] = None,
            cov_kwds: Optional[dict] = None,
            n_sims: int = 2000,
            sim_seed: Optional[int] = None,
            n_boot: int = 1000,
            boot_seed: Optional[int] = None,
            boot_method: str = "pairs",
            cluster: Optional[np.ndarray] = None,
            block_size: Optional[int] = None,
            verbose: bool = False,
            n_jobs: int = 1,
            ci_method: str = "pointwise",
            ci_alpha: float = 0.05,
            values: Optional[Mapping[str, Any]] = None,
            default_values: str = "asobserved",
            newdata: Optional[pd.DataFrame] = None,
        ) -> MarginsResult:
        r"""Compute adjusted predictions.

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
        scale : str or Transform, default "response"
            Scale on which to report predictions. Built-in string values:
            ``"response"`` (default, model's own ``predict()``),
            ``"linear"`` (linear predictor η = Xβ, Stata's ``xb``),
            ``"pr"`` (probability scale; probability models only),
            ``"ir"`` (incidence rate; log-link count models only),
            ``"or"`` (odds ratio; logit only),
            ``"exp"`` (exp(η)), ``"log"`` (log of the response).
            A custom :class:`~smmargins.transforms.Transform` object is
            also accepted.
        outcome : int, str, or sequence, optional
            For multi-outcome models, subset to the specified outcome
            class(es).
        vce : {"delta", "simulation", "bootstrap"}, default "delta"
            Variance estimation method.
        cov_type : str, optional
            Recompute the parameter covariance using this covariance type
            (e.g. "HC1", "cluster"). Passed through to statsmodels.
        vcov : ndarray, optional
            User-supplied parameter covariance matrix. Mutually exclusive
            with ``cov_type``.
        cov_kwds : dict, optional
            Additional keywords for covariance computation (e.g.
            ``{"groups": ...}`` for cluster).
        n_sims : int, default 2000
            Number of draws for Krinsky–Robb simulation.
        sim_seed : int, optional
            Seed for simulation draws.
        n_boot : int, default 1000
            Number of bootstrap replications.
        boot_seed : int, optional
            Seed for bootstrap resampling.
        boot_method : {"pairs", "cluster", "block"}, default "pairs"
            Bootstrap resampling scheme.
        cluster : ndarray, optional
            Cluster IDs for cluster bootstrap.
        block_size : int, optional
            Block size for block bootstrap.
        verbose : bool, default False
            Show progress bar during bootstrap.
        n_jobs : int, default 1
            Parallel jobs for bootstrap. Requires ``joblib``.
        ci_method : {"pointwise", "bonferroni", "sidak", "sup-t"}, default "pointwise"
            Confidence interval method.
        ci_alpha : float, default 0.05
            Significance level for confidence intervals.

        Returns
        -------
        MarginsResult

        See Also
        --------
        Margins.dydx
        Margins.did
        """
        check_at(at)
        if factor_stat is None:
            factor_stat = default_factor_stat(at)
        check_factor_stat(factor_stat)

        transform = self._engine.resolve_scale(scale)

        if newdata is not None:
            if at != "overall" or atexog is not None or values is not None or over is not None:
                raise ValueError(
                    "newdata= is mutually exclusive with at/atexog/values/over. "
                    "Pass an arbitrary frame OR use the at/atexog/values DSL, not both."
                )
            self._design.validate_newdata(newdata)
            profile = _Profile(frame=newdata, collapse_design=False)
            frames = [newdata]
            labels = ["newdata"]
            weights_list = [None]
        else:
            profile, frames, labels, weights_list = self._design.expand_over(
                at, factor_stat, atexog, over=over,
                weights=self._engine.weights,
                get_weights=self._engine.get_weights,
                values=values,
                default_values=default_values,
            )

        stat_name = "prediction"
        if over is None and atexog is None and values is None and newdata is None:
            if at == "overall":
                labels = ["AAP"]
            elif at == "mean":
                labels = ["APM"]
            else:
                labels = [f"AP @ {at}"]

        Xs = [profile.materialize(f, self._design) for f in frames]

        if transform is None:
            # Response-scale path (default)
            def statistic(beta: np.ndarray) -> np.ndarray:
                parts = []
                for i, X in enumerate(Xs):
                    w = weights_list[i]
                    parts.append(
                        np.atleast_1d(
                            self._engine.weighted_mean(self._engine.predict(beta, X), weights=w)
                        )
                    )
                return np.vstack(parts)

            fprime = self._engine.link_deriv()
            if fprime is not None or self._engine._is_mnlogit():
                def jac(beta: np.ndarray) -> np.ndarray:
                    parts = []
                    for i, X in enumerate(Xs):
                        w = weights_list[i]
                        parts.append(self._engine.grad_mean_predict(X, beta, fprime, weights=w))
                    return np.vstack(parts)
            else:
                jac = None
        else:
            # Transformed scale path
            def statistic(beta: np.ndarray) -> np.ndarray:
                parts = []
                for i, X in enumerate(Xs):
                    w = weights_list[i]
                    parts.append(
                        np.atleast_1d(
                            self._engine.weighted_mean(
                                self._engine.predict_on_scale(beta, X, transform), weights=w
                            )
                        )
                    )
                return np.vstack(parts)

            def jac(beta: np.ndarray) -> np.ndarray:
                parts = []
                for i, X in enumerate(Xs):
                    w = weights_list[i]
                    parts.append(
                        self._engine.grad_mean_predict_on_scale(X, beta, transform, weights=w)
                    )
                return np.vstack(parts)

        if vce == "bootstrap":
            def _margin_factory(results_b):
                M = Margins(
                    results_b,
                    level=self.level,
                    use_t=self.df is not None,
                    analytic=self.analytic,
                )
                return M.predict(
                    at=at,
                    atexog=atexog,
                    factor_stat=factor_stat,
                    scale=scale,
                    outcome=outcome,
                    over=over,
                    values=values,
                    default_values=default_values,
                    vce="delta",
                )
        else:
            _margin_factory = None

        return self._delta(
            statistic, labels=labels, stat_name=stat_name, jac=jac,
            outcome=outcome,
            vce=vce,
            cov_type=cov_type,
            vcov=vcov,
            cov_kwds=cov_kwds,
            n_sims=n_sims,
            sim_seed=sim_seed,
            n_boot=n_boot,
            boot_seed=boot_seed,
            boot_method=boot_method,
            cluster=cluster,
            block_size=block_size,
            verbose=verbose,
            n_jobs=n_jobs,
            ci_method=ci_method,
            ci_alpha=ci_alpha,
            _margin_factory=_margin_factory,
        )

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
            over: Optional[Union[str, List[str]]] = None,
            method: str = "dydx",
            scale: Union[str, Transform] = "response",
            outcome: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
            vce: str = "delta",
            cov_type: Optional[str] = None,
            vcov: Optional[np.ndarray] = None,
            cov_kwds: Optional[dict] = None,
            n_sims: int = 2000,
            sim_seed: Optional[int] = None,
            n_boot: int = 1000,
            boot_seed: Optional[int] = None,
            boot_method: str = "pairs",
            cluster: Optional[np.ndarray] = None,
            block_size: Optional[int] = None,
            verbose: bool = False,
            n_jobs: int = 1,
            ci_method: str = "pointwise",
            ci_alpha: float = 0.05,
            values: Optional[Mapping[str, Any]] = None,
            default_values: str = "asobserved",
            newdata: Optional[pd.DataFrame] = None,
        ) -> MarginsResult:
        r"""Marginal effect of ``variable`` on the chosen scale.

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
        scale : str or Transform, default "response"
            Scale on which to report marginal effects. See :meth:`predict`
            for built-in values.
        outcome : int, str, or sequence, optional
            For multi-outcome models, subset to the specified outcome
            class(es).
        vce : {"delta", "simulation", "bootstrap"}, default "delta"
            Variance estimation method.
        cov_type : str, optional
            Recompute the parameter covariance using this covariance type.
        vcov : ndarray, optional
            User-supplied parameter covariance matrix.
        cov_kwds : dict, optional
            Additional keywords for covariance computation.
        n_sims : int, default 2000
            Number of draws for Krinsky–Robb simulation.
        sim_seed : int, optional
            Seed for simulation draws.
        n_boot : int, default 1000
            Number of bootstrap replications.
        boot_seed : int, optional
            Seed for bootstrap resampling.
        boot_method : {"pairs", "cluster", "block"}, default "pairs"
            Bootstrap resampling scheme.
        cluster : ndarray, optional
            Cluster IDs for cluster bootstrap.
        block_size : int, optional
            Block size for block bootstrap.
        verbose : bool, default False
            Show progress bar during bootstrap.
        n_jobs : int, default 1
            Parallel jobs for bootstrap. Requires ``joblib``.
        ci_method : {"pointwise", "bonferroni", "sidak", "sup-t"}, default "pointwise"
            Confidence interval method.
        ci_alpha : float, default 0.05
            Significance level for confidence intervals.

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
        check_at(at)
        if factor_stat is None:
            factor_stat = default_factor_stat(at)
        check_factor_stat(factor_stat)

        transform = self._engine.resolve_scale(scale)

        # Custom-transform validation for dydx
        if isinstance(scale, Transform) and transform is not None and transform.hess is None:
            raise ValueError(
                "Custom transforms used in dydx require an analytic second "
                "derivative (hess=). Predicted-value calls (Margins.predict) "
                "only need grad=."
            )

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

        if newdata is not None:
            if at != "overall" or atexog is not None or values is not None or over is not None:
                raise ValueError(
                    "newdata= is mutually exclusive with at/atexog/values/over. "
                    "Pass an arbitrary frame OR use the at/atexog/values DSL, not both."
                )
            self._design.validate_newdata(newdata)
            profile = _Profile(frame=newdata, collapse_design=False)
            frames = [newdata]
            at_labels = ["newdata"]
            weights_list = [None]
        else:
            profile, frames, at_labels, weights_list = self._design.expand_over(
                at, factor_stat, atexog, over=over,
                weights=self._engine.weights,
                get_weights=self._engine.get_weights,
                values=values,
                default_values=default_values,
            )

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
                    warnings.warn(
                        f"count=True applied to a float column ({v!r}); "
                        "the contrast x -> x+1 may not be physically meaningful.",
                        RuntimeWarning
                    )
                res_parts = self._derivs.count_components(
                    v, profile, frames, at_labels, transform=transform,
                    weights_list=weights_list,
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
                res_parts = self._derivs.discrete_components(
                    v, profile, frames, at_labels, reference=reference,
                    transform=transform, weights_list=weights_list,
                )
                parts.append(res_parts)
            else:
                res_parts = self._derivs.continuous_components(
                    v, profile, frames, at_labels, step=step, at=at,
                    method=method, transform=transform,
                    weights_list=weights_list,
                )
                parts.append(res_parts)

        def make_stacked_stat(p_list, m_obj):
            def stacked_stat(beta: np.ndarray) -> np.ndarray:
                res = []
                for fn, _, _, _ in p_list:
                    # fn is the statistic function for one variable.
                    # We MUST pass beta to it.
                    val = np.asarray(fn(beta), dtype=float)
                    res.append(val)
                return np.vstack(res)
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
                    val = np.asarray(j(beta), dtype=float)
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

        if vce == "bootstrap":
            def _margin_factory(results_b):
                M = Margins(
                    results_b,
                    level=self.level,
                    use_t=self.df is not None,
                    analytic=self.analytic,
                )
                return M.dydx(
                    variable=variable,
                    at=at,
                    atexog=atexog,
                    discrete=discrete,
                    count=count,
                    step=step,
                    reference=reference,
                    factor_stat=factor_stat,
                    over=over,
                    method=method,
                    scale=scale,
                    outcome=outcome,
                    values=values,
                    default_values=default_values,
                    vce="delta",
                )
        else:
            _margin_factory = None

        return self._delta(
            statistic, labels=labels, stat_name=stat_name, jac=jac_fn,
            outcome=outcome,
            vce=vce,
            cov_type=cov_type,
            vcov=vcov,
            cov_kwds=cov_kwds,
            n_sims=n_sims,
            sim_seed=sim_seed,
            n_boot=n_boot,
            boot_seed=boot_seed,
            boot_method=boot_method,
            cluster=cluster,
            block_size=block_size,
            verbose=verbose,
            n_jobs=n_jobs,
            ci_method=ci_method,
            ci_alpha=ci_alpha,
            _margin_factory=_margin_factory,
        )

    # ----- continuous case (numerical derivative) ---------------------------
    # ----- count case (unit increment x -> x+1) ----------------------------
    # ----- discrete case (contrast vs reference level) ---------------------
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
        scale: Union[str, Transform] = "response",
        vce: str = "delta",
        cov_type: Optional[str] = None,
        vcov: Optional[np.ndarray] = None,
        cov_kwds: Optional[dict] = None,
        ci_method: str = "pointwise",
        ci_alpha: float = 0.05,
        values: Optional[Mapping[str, Any]] = None,
        default_values: str = "asobserved",
    ) -> "DiDResult":
        r"""Difference-in-differences on the chosen scale.

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
            levels, on the chosen scale.
        group_levels, condition_levels : sequence of length 2, optional
            Which two levels to use (reference first, treated second).
            Default: the two smallest observed levels.
        at, atexog, factor_stat : see :meth:`predict`.
        scale : str or Transform, default "response"
            Scale on which to report predictions. See :meth:`predict`.
        vce : {"delta"}, default "delta"
            Only delta-method VCE is supported for DiD in this release.
        cov_type : str, optional
            Recompute the parameter covariance using this covariance type.
        vcov : ndarray, optional
            User-supplied parameter covariance matrix.
        cov_kwds : dict, optional
            Additional keywords for covariance computation.
        ci_method : {"pointwise", "bonferroni", "sidak", "sup-t"}, default "pointwise"
            Confidence interval method.
        ci_alpha : float, default 0.05
            Significance level for confidence intervals.

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

        For multi-outcome models (MNLogit, OrderedModel), each cell
        prediction is a K-vector of class probabilities. The contrast
        matrix lifts to a block-diagonal (3*K, 4*K) matrix via a
        Kronecker product with ``eye(K)``, so the returned ``DiDResult``
        carries the K-outcome axis on ``cells``, ``simple_effects``,
        ``did``, and ``joint``.  The ``joint`` field can be used for
        cross-outcome contrasts (e.g. ``joint.contrast(...)`` to compare
        the DiD of one class against another).

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
        if vce != "delta":
            raise NotImplementedError(
                "Only vce='delta' is supported for did() in this release"
            )
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
        cells = self.predict(
            at=at, atexog=atexog_full, factor_stat=factor_stat,
            scale=scale,
            vce=vce, cov_type=cov_type, vcov=vcov, cov_kwds=cov_kwds,
            ci_method=ci_method, ci_alpha=ci_alpha,
            values=values,
            default_values=default_values,
        )

        K = self.n_outcomes
        outcome_names = (
            self.outcome_labels if self.outcome_labels is not None
            else [str(i) for i in range(K)]
        )

        # Rewrite the cell labels to the compact "(group=A, condition=0)" form
        if K > 1:
            cells.labels = [
                f"{group}={gv}, {condition}={cv} | outcome={ok}"
                for gv in g for cv in c
                for ok in outcome_names
            ]
        else:
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
        if K > 1:
            C_all = np.kron(C_all, np.eye(K))  # shape (3*K, 4*K)

        if K > 1:
            contrast_labels = [
                f"{group}: {g[1]} vs {g[0]} | {condition}={c[0]} | outcome={ok}"
                for ok in outcome_names
            ] + [
                f"{group}: {g[1]} vs {g[0]} | {condition}={c[1]} | outcome={ok}"
                for ok in outcome_names
            ] + [
                f"DiD: {group}({g[1]}-{g[0]}) × {condition}({c[1]}-{c[0]}) | outcome={ok}"
                for ok in outcome_names
            ]
        else:
            contrast_labels = [
                f"{group}: {g[1]} vs {g[0]} | {condition}={c[0]}",
                f"{group}: {g[1]} vs {g[0]} | {condition}={c[1]}",
                f"DiD: {group}({g[1]}-{g[0]}) × {condition}({c[1]}-{c[0]})",
            ]

        all_contrasts = cells.contrast(
            C_all,
            labels=contrast_labels,
            name="estimate",
        )

        level = 1 - ci_alpha
        # Build joint result with outcome metadata for multi-outcome models
        joint_outcome_index = (np.tile(np.arange(K), 3) if K > 1 else None)
        joint = MarginsResult(
            estimate=all_contrasts.estimate,
            vcov=all_contrasts.vcov,
            labels=all_contrasts.labels,
            level=level, df=self.df, stat_name="estimate",
            outcome_labels=self.outcome_labels if K > 1 else None,
            outcome_index=joint_outcome_index,
            ci_method=ci_method,
        )

        # Slice back apart so users can grab each piece
        simple = MarginsResult(
            estimate=joint.estimate[:2 * K],
            vcov=joint.vcov[:2 * K, :2 * K],
            labels=joint.labels[:2 * K],
            level=level, df=self.df, stat_name="simple effect",
            outcome_labels=self.outcome_labels if K > 1 else None,
            outcome_index=(np.tile(np.arange(K), 2) if K > 1 else None),
            ci_method=ci_method,
        )
        did_ = MarginsResult(
            estimate=joint.estimate[2 * K:3 * K],
            vcov=joint.vcov[2 * K:3 * K, 2 * K:3 * K],
            labels=joint.labels[2 * K:3 * K],
            level=level, df=self.df, stat_name="DiD",
            outcome_labels=self.outcome_labels if K > 1 else None,
            outcome_index=(np.arange(K) if K > 1 else None),
            ci_method=ci_method,
        )
        return DiDResult(cells=cells, simple_effects=simple, did=did_,
                         joint=joint)

    def contrast(
        self,
        a: Optional[Mapping[str, Any]] = None,
        b: Optional[Mapping[str, Any]] = None,
        a_newdata: Optional[pd.DataFrame] = None,
        b_newdata: Optional[pd.DataFrame] = None,
        at: str = "overall",
        default_values: str = "asobserved",
        over: Optional[Union[str, List[str]]] = None,
        scale: Union[str, Transform] = "response",
        outcome: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        vce: str = "delta",
        cov_type: Optional[str] = None,
        vcov: Optional[np.ndarray] = None,
        cov_kwds: Optional[dict] = None,
        n_sims: int = 2000,
        sim_seed: Optional[int] = None,
        n_boot: int = 1000,
        boot_seed: Optional[int] = None,
        boot_method: str = "pairs",
        cluster: Optional[np.ndarray] = None,
        block_size: Optional[int] = None,
        verbose: bool = False,
        n_jobs: int = 1,
        ci_method: str = "pointwise",
        ci_alpha: float = 0.05,
    ) -> MarginsResult:
        r"""Joint contrast between two counterfactual specifications.

        Computes :math:`g_A(\beta)`, :math:`g_B(\beta)`, and
        :math:`g_A(\beta) - g_B(\beta)` with the full joint covariance,
        so the contrast SE is correct (including the off-diagonal term
        between A and B).

        Parameters
        ----------
        a, b : mapping, optional
            ``values=``-style specification for arm A and arm B.
            Mutually exclusive with ``a_newdata`` / ``b_newdata``.
        a_newdata, b_newdata : DataFrame, optional
            Escape-hatch frames for arm A and arm B.
            Mutually exclusive with ``a`` / ``b``.
        at, default_values, over : see :meth:`predict`.
        scale, outcome : see :meth:`predict`.
        vce, cov_type, vcov, cov_kwds, ci_method, ci_alpha : inference kwargs.

        Returns
        -------
        MarginsResult
            Three rows (or 3 blocks for multi-outcome): A, B, A-B.
        """
        # Validate arm specs
        if a_newdata is not None and a is not None:
            raise ValueError("Pass either a= or a_newdata=, not both")
        if b_newdata is not None and b is not None:
            raise ValueError("Pass either b= or b_newdata=, not both")
        if a is None and a_newdata is None:
            raise ValueError("Must provide a= or a_newdata=")
        if b is None and b_newdata is None:
            raise ValueError("Must provide b= or b_newdata=")

        # Resolve arm A
        if a_newdata is not None:
            self._design.validate_newdata(a_newdata)
            prof_a = _Profile(frame=a_newdata, collapse_design=False)
            frames_a = [a_newdata]
            labels_a = ["A"]
            weights_a = [None]
        else:
            prof_a, frames_a, labels_a, weights_a = self._design.expand_over(
                at, default_factor_stat(at), None, over=over,
                weights=self._engine.weights,
                get_weights=self._engine.get_weights,
                values=a,
                default_values=default_values,
            )

        # Resolve arm B
        if b_newdata is not None:
            self._design.validate_newdata(b_newdata)
            prof_b = _Profile(frame=b_newdata, collapse_design=False)
            frames_b = [b_newdata]
            labels_b = ["B"]
            weights_b = [None]
        else:
            prof_b, frames_b, labels_b, weights_b = self._design.expand_over(
                at, default_factor_stat(at), None, over=over,
                weights=self._engine.weights,
                get_weights=self._engine.get_weights,
                values=b,
                default_values=default_values,
            )

        # Materialize design matrices
        Xs_a = [prof_a.materialize(f, self._design) for f in frames_a]
        Xs_b = [prof_b.materialize(f, self._design) for f in frames_b]

        transform = self._engine.resolve_scale(scale)

        # Build stacked statistic [g_A, g_B, g_A - g_B]
        if transform is None:
            def _stacked_stat(beta: np.ndarray) -> np.ndarray:
                parts_a = []
                for i, X in enumerate(Xs_a):
                    w = weights_a[i]
                    parts_a.append(
                        np.atleast_1d(
                            self._engine.weighted_mean(
                                self._engine.predict(beta, X), weights=w
                            )
                        )
                    )
                parts_b = []
                for i, X in enumerate(Xs_b):
                    w = weights_b[i]
                    parts_b.append(
                        np.atleast_1d(
                            self._engine.weighted_mean(
                                self._engine.predict(beta, X), weights=w
                            )
                        )
                    )
                parts_a_2d = [p.reshape(1, -1) for p in parts_a]
                parts_b_2d = [p.reshape(1, -1) for p in parts_b]
                g_a = np.vstack(parts_a_2d)
                g_b = np.vstack(parts_b_2d)
                g_diff = g_a - g_b
                return np.vstack([g_a, g_b, g_diff])

            fprime = self._engine.link_deriv()
            if fprime is not None or self._engine._is_mnlogit():
                def _stacked_jac(beta: np.ndarray) -> np.ndarray:
                    parts_a = []
                    for i, X in enumerate(Xs_a):
                        w = weights_a[i]
                        parts_a.append(self._engine.grad_mean_predict(X, beta, fprime, weights=w))
                    parts_b = []
                    for i, X in enumerate(Xs_b):
                        w = weights_b[i]
                        parts_b.append(self._engine.grad_mean_predict(X, beta, fprime, weights=w))
                    J_a = np.vstack(parts_a)
                    J_b = np.vstack(parts_b)
                    J_diff = J_a - J_b
                    return np.vstack([J_a, J_b, J_diff])
            else:
                _stacked_jac = None
        else:
            def _stacked_stat(beta: np.ndarray) -> np.ndarray:
                parts_a = []
                for i, X in enumerate(Xs_a):
                    w = weights_a[i]
                    parts_a.append(
                        np.atleast_1d(
                            self._engine.weighted_mean(
                                self._engine.predict_on_scale(beta, X, transform), weights=w
                            )
                        )
                    )
                parts_b = []
                for i, X in enumerate(Xs_b):
                    w = weights_b[i]
                    parts_b.append(
                        np.atleast_1d(
                            self._engine.weighted_mean(
                                self._engine.predict_on_scale(beta, X, transform), weights=w
                            )
                        )
                    )
                parts_a_2d = [p.reshape(1, -1) for p in parts_a]
                parts_b_2d = [p.reshape(1, -1) for p in parts_b]
                g_a = np.vstack(parts_a_2d)
                g_b = np.vstack(parts_b_2d)
                g_diff = g_a - g_b
                return np.vstack([g_a, g_b, g_diff])

            def _stacked_jac(beta: np.ndarray) -> np.ndarray:
                parts_a = []
                for i, X in enumerate(Xs_a):
                    w = weights_a[i]
                    parts_a.append(
                        self._engine.grad_mean_predict_on_scale(X, beta, transform, weights=w)
                    )
                parts_b = []
                for i, X in enumerate(Xs_b):
                    w = weights_b[i]
                    parts_b.append(
                        self._engine.grad_mean_predict_on_scale(X, beta, transform, weights=w)
                    )
                J_a = np.vstack(parts_a)
                J_b = np.vstack(parts_b)
                J_diff = J_a - J_b
                return np.vstack([J_a, J_b, J_diff])

        # Labels
        if len(labels_a) != len(labels_b):
            raise ValueError(
                f"Arms A and B must have the same number of rows for subtraction, "
                f"got {len(labels_a)} vs {len(labels_b)}"
            )
        all_labels = [f"A: {lab}" for lab in labels_a] + [f"B: {lab}" for lab in labels_b] + [f"A - B: {lab}" for lab in labels_a]

        if vce == "bootstrap":
            def _margin_factory(results_b):
                M = Margins(
                    results_b,
                    level=self.level,
                    use_t=self.df is not None,
                    analytic=self.analytic,
                )
                return M.contrast(
                    a=a, b=b,
                    a_newdata=a_newdata, b_newdata=b_newdata,
                    at=at, default_values=default_values, over=over,
                    scale=scale, outcome=outcome,
                    vce="delta",
                )
        else:
            _margin_factory = None

        return self._delta(
            _stacked_stat, labels=all_labels, stat_name="contrast", jac=_stacked_jac,
            outcome=outcome,
            vce=vce,
            cov_type=cov_type,
            vcov=vcov,
            cov_kwds=cov_kwds,
            n_sims=n_sims,
            sim_seed=sim_seed,
            n_boot=n_boot,
            boot_seed=boot_seed,
            boot_method=boot_method,
            cluster=cluster,
            block_size=block_size,
            verbose=verbose,
            n_jobs=n_jobs,
            ci_method=ci_method,
            ci_alpha=ci_alpha,
            _margin_factory=_margin_factory,
        )

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

