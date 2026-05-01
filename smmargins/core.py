from __future__ import annotations

import itertools
import warnings
from typing import Callable, Optional, Union, Sequence, Mapping, List

import numpy as np
import pandas as pd
import patsy

from .results import MarginsResult, DiDResult
from .data import _Profile
from .utils import _central_jacobian, _METHOD_META

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
        params_arr = np.asarray(results.params, dtype=float)
        if params_arr.ndim == 2:
            # MNLogit and similar store params as (p, K-1) matrices.
            # statsmodels' flat vector and cov_params use Fortran (column-major)
            # order, so we must match that.
            self.params = params_arr.ravel(order="F")
        else:
            self.params = params_arr.ravel()
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
        For OrderedModel, ``exog_names`` includes threshold parameters;
        we truncate to the actual number of exog columns.
        """
        if self._raw_mode:
            names = list(self.model.exog_names)
            n_exog = self.model.exog.shape[1]
            cols = names[:n_exog]
            return np.asarray(frame[cols].to_numpy(dtype=float))
        dm = patsy.dmatrix(self.design_info, frame, return_type="matrix")
        return np.asarray(dm)

    # ---- core prediction on the response scale ----

    def _predict(self, params: np.ndarray, exog: np.ndarray) -> np.ndarray:
        """Return E[Y | X] (response scale) using the model's own predict.

        Parameters
        ----------
        params : ndarray of shape (p,) or (p, K-1)
            Parameter vector override.
        exog : ndarray of shape (n, p)
            Design matrix.

        Returns
        -------
        ndarray of shape (n, K)
            Predicted values on the response scale. Single-outcome models
            return ``(n, 1)``; multi-outcome models return ``(n, K)``.

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

        # Multi-outcome models may need params in matrix form
        params_use = np.asarray(params)
        if self._is_mnlogit():
            p = exog_arr.shape[1]
            K = self.n_outcomes
            params_use = params_use.reshape(p, K - 1, order="F")

        try:
            pred = np.asarray(self.model.predict(params_use, exog, **kwargs))
        except (TypeError, ValueError, KeyError):
            eta = exog_arr @ np.asarray(params_use)
            if "offset" in kwargs:
                eta = eta + kwargs["offset"]
            fam = getattr(self.model, "family", None)
            if fam is not None:
                pred = np.asarray(fam.link.inverse(eta))
            else:
                pred = eta
        # Enforce (n, K) shape invariant
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

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

    def _grad_mean_predict(
        self,
        X: np.ndarray,
        beta: np.ndarray,
        fprime: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        r"""Gradient of the mean prediction with respect to parameters.

        Computes
        :math:`\partial/\partial\beta` of
        :math:`(1/n)\sum_i f(x_i^\top\beta)`, returning shape ``(K, p)``.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Design matrix.
        beta : ndarray of shape (p,) or (p, K-1)
            Parameter vector.
        fprime : callable, optional
            Function returning :math:`f'(\eta)` for each linear
            predictor value. Required for single-outcome models.

        Returns
        -------
        ndarray of shape (K, p_full)
            Gradient matrix. Single-outcome models return ``(1, p)``;
            MNLogit returns ``(K, p*(K-1))``.

        Notes
        -----
        Building block for every analytic Jacobian in this module: every
        statistic we compute is a linear combination of mean-predictions
        on different design matrices, so each row of the analytic ``J``
        is a linear combination of these gradients.
        """
        X = np.asarray(X, dtype=float)
        if self._is_mnlogit():
            return self._softmax_grad_mean(X, beta)
        # Single-outcome path (OLS, GLM, etc.)
        if fprime is None:
            raise ValueError(
                "fprime is required for single-outcome _grad_mean_predict"
            )
        eta = X @ beta
        fp = np.asarray(fprime(eta), dtype=float).ravel()
        grad = (fp[:, None] * X).sum(axis=0) / X.shape[0]
        return grad.reshape(1, -1)

    def _is_mnlogit(self) -> bool:
        """Return True if the fitted model is MNLogit."""
        return hasattr(self.model, "J") and getattr(self.model, "k_extra", None) == 0

    def _is_ordered_model(self) -> bool:
        """Return True if the fitted model is OrderedModel."""
        k_extra = getattr(self.model, "k_extra", None)
        return k_extra is not None and k_extra > 0

    @staticmethod
    def _softmax_grad_mean(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        r"""Analytic gradient of mean softmax probabilities for MNLogit.

        For a K-class multinomial logit with class 0 as reference,
        computes the gradient of
        :math:`(1/n)\sum_i P(Y=c \mid x_i)` w.r.t. the flat parameter
        vector for every class c.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Design matrix.
        beta : ndarray of shape (p*(K-1),)
            Flat parameter vector in column-major (Fortran) order:
            ``[beta_1, beta_2, ..., beta_{K-1}]`` where each ``beta_k``
            has length ``p``.

        Returns
        -------
        ndarray of shape (K, p*(K-1))
            Gradient matrix. Row ``c`` is the gradient for outcome class ``c``.
        """
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        # beta is flat vector of length p*(K-1)
        # Reshape to (p, K-1) in Fortran order (class blocks contiguous)
        K = beta.size // p + 1
        B = beta.reshape(p, K - 1, order="F")
        # Linear predictors for classes 1..K-1
        eta = X @ B  # (n, K-1)
        # Probabilities
        exp_eta = np.exp(eta)
        denom = 1 + exp_eta.sum(axis=1, keepdims=True)  # (n, 1)
        P_alt = exp_eta / denom  # (n, K-1)
        P_0 = 1 / denom  # (n, 1)
        P = np.hstack([P_0, P_alt])  # (n, K)

        # delta_{c,k} for c=0..K-1, k=1..K-1
        delta = np.zeros((K, K - 1))
        for k in range(1, K):
            delta[k, k - 1] = 1.0

        # weight[i, c, k] = P_{c,i} * (delta_{c,k} - P_{k,i})
        diff = delta[None, :, :] - P[:, None, 1:]  # (n, K, K-1)
        weight = P[:, :, None] * diff  # (n, K, K-1)

        # grad_{c,k} = (1/n) sum_i weight[i,c,k] * x_i
        grad_3d = np.einsum("nck,np->ckp", weight, X) / n  # (K, K-1, p)
        return grad_3d.reshape(K, p * (K - 1))

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
            parts = []
            for i, (Xa, Xb) in enumerate(pairs):
                ya = np.mean(self._predict(beta, Xa), axis=0)
                yb = np.mean(self._predict(beta, Xb), axis=0)
                parts.append(np.atleast_1d((ya - yb) * scale))
            return np.vstack(parts)

        fprime = self._link_deriv()
        if fprime is None and not self._is_mnlogit():
            return statistic, None

        def jac(beta: np.ndarray) -> np.ndarray:
            parts = []
            for i, (Xa, Xb) in enumerate(pairs):
                ga = self._grad_mean_predict(Xa, beta, fprime)
                gb = self._grad_mean_predict(Xb, beta, fprime)
                parts.append((ga - gb) * scale)
            return np.vstack(parts)

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
        outcome: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
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

        For multi-outcome models, the statistic returns shape ``(m, K)``.
        The estimate is flattened row-major (outcome varies fastest) and
        labels are expanded to ``m * K`` entries. The Jacobian is expected
        to already be ``(m*K, p)``.
        """
        beta = self.params
        est_val = statistic(beta)
        est_arr = np.asarray(est_val, dtype=float)

        outcome_index: Optional[np.ndarray] = None
        outcome_labels_out: Optional[List[str]] = None

        if est_arr.ndim == 2 and est_arr.shape[1] > 1:
            m, K = est_arr.shape
            est = est_arr.ravel()
            # Expand labels: each base label repeated K times with outcome suffix
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

        if jac is not None:
            J = np.atleast_2d(np.asarray(jac(beta), dtype=float))
        else:
            # We must pass the user-provided 'statistic' directly to the
            # numerical differentiator.
            J = _central_jacobian(statistic, beta)

        V = J @ self.cov @ J.T

        # Apply outcome subsetting if requested
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
            J = J[mask, :]
            V = V[np.ix_(mask, mask)]
            labels = [labels[i] for i in np.where(mask)[0]]
            outcome_index = outcome_index[mask]

        return MarginsResult(
            estimate=est, vcov=V, labels=labels,
            level=self.level, df=self.df, stat_name=stat_name,
            outcome_labels=outcome_labels_out,
            outcome_index=outcome_index,
        )

    # ======================================================================}
    # Public API: adjusted predictions
    # ======================================================================}

    def predict(
            self,
            at: str = "overall",
            atexog: Optional[Mapping[str, object]] = None,
            factor_stat: Optional[str] = None,
            outcome: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
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
            parts = []
            for i, X in enumerate(Xs):
                parts.append(np.atleast_1d(np.mean(self._predict(beta, X), axis=0)))
            return np.vstack(parts)

        fprime = self._link_deriv()
        if fprime is not None or self._is_mnlogit():
            def jac(beta: np.ndarray) -> np.ndarray:
                parts = []
                for i, X in enumerate(Xs):
                    parts.append(self._grad_mean_predict(X, beta, fprime))
                return np.vstack(parts)
        else:
            jac = None

        return self._delta(
            statistic, labels=labels, stat_name=stat_name, jac=jac,
            outcome=outcome,
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
            method: str = "dydx",
            outcome: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
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

        return self._delta(
            statistic, labels=labels, stat_name=stat_name, jac=jac_fn,
            outcome=outcome,
        )

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
            parts = []
            for i in range(len(Xs_plus)):
                dydx_i = (self._predict(beta, Xs_plus[i])
                          - self._predict(beta, Xs_minus[i])) / (2.0 * h)
                x_i = np.asarray(x_vals[i])
                if x_i.ndim == 1:
                    x_i = x_i[:, None]
                if method == "dyex":
                    contrib = dydx_i * x_i
                else:  # eyex / eydx
                    y_i = self._predict(beta, Xs_orig[i])
                    if np.any(y_i < 1e-12):
                        warnings.warn(
                            "predicted probabilities below 1e-12 for some "
                            "outcome classes; elasticities for those classes "
                            "are unstable",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    if method == "eyex":
                        contrib = dydx_i * x_i / y_i
                    else:  # eydx
                        contrib = dydx_i / y_i
                parts.append(np.atleast_1d(np.mean(contrib, axis=0)))
            return np.vstack(parts)

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

        # Build joint result with outcome metadata for multi-outcome models
        joint_outcome_index = (np.tile(np.arange(K), 3) if K > 1 else None)
        joint = MarginsResult(
            estimate=all_contrasts.estimate,
            vcov=all_contrasts.vcov,
            labels=all_contrasts.labels,
            level=self.level, df=self.df, stat_name="estimate",
            outcome_labels=self.outcome_labels if K > 1 else None,
            outcome_index=joint_outcome_index,
        )

        # Slice back apart so users can grab each piece
        simple = MarginsResult(
            estimate=joint.estimate[:2 * K],
            vcov=joint.vcov[:2 * K, :2 * K],
            labels=joint.labels[:2 * K],
            level=self.level, df=self.df, stat_name="simple effect",
            outcome_labels=self.outcome_labels if K > 1 else None,
            outcome_index=(np.tile(np.arange(K), 2) if K > 1 else None),
        )
        did_ = MarginsResult(
            estimate=joint.estimate[2 * K:3 * K],
            vcov=joint.vcov[2 * K:3 * K, 2 * K:3 * K],
            labels=joint.labels[2 * K:3 * K],
            level=self.level, df=self.df, stat_name="DiD",
            outcome_labels=self.outcome_labels if K > 1 else None,
            outcome_index=(np.arange(K) if K > 1 else None),
        )
        return DiDResult(cells=cells, simple_effects=simple, did=did_,
                         joint=joint)

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


