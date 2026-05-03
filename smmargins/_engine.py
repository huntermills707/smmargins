from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._design import DesignResolver
from .transforms import Transform, _BUILT_IN_SCALES, _logistic_cdf, _logistic_pdf
from .utils import softmax_grad_mean


@dataclass
class PredictionEngine:
    """Predictions, gradients, scale dispatch, and weight resolution.

    Knows about the model's predict() function, link derivatives,
    family-specific quirks (MNLogit, OrderedModel, log-link Poisson),
    and the scale= dispatch table. Does NOT know about derivatives
    (dydx) or inference (vce).
    """
    results: Any
    design: DesignResolver
    weights: Optional[np.ndarray]
    weight_type: str
    n_outcomes: int
    outcome_labels: Optional[List[str]]
    n_train: Optional[int] = None
    mean_log_offset: Optional[float] = None
    analytic: bool = True

    @classmethod
    def from_results(
        cls,
        results,
        design: DesignResolver,
        weights=None,
        weight_type: str = "sampling",
        analytic: bool = True,
    ) -> "PredictionEngine":
        n_train = (
            results.model.exog.shape[0]
            if getattr(results.model, "exog", None) is not None
            else None
        )
        mean_log_offset = cls._compute_mean_log_offset(results)
        n_outcomes = cls._detect_n_outcomes(results)
        outcome_labels = cls._detect_outcome_labels(results, n_outcomes)
        weights, weight_type = cls._resolve_weights(
            results, weights, weight_type, n_train
        )
        return cls(
            results=results,
            design=design,
            weights=weights,
            weight_type=weight_type,
            n_outcomes=n_outcomes,
            outcome_labels=outcome_labels,
            n_train=n_train,
            mean_log_offset=mean_log_offset,
            analytic=analytic,
        )

    @staticmethod
    def _detect_n_outcomes(results) -> int:
        """Detect the number of outcome classes (K) for the fitted model."""
        model = results.model
        if hasattr(model, "J"):
            return int(model.J)
        if hasattr(model, "k_extra"):
            return int(model.k_extra) + 1
        try:
            exog = getattr(model, "exog", None)
            if exog is not None and exog.shape[0] > 0:
                probe = exog[:1]
                pred = model.predict(results.params, probe)
                pred_arr = np.asarray(pred)
                if pred_arr.ndim == 2:
                    return pred_arr.shape[1]
        except Exception:
            pass
        return 1

    @staticmethod
    def _detect_outcome_labels(results, n_outcomes: int) -> Optional[List[str]]:
        """Detect outcome class labels for multi-outcome models."""
        if n_outcomes == 1:
            return None
        model = results.model
        ynames = getattr(model, "_ynames_map", None)
        if ynames is not None:
            labels = [str(ynames.get(i, i)) for i in range(n_outcomes)]
            return labels
        endog = getattr(model, "endog", None)
        if endog is not None:
            try:
                uniq = np.unique(endog)
                if len(uniq) == n_outcomes:
                    return [str(u) for u in uniq]
            except Exception:
                pass
        return [str(i) for i in range(n_outcomes)]

    @staticmethod
    def _resolve_weights(
        results,
        weights: Optional[Union[np.ndarray, Sequence[float]]],
        weight_type: str,
        n_train: Optional[int],
    ) -> tuple[Optional[np.ndarray], str]:
        """Resolve and validate weights."""
        if weight_type not in ("sampling", "frequency"):
            raise ValueError(
                f"weight_type must be 'sampling' or 'frequency', "
                f"got {weight_type!r}"
            )

        if weights is not None:
            w = np.asarray(weights, dtype=float)
            if w.ndim != 1:
                raise ValueError("weights must be a 1-D array")
            if n_train is not None and w.shape[0] != n_train:
                raise ValueError(
                    f"weights has length {w.shape[0]} but model has "
                    f"{n_train} observations"
                )
            return w, weight_type

        from statsmodels.regression.linear_model import WLS
        model = results.model
        if isinstance(model, WLS):
            model_weights = getattr(results.model, "weights", None)
            if model_weights is not None:
                w = np.asarray(model_weights, dtype=float)
                if w.ndim == 1 and (n_train is None or w.shape[0] == n_train):
                    if not np.allclose(w, 1.0):
                        return w, weight_type

        fw = getattr(results.model, "freq_weights", None)
        if fw is not None:
            w = np.asarray(fw, dtype=float)
            if w.ndim == 1 and (n_train is None or w.shape[0] == n_train):
                if not np.allclose(w, 1.0):
                    return w, weight_type

        vw = getattr(results.model, "var_weights", None)
        if vw is not None:
            w = np.asarray(vw, dtype=float)
            if w.ndim == 1 and (n_train is None or w.shape[0] == n_train):
                if not np.allclose(w, 1.0):
                    return w, weight_type

        return None, weight_type

    @staticmethod
    def _compute_mean_log_offset(results) -> Optional[float]:
        """Mean of (offset + log(exposure)) across the training sample."""
        model = results.model
        oe = getattr(model, "_offset_exposure", None)
        if oe is not None:
            a = np.asarray(oe, dtype=float).ravel()
            if a.size > 0 and np.any(a):
                return float(np.mean(a))
            return None
        parts = []
        for attr in ("offset", "exposure"):
            v = getattr(model, attr, None)
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

    def get_weights(self, n_rows: int) -> Optional[np.ndarray]:
        """Return weights when they apply to a design of size ``n_rows``."""
        if self.weights is None:
            return None
        if self.n_train is not None and n_rows != self.n_train:
            return None
        return self.weights

    def weighted_mean(
        self,
        values: np.ndarray,
        weights: Optional[np.ndarray] = None,
        axis: int = 0,
    ) -> np.ndarray:
        """Weighted mean of ``values`` along ``axis``."""
        if weights is None:
            return np.mean(values, axis=axis)
        w = np.asarray(weights, dtype=float)
        if values.ndim == 1:
            return np.sum(values * w) / np.sum(w)
        return np.sum(values * w[:, None], axis=axis) / np.sum(w)

    def predict(self, params: np.ndarray, exog: np.ndarray) -> np.ndarray:
        """Return E[Y | X] (response scale) using the model's own predict."""
        exog_arr = np.asarray(exog)
        n_exog = exog_arr.shape[0]
        kwargs = {}
        if (self.n_train is not None
                and n_exog != self.n_train
                and self.mean_log_offset is not None):
            kwargs["offset"] = np.full(n_exog, self.mean_log_offset)

        params_use = np.asarray(params)
        if self._is_mnlogit():
            p = exog_arr.shape[1]
            K = self.n_outcomes
            params_use = params_use.reshape(p, K - 1, order="F")

        try:
            pred = np.asarray(self.results.model.predict(params_use, exog, **kwargs))
        except (TypeError, ValueError, KeyError):
            eta = exog_arr @ np.asarray(params_use)
            if "offset" in kwargs:
                eta = eta + kwargs["offset"]
            fam = getattr(self.results.model, "family", None)
            if fam is not None:
                pred = np.asarray(fam.link.inverse(eta))
            else:
                pred = eta
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        return pred

    def link_deriv(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Return the link derivative :math:`f'(\\eta)`, or ``None``."""
        if not self.analytic:
            return None
        model = self.results.model
        off = getattr(model, "_offset_exposure", None)
        if off is not None and np.any(np.asarray(off)):
            return None
        for attr in ("offset", "exposure"):
            v = getattr(model, attr, None)
            if v is not None and np.any(np.asarray(v)):
                return None

        fam = getattr(model, "family", None)
        if fam is not None:
            link = getattr(fam, "link", None)
            if link is not None and callable(getattr(link, "inverse_deriv", None)):
                return link.inverse_deriv
            return None

        try:
            from statsmodels.regression.linear_model import RegressionModel
        except ImportError:
            return None
        if isinstance(model, RegressionModel):
            return lambda eta: np.ones_like(np.asarray(eta, dtype=float))
        return None

    def grad_mean_predict(
        self,
        X: np.ndarray,
        beta: np.ndarray,
        fprime: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gradient of the mean prediction with respect to parameters."""
        X = np.asarray(X, dtype=float)
        if self._is_mnlogit():
            return softmax_grad_mean(X, beta, weights=weights)
        if fprime is None:
            raise ValueError(
                "fprime is required for single-outcome grad_mean_predict"
            )
        eta = X @ beta
        fp = np.asarray(fprime(eta), dtype=float).ravel()
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            grad = np.sum(fp[:, None] * X * w[:, None], axis=0) / np.sum(w)
        else:
            grad = (fp[:, None] * X).sum(axis=0) / X.shape[0]
        return grad.reshape(1, -1)

    def _is_mnlogit(self) -> bool:
        """Return True if the fitted model is MNLogit."""
        model = self.results.model
        return hasattr(model, "J") and getattr(model, "k_extra", None) == 0

    def _is_ordered_model(self) -> bool:
        """Return True if the fitted model is OrderedModel."""
        k_extra = getattr(self.results.model, "k_extra", None)
        return k_extra is not None and k_extra > 0

    def _is_probability_model(self) -> bool:
        """Return True if the model produces probabilities."""
        from statsmodels.discrete.discrete_model import Logit, Probit, MNLogit
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        return isinstance(self.results.model, (Logit, Probit, MNLogit, OrderedModel))

    def _is_logit_model(self) -> bool:
        """Return True if the fitted model is a logit family model."""
        from statsmodels.discrete.discrete_model import Logit, MNLogit
        if isinstance(self.results.model, (Logit, MNLogit)):
            return True
        fam = getattr(self.results.model, "family", None)
        if fam is not None:
            link = getattr(fam, "link", None)
            if link is not None and getattr(link, "name", None) == "logit":
                return True
        return False

    def _is_count_model_with_log_link(self) -> bool:
        """Return True if the model is a count model with log link."""
        fam = getattr(self.results.model, "family", None)
        if fam is not None:
            link = getattr(fam, "link", None)
            if link is not None and getattr(link, "name", None) == "log":
                return True
        return False

    def resolve_scale(self, scale):
        """Validate and resolve a ``scale=`` argument to a Transform or None."""
        if scale is None or scale == "response":
            return None

        if isinstance(scale, Transform):
            return scale

        if scale not in _BUILT_IN_SCALES and scale not in ("pr", "ir"):
            raise ValueError(
                f"Unknown scale={scale!r}. Built-ins: "
                f"{list(_BUILT_IN_SCALES)} + 'pr', 'ir'. "
                "Or pass a custom Transform object."
            )

        if scale == "pr":
            if not self._is_probability_model():
                raise ValueError(
                    'scale="pr" is only valid for probability models '
                    "(Logit, Probit, MNLogit, OrderedModel)."
                )
            return None

        if scale == "ir":
            if not self._is_count_model_with_log_link():
                raise ValueError(
                    'scale="ir" is only valid for log-link count models '
                    "(e.g. Poisson with log link)."
                )
            return None

        if scale == "or":
            if not self._is_logit_model():
                raise ValueError('scale="or" is only valid for logit models.')
            return _BUILT_IN_SCALES["or"]

        if scale == "log":
            return self.log_transform()

        return _BUILT_IN_SCALES[scale]

    def log_transform(self) -> Transform:
        """Build a Transform for ``scale='log'`` based on the model's link."""
        model = self.results.model
        fam = getattr(model, "family", None)
        link_name = None
        if fam is not None:
            link = getattr(fam, "link", None)
            link_name = getattr(link, "name", None)

        from statsmodels.regression.linear_model import RegressionModel
        if isinstance(model, RegressionModel):
            link_name = "identity"

        if link_name == "identity":
            return _BUILT_IN_SCALES["log"]
        if link_name == "log":
            return _BUILT_IN_SCALES["linear"]
        if link_name == "logit":
            return Transform(
                value=lambda eta: np.log(_logistic_cdf(np.asarray(eta, dtype=float))),
                grad=lambda eta: _logistic_pdf(np.asarray(eta, dtype=float))
                / _logistic_cdf(np.asarray(eta, dtype=float)),
                hess=lambda eta: (
                    _logistic_pdf(np.asarray(eta, dtype=float))
                    * (1.0 - 2.0 * _logistic_cdf(np.asarray(eta, dtype=float)))
                    * _logistic_cdf(np.asarray(eta, dtype=float))
                    - _logistic_pdf(np.asarray(eta, dtype=float)) ** 2
                )
                / _logistic_cdf(np.asarray(eta, dtype=float)) ** 2,
                name="log",
            )
        if link_name == "probit":
            from scipy import stats
            return Transform(
                value=lambda eta: np.log(stats.norm.cdf(np.asarray(eta, dtype=float))),
                grad=lambda eta: stats.norm.pdf(np.asarray(eta, dtype=float))
                / stats.norm.cdf(np.asarray(eta, dtype=float)),
                hess=lambda eta: (
                    -np.asarray(eta, dtype=float)
                    * stats.norm.pdf(np.asarray(eta, dtype=float))
                    * stats.norm.cdf(np.asarray(eta, dtype=float))
                    - stats.norm.pdf(np.asarray(eta, dtype=float)) ** 2
                )
                / stats.norm.cdf(np.asarray(eta, dtype=float)) ** 2,
                name="log",
            )

        raise ValueError(
            'scale="log" is not supported for this model; '
            "the link derivative is not available."
        )

    def linear_predictor(self, params: np.ndarray, exog: np.ndarray) -> np.ndarray:
        """Return the linear predictor η = Xβ (+ offset)."""
        exog_arr = np.asarray(exog, dtype=float)
        n_exog = exog_arr.shape[0]

        offset = 0.0
        if (
            self.n_train is not None
            and n_exog != self.n_train
            and self.mean_log_offset is not None
        ):
            offset = self.mean_log_offset

        params_use = np.asarray(params, dtype=float)

        if self._is_mnlogit():
            p = exog_arr.shape[1]
            K = self.n_outcomes
            B = params_use.reshape(p, K - 1, order="F")
            eta = exog_arr @ B
            if offset:
                eta = eta + offset
            return eta

        if self._is_ordered_model():
            p = exog_arr.shape[1]
            beta = params_use[:p]
            eta = exog_arr @ beta
            if offset:
                eta = eta + offset
            return eta

        eta = exog_arr @ params_use
        if offset:
            eta = eta + offset
        return eta

    def predict_on_scale(
        self,
        params: np.ndarray,
        exog: np.ndarray,
        transform: Transform,
    ) -> np.ndarray:
        """Return predictions on a transformed scale."""
        exog_arr = np.asarray(exog, dtype=float)
        eta = self.linear_predictor(params, exog_arr)

        if transform.name == "log" and np.any(eta <= 0):
            raise ValueError(
                'scale="log" requires strictly positive predictions; '
                f"some linear predictors are <= 0 for this "
                f"{self.results.model.__class__.__name__} model."
            )

        if self._is_mnlogit():
            n = eta.shape[0]
            eta_full = np.hstack([np.zeros((n, 1)), eta])
            g_eta = np.asarray(transform.value(eta_full), dtype=float)
            if g_eta.ndim == 1:
                g_eta = g_eta.reshape(-1, 1)
            return g_eta

        g_eta = np.asarray(transform.value(eta), dtype=float)
        if g_eta.ndim == 1:
            g_eta = g_eta.reshape(-1, 1)
        elif g_eta.ndim != 2 or g_eta.shape[0] != eta.shape[0]:
            raise ValueError(
                f"transform.value returned shape {g_eta.shape} but expected "
                f"({eta.shape[0]},) or ({eta.shape[0]}, 1) for single-outcome model."
            )
        return g_eta

    def grad_mean_predict_on_scale(
        self,
        X: np.ndarray,
        beta: np.ndarray,
        transform: Transform,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gradient of mean prediction on a transformed scale w.r.t. β."""
        X = np.asarray(X, dtype=float)

        if self._is_mnlogit():
            return self.mnlogit_grad_mean_scaled(X, beta, transform, weights=weights)

        if self._is_ordered_model():
            p = X.shape[1]
            beta_exog = np.asarray(beta)[:p]
            eta = X @ beta_exog
            if transform.name == "log" and np.any(eta <= 0):
                raise ValueError(
                    'scale="log" requires strictly positive predictions; '
                    f"some linear predictors are <= 0 for this "
                    f"{self.results.model.__class__.__name__} model."
                )
            gp = np.asarray(transform.grad(eta), dtype=float).ravel()
            if weights is not None:
                w = np.asarray(weights, dtype=float)
                grad = np.sum(gp[:, None] * X * w[:, None], axis=0) / np.sum(w)
            else:
                grad = (gp[:, None] * X).sum(axis=0) / X.shape[0]
            k_extra = getattr(self.results.model, "k_extra", 0)
            if k_extra > 0:
                grad = np.concatenate([grad, np.zeros(k_extra)])
            return grad.reshape(1, -1)

        eta = X @ beta
        if transform.name == "log" and np.any(eta <= 0):
            raise ValueError(
                'scale="log" requires strictly positive predictions; '
                f"some linear predictors are <= 0 for this "
                f"{self.results.model.__class__.__name__} model."
            )
        gp = np.asarray(transform.grad(eta), dtype=float)
        if gp.shape != eta.shape:
            raise ValueError(
                f"transform.grad returned shape {gp.shape} but expected "
                f"{eta.shape} for single-outcome model."
            )
        gp = gp.ravel()
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            grad = np.sum(gp[:, None] * X * w[:, None], axis=0) / np.sum(w)
        else:
            grad = (gp[:, None] * X).sum(axis=0) / X.shape[0]
        return grad.reshape(1, -1)

    def dydx_jac_direct(
        self,
        X: np.ndarray,
        beta: np.ndarray,
        transform: Transform,
        variable: str,
        weights: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Direct analytic Jacobian of AME_k w.r.t. β using transform.hess."""
        try:
            col_idx = self.results.model.exog_names.index(variable)
        except ValueError:
            return None

        X = np.asarray(X, dtype=float)
        beta = np.asarray(beta, dtype=float)
        eta = X @ beta
        if transform.name == "log" and np.any(eta <= 0):
            raise ValueError(
                'scale="log" requires strictly positive predictions; '
                f"some linear predictors are <= 0 for this "
                f"{self.results.model.__class__.__name__} model."
            )
        gp = np.asarray(transform.grad(eta), dtype=float)
        if gp.shape != eta.shape:
            raise ValueError(
                f"transform.grad returned shape {gp.shape} but expected "
                f"{eta.shape} for single-outcome model."
            )
        gpp = np.asarray(transform.hess(eta), dtype=float)
        if gpp.shape != eta.shape:
            raise ValueError(
                f"transform.hess returned shape {gpp.shape} but expected "
                f"{eta.shape} for single-outcome model."
            )
        gp = gp.ravel()
        gpp = gpp.ravel()

        beta_k = beta[col_idx]

        if weights is not None:
            w = np.asarray(weights, dtype=float)
            sw = np.sum(w)
            mean_gpp_X = np.sum(gpp[:, None] * X * w[:, None], axis=0) / sw
            mean_gp = np.sum(gp * w) / sw
        else:
            mean_gpp_X = np.mean(gpp[:, None] * X, axis=0)
            mean_gp = np.mean(gp)

        J = mean_gpp_X * beta_k
        J[col_idx] += mean_gp
        return J.reshape(1, -1)

    def mnlogit_grad_mean_scaled(
        self,
        X: np.ndarray,
        beta: np.ndarray,
        transform: Transform,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Analytic gradient of mean transformed linear predictors for MNLogit."""
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        K = self.n_outcomes
        B = beta.reshape(p, K - 1, order="F")
        eta = X @ B  # (n, K-1)
        eta_full = np.hstack([np.zeros((n, 1)), eta])  # (n, K)

        gp = np.asarray(transform.grad(eta_full), dtype=float)  # (n, K)
        w = np.asarray(weights, dtype=float) if weights is not None else None
        sw = np.sum(w) if w is not None else n

        grad = np.zeros((K, p * (K - 1)))
        for k in range(1, K):
            block_start = (k - 1) * p
            block_end = k * p
            if w is not None:
                grad[k, block_start:block_end] = (
                    np.sum(gp[:, k][:, None] * X * w[:, None], axis=0) / sw
                )
            else:
                grad[k, block_start:block_end] = (
                    (gp[:, k][:, None] * X).sum(axis=0) / n
                )
        return grad
