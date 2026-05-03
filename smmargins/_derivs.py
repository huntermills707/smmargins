from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._design import DesignResolver
from ._engine import PredictionEngine
from .data import _Profile
from .transforms import Transform
from .utils import _METHOD_META, _central_jacobian

# Module-level testing flag: when True, continuous dydx forces FD
# Jacobian even when an analytic hess is available.
_DYDX_FD_ONLY = False


@dataclass
class DerivativeEngine:
    """Builds the (statistic, jacobian, labels) components for dydx
    calls. Knows continuous, count, and discrete paths and defers all
    prediction work to PredictionEngine.
    """
    design: DesignResolver
    engine: PredictionEngine

    @staticmethod
    def continuous_label_suffix(at: str) -> str:
        """Human-readable suffix for an evaluation profile."""
        return {"overall": "", "mean": " (at means)",
                "median": " (at medians)", "zero": " (at zero)"}[at]

    def continuous_labels(
        self, variable: str, method: str, at: str, at_labels: List[str],
        n_frames: int,
    ) -> List[str]:
        """Build column labels for a continuous marginal effect."""
        prefix = _METHOD_META[method]["prefix"]
        if n_frames == 1 and at_labels[0] == "":
            return [f"{prefix}{variable}{self.continuous_label_suffix(at)}"]
        return [f"{prefix}{variable} | {lab}" if lab else f"{prefix}{variable}"
                for lab in at_labels]

    def _design_pairs(
        self,
        profile: _Profile,
        frames: List[pd.DataFrame],
        perturb: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """Build paired design matrices for a perturbation-based contrast."""
        out: List[tuple[np.ndarray, np.ndarray]] = []
        for fr in frames:
            prepared = profile.prepare_frame(fr, self.design)
            fa, fb = perturb(prepared)
            Xa = self.design.build_exog(fa)
            Xb = self.design.build_exog(fb)
            if profile.collapse_design:
                Xa = Xa.mean(axis=0, keepdims=True)
                Xb = Xb.mean(axis=0, keepdims=True)
            out.append((Xa, Xb))
        return out

    def _paired_contrast(
        self,
        pairs: Sequence[tuple[np.ndarray, np.ndarray]],
        scale: float = 1.0,
        transform: Optional[Transform] = None,
        weights: Optional[Sequence[Optional[np.ndarray]]] = None,
    ) -> tuple[Callable[[np.ndarray], np.ndarray],
               Optional[Callable[[np.ndarray], np.ndarray]]]:
        """Compute ``E[f(X_a)] - E[f(X_b)]`` per pair, scaled by ``scale``."""
        n = len(pairs)

        if transform is None:
            def statistic(beta: np.ndarray) -> np.ndarray:
                parts = []
                for i, (Xa, Xb) in enumerate(pairs):
                    w = weights[i] if weights is not None else self.engine.get_weights(Xa.shape[0])
                    ya = self.engine.weighted_mean(self.engine.predict(beta, Xa), weights=w)
                    yb = self.engine.weighted_mean(self.engine.predict(beta, Xb), weights=w)
                    parts.append(np.atleast_1d((ya - yb) * scale))
                return np.vstack(parts)

            fprime = self.engine.link_deriv()
            if fprime is None and not self.engine._is_mnlogit():
                return statistic, None

            def jac(beta: np.ndarray) -> np.ndarray:
                parts = []
                for i, (Xa, Xb) in enumerate(pairs):
                    w = weights[i] if weights is not None else self.engine.get_weights(Xa.shape[0])
                    ga = self.engine.grad_mean_predict(Xa, beta, fprime, weights=w)
                    gb = self.engine.grad_mean_predict(Xb, beta, fprime, weights=w)
                    parts.append((ga - gb) * scale)
                return np.vstack(parts)

            return statistic, jac

        def statistic(beta: np.ndarray) -> np.ndarray:
            parts = []
            for i, (Xa, Xb) in enumerate(pairs):
                w = weights[i] if weights is not None else self.engine.get_weights(Xa.shape[0])
                ya = self.engine.weighted_mean(
                    self.engine.predict_on_scale(beta, Xa, transform), weights=w
                )
                yb = self.engine.weighted_mean(
                    self.engine.predict_on_scale(beta, Xb, transform), weights=w
                )
                parts.append(np.atleast_1d((ya - yb) * scale))
            return np.vstack(parts)

        def jac(beta: np.ndarray) -> np.ndarray:
            parts = []
            for i, (Xa, Xb) in enumerate(pairs):
                w = weights[i] if weights is not None else self.engine.get_weights(Xa.shape[0])
                ga = self.engine.grad_mean_predict_on_scale(Xa, beta, transform, weights=w)
                gb = self.engine.grad_mean_predict_on_scale(Xb, beta, transform, weights=w)
                parts.append((ga - gb) * scale)
            return np.vstack(parts)

        return statistic, jac

    def continuous_components(
        self,
        variable: str,
        profile: _Profile,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        step: Optional[float],
        at: str,
        method: str = "dydx",
        transform: Optional[Transform] = None,
        weights_list: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """Build statistic and Jacobian for a continuous derivative marginal effect."""
        if step is None:
            sd = float(self.design.frame[variable].std(ddof=0))
            scale = max(sd, abs(float(self.design.frame[variable].mean())), 1.0)
            step = (np.finfo(float).eps ** (1.0 / 3.0)) * scale
        h = float(step)

        labels = self.continuous_labels(variable, method, at, at_labels, len(frames))
        stat_name = _METHOD_META[method]["stat_name"]

        def perturb(fr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            x_col = np.asarray(fr[variable], dtype=float)
            fp = fr.copy(); fp[variable] = x_col + h
            fm = fr.copy(); fm[variable] = x_col - h
            return fp, fm

        if method == "dydx":
            pairs = self._design_pairs(profile, frames, perturb)
            statistic, jac_paired = self._paired_contrast(
                pairs, scale=1.0 / (2.0 * h), transform=transform,
                weights=weights_list,
            )

            if (
                not _DYDX_FD_ONLY
                and transform is not None
                and transform.hess is not None
                and not self.engine._is_mnlogit()
            ):
                jac_direct_parts = []
                for i, fr in enumerate(frames):
                    X = profile.materialize(fr, self.design)
                    w = weights_list[i] if weights_list is not None else None
                    jac_direct = self.engine.dydx_jac_direct(
                        X, self.engine.results.params.values, transform, variable, weights=w
                    )
                    if jac_direct is None:
                        break
                    jac_direct_parts.append(jac_direct)
                else:
                    def jac(beta: np.ndarray) -> np.ndarray:
                        parts = []
                        for i, fr in enumerate(frames):
                            X = profile.materialize(fr, self.design)
                            w = weights_list[i] if weights_list is not None else None
                            part = self.engine.dydx_jac_direct(
                                X, beta, transform, variable, weights=w
                            )
                            if part is None:
                                return jac_paired(beta)
                            parts.append(part)
                        return np.vstack(parts)
                    return (statistic, jac, labels, stat_name)

            return (statistic, jac_paired, labels, stat_name)

        need_y = method in ("eyex", "eydx")
        Xs_plus: List[np.ndarray] = []
        Xs_minus: List[np.ndarray] = []
        Xs_orig: List[np.ndarray] = []
        x_vals: List[np.ndarray] = []
        for fr in frames:
            x_col = np.asarray(fr[variable], dtype=float)
            fp, fm = perturb(fr)
            Xs_plus.append(profile.materialize(fp, self.design))
            Xs_minus.append(profile.materialize(fm, self.design))
            if need_y:
                Xs_orig.append(profile.materialize(fr, self.design))
            if profile.collapse_design:
                x_col = np.array([x_col.mean()])
            x_vals.append(x_col)

        if transform is None:
            _predict_fn = self.engine.predict
        else:
            def _predict_fn(params, exog):
                return self.engine.predict_on_scale(params, exog, transform)

        def statistic(beta: np.ndarray) -> np.ndarray:
            parts = []
            for i in range(len(Xs_plus)):
                dydx_i = (_predict_fn(beta, Xs_plus[i])
                          - _predict_fn(beta, Xs_minus[i])) / (2.0 * h)
                x_i = np.asarray(x_vals[i])
                if x_i.ndim == 1:
                    x_i = x_i[:, None]
                if method == "dyex":
                    contrib = dydx_i * x_i
                else:  # eyex / eydx
                    y_i = _predict_fn(beta, Xs_orig[i])
                    if np.any(y_i < 1e-12):
                        import warnings
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
                w = (weights_list[i] if weights_list is not None
                     else self.engine.get_weights(contrib.shape[0]))
                parts.append(
                    np.atleast_1d(self.engine.weighted_mean(contrib, weights=w))
                )
            return np.vstack(parts)

        return (statistic, None, labels, stat_name)

    def count_components(
        self,
        variable: str,
        profile: _Profile,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        transform: Optional[Transform] = None,
        weights_list: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """Build statistic and Jacobian for a count (unit increment) marginal effect."""
        profile = _Profile(
            frame=profile.frame,
            collapse_design=profile.collapse_design,
            collapse_numerics=profile.collapse_design,
        )

        def perturb(fr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            fp = fr.copy(); fp[variable] = fp[variable] + 1
            return fp, fr

        pairs = self._design_pairs(profile, frames, perturb)
        statistic, jac = self._paired_contrast(
            pairs, transform=transform, weights=weights_list,
        )
        base = f"{variable} (count)"
        labels = [f"{base} | {lab}" if lab else base for lab in at_labels]
        return (statistic, jac, labels, "dy/dx")

    def discrete_components(
        self,
        variable: str,
        profile: _Profile,
        frames: List[pd.DataFrame],
        at_labels: List[str],
        reference: Optional[object],
        transform: Optional[Transform] = None,
        weights_list: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """Build statistic and Jacobian for discrete contrasts of a factor variable."""
        profile = _Profile(
            frame=profile.frame,
            collapse_design=profile.collapse_design,
            collapse_numerics=profile.collapse_design,
        )

        levels = self.design.frame[variable].dropna().unique().tolist()
        try:
            levels = sorted(levels)
        except TypeError:
            pass
        if len(levels) < 2:
            raise ValueError(f"variable {variable!r} has <2 unique levels")
        if reference is None:
            reference = levels[0]
        others = [lv for lv in levels if lv != reference]

        pairs: List[tuple[np.ndarray, np.ndarray]] = []
        labels: List[str] = []
        for frame, at_lab in zip(frames, at_labels):
            f_ref = frame.copy(); f_ref[variable] = reference
            Xref = profile.materialize(f_ref, self.design)
            for lvl in others:
                f_lvl = frame.copy(); f_lvl[variable] = lvl
                pairs.append((profile.materialize(f_lvl, self.design), Xref))
                base = f"{variable}: {lvl} vs {reference}"
                labels.append(f"{base} | {at_lab}" if at_lab else base)

        pair_weights = None
        if weights_list is not None:
            pair_weights = []
            for w in weights_list:
                pair_weights.extend([w] * len(others))

        statistic, jac = self._paired_contrast(
            pairs, transform=transform, weights=pair_weights,
        )
        return (statistic, jac, labels, "contrast")
