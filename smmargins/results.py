from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Union, Sequence, List


class MarginsResult:
    """Container for marginal effects or adjusted predictions.

    This class holds the point estimates, their delta-method covariance
    matrix, and provides methods for summarizing results, computing
    linear contrasts, or subsetting multi-outcome results.

    Attributes
    ----------
    estimate : ndarray
        Point estimates for each margin.
    vcov : ndarray
        The estimated covariance matrix of the margins (delta method
        or empirical covariance of simulation/bootstrap draws).
    se : ndarray
        Standard errors (square root of the diagonal of ``vcov``, or
        the standard deviation of draws when available).
    tvalues : ndarray
        Test statistics (estimate / se).
    pvalues : ndarray
        Two-sided p-values based on either the normal or t distribution.
    labels : list of str
        Labels for each row of the results.
    ci_method : str
        Method used to compute confidence intervals ("pointwise",
        "bonferroni", "sidak", or "sup-t").
    draws : ndarray or None
        Simulation/bootstrap draw matrix of shape (S, m) when available.
    """

    def __init__(
        self,
        estimate: np.ndarray,
        vcov: np.ndarray,
        labels: Optional[Sequence[str]] = None,
        level: float = 0.95,
        df: Optional[int] = None,
        stat_name: str = "margin",
        outcome_labels: Optional[Sequence[str]] = None,
        outcome_index: Optional[np.ndarray] = None,
        ci_method: str = "pointwise",
        draws: Optional[np.ndarray] = None,
    ):
        self.estimate = np.asarray(estimate)
        self.vcov = np.asarray(vcov)
        self.labels = list(labels) if labels is not None else [str(i) for i in range(len(self.estimate))]
        self.level = level
        self.df = df
        self.stat_name = stat_name
        self.outcome_labels = list(outcome_labels) if outcome_labels is not None else None
        self.outcome_index = np.asarray(outcome_index) if outcome_index is not None else None
        self.ci_method = ci_method
        self.draws = draws

    @property
    def se(self) -> np.ndarray:
        if self.draws is not None:
            return np.std(self.draws, axis=0, ddof=1)
        return np.sqrt(np.maximum(np.diag(self.vcov), 0))

    @property
    def tvalues(self) -> np.ndarray:
        return self.estimate / self.se

    @property
    def pvalues(self) -> np.ndarray:
        if self.df is not None:
            return 2 * stats.t.sf(np.abs(self.tvalues), self.df)
        return 2 * stats.norm.sf(np.abs(self.tvalues))

    def _crit_value(self) -> float:
        """Critical value for confidence intervals."""
        alpha = 1 - self.level
        m = len(self.estimate)
        if self.ci_method == "bonferroni":
            alpha_adj = alpha / m
        elif self.ci_method == "sidak":
            alpha_adj = 1 - (1 - alpha) ** (1 / m)
        elif self.ci_method == "sup-t":
            if self.draws is None:
                raise ValueError(
                    "sup-t requires draws (use vce='simulation' or 'bootstrap')"
                )
            se_draws = np.std(self.draws, axis=0, ddof=1)
            valid = se_draws > 1e-14
            if not np.any(valid):
                return np.inf
            # Standardize deviations around the analytic point estimate
            # (conservative relative to centering on the draw mean).
            devs = np.abs(self.draws[:, valid] - self.estimate[None, valid]) / se_draws[None, valid]
            sup_devs = np.max(devs, axis=1)
            return float(np.quantile(sup_devs, 1 - alpha))
        else:  # pointwise
            alpha_adj = alpha

        if self.df is not None:
            return float(stats.t.ppf(1 - alpha_adj / 2, self.df))
        return float(stats.norm.ppf(1 - alpha_adj / 2))

    @property
    def ci_lower(self) -> np.ndarray:
        """Lower confidence bound.

        For ``ci_method="pointwise"`` with simulation/bootstrap draws,
        returns an empirical percentile (asymmetric) rather than the
        symmetric ``estimate - crit * se`` used for delta-method VCE.
        """
        if self.ci_method == "pointwise" and self.draws is not None:
            return np.percentile(self.draws, 100 * (1 - self.level) / 2, axis=0)
        crit = self._crit_value()
        return self.estimate - crit * self.se

    @property
    def ci_upper(self) -> np.ndarray:
        """Upper confidence bound.

        For ``ci_method="pointwise"`` with simulation/bootstrap draws,
        returns an empirical percentile (asymmetric) rather than the
        symmetric ``estimate + crit * se`` used for delta-method VCE.
        """
        if self.ci_method == "pointwise" and self.draws is not None:
            return np.percentile(self.draws, 100 * (1 + self.level) / 2, axis=0)
        crit = self._crit_value()
        return self.estimate + crit * self.se

    def contrast(
        self,
        c: Union[Sequence[float], np.ndarray],
        labels: Optional[Sequence[str]] = None,
        name: str = "contrast",
    ) -> "MarginsResult":
        """Compute linear contrasts of the estimates.

        Parameters
        ----------
        c : array-like
            Contrast vector (length ``k``) or matrix (shape ``(m, k)``)
            to apply to the estimates.
        labels : list of str, optional
            Labels for the new contrast(s).
        name : str
            Column header for the summary table.

        Returns
        -------
        MarginsResult
            A new result object containing the contrasted estimates and
            their joint covariance.
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
        draws = None
        if self.draws is not None:
            draws = self.draws @ C.T
        return MarginsResult(
            estimate=est, vcov=vcov, labels=labels,
            level=self.level, df=self.df, stat_name=name,
            ci_method=self.ci_method,
            draws=draws,
        )

    def outcome(self, k: Union[int, str, Sequence[Union[int, str]]]) -> "MarginsResult":
        """Slice rows belonging to the specified outcome class(es).

        Parameters
        ----------
        k : int, str, or sequence of int/str
            Outcome class label(s) or index/ices to retain.

        Returns
        -------
        MarginsResult
            A new result with only the rows for the chosen outcome(s).
        """
        if self.outcome_index is None:
            raise ValueError(
                "outcome() is only valid for multi-outcome results"
            )
        keys = [k] if isinstance(k, (int, str)) else list(k)
        indices: set[int] = set()
        for key in keys:
            if isinstance(key, str):
                if self.outcome_labels is None or key not in self.outcome_labels:
                    raise ValueError(f"Outcome {key!r} not found")
                indices.add(self.outcome_labels.index(key))
            else:
                idx = int(key)
                if self.outcome_labels is not None and not (0 <= idx < len(self.outcome_labels)):
                    raise ValueError(f"Outcome index {idx} out of range")
                indices.add(idx)

        mask = np.isin(self.outcome_index, list(indices))
        if not np.any(mask):
            raise ValueError(f"No rows found for outcome(s) {keys}")

        draws = self.draws[:, mask] if self.draws is not None else None
        return MarginsResult(
            estimate=self.estimate[mask],
            vcov=self.vcov[np.ix_(mask, mask)],
            labels=[self.labels[i] for i in np.where(mask)[0]],
            level=self.level,
            df=self.df,
            stat_name=self.stat_name,
            outcome_labels=self.outcome_labels,
            outcome_index=self.outcome_index[mask],
            ci_method=self.ci_method,
            draws=draws,
        )

    def summary(self) -> pd.DataFrame:
        """A summary table of the results."""
        df = pd.DataFrame({
            self.stat_name: self.estimate,
            "std err": self.se,
            "z" if self.df is None else "t": self.tvalues,
            "P>|z|" if self.df is None else "P>|t|": self.pvalues,
            f"[{int(self.level*100)}% Conf.": self.ci_lower,
            "Interval]": self.ci_upper,
        }, index=self.labels)
        return df

    def __repr__(self) -> str:
        return self.summary().to_string()


class DiDResult:
    """Container for difference-in-differences results.

    Parameters
    ----------
    cells : MarginsResult
        The 4 cell predictions.
    simple_effects : MarginsResult
        The 2 simple effects (group differences per condition).
    did : MarginsResult
        The difference-in-differences (difference of simple effects).
    joint : MarginsResult
        All of the above in one joint result.
    """

    def __init__(
        self,
        cells: MarginsResult,
        simple_effects: MarginsResult,
        did: MarginsResult,
        joint: MarginsResult,
    ):
        self.cells = cells
        self.simple_effects = simple_effects
        self.did = did
        self.joint = joint

    def outcome(self, k: Union[int, str, Sequence[Union[int, str]]]) -> "DiDResult":
        """Slice rows belonging to the specified outcome class(es)."""
        return DiDResult(
            cells=self.cells.outcome(k),
            simple_effects=self.simple_effects.outcome(k),
            did=self.did.outcome(k),
            joint=self.joint.outcome(k),
        )

    def summary(self) -> pd.DataFrame:
        """Combined summary of cells, simple effects, and DiD."""
        c = self.cells.summary()
        s = self.simple_effects.summary()
        d = self.did.summary()
        return pd.concat([c, s, d], keys=["Cells", "Simple Effects", "DiD"])

    def __repr__(self) -> str:
        def banner(title: str) -> str:
            return f"\n{title}\n" + "-" * len(title) + "\n"

        out = banner("Adjusted Predictions")
        out += self.cells.summary().to_string()
        out += banner("Simple Effects")
        out += self.simple_effects.summary().to_string()
        out += banner("Difference-in-Differences")
        out += self.did.summary().to_string()
        return out
