from __future__ import annotations

import re
from dataclasses import dataclass

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
        ci_method: Optional[str] = None,
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
            ci_method=ci_method if ci_method is not None else self.ci_method,
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

    def wald(
        self,
        C: Optional[Union[Sequence[float], np.ndarray]] = None,
        value: Union[float, Sequence[float]] = 0.0,
    ) -> "WaldResult":
        r"""Joint Wald test of linear restrictions on the margins.

        Computes :math:`\chi^2 = (C\hat\theta - c_0)' (C V C')^{-1}
        (C\hat\theta - c_0)` with degrees of freedom equal to the rank
        of the contrast covariance.

        Parameters
        ----------
        C : array-like, optional
            Contrast matrix of shape ``(m, k)`` where ``k`` is the number
            of margins. If ``None``, tests that *all* margins equal
            ``value`` (i.e. ``C = eye(k)``).
        value : float or array-like, default 0
            Hypothesized value under the null.

        Returns
        -------
        WaldResult

        Raises
        ------
        ValueError
            If the contrast covariance is singular.
        """
        if C is None:
            C = np.eye(len(self.estimate))
        C_arr = np.asarray(C, dtype=float)
        if C_arr.ndim == 1:
            C_arr = C_arr.reshape(1, -1)
        if C_arr.shape[1] != self.estimate.size:
            raise ValueError(
                f"contrast has {C_arr.shape[1]} columns, but there are "
                f"{self.estimate.size} estimates"
            )

        val = np.atleast_1d(np.asarray(value, dtype=float))
        if val.size == 1:
            val = np.full(C_arr.shape[0], val.item())
        elif val.size != C_arr.shape[0]:
            raise ValueError(
                "value length must match number of contrasts "
                f"({C_arr.shape[0]})"
            )

        theta = C_arr @ self.estimate - val
        V_theta = C_arr @ self.vcov @ C_arr.T

        # Test for singularity using a small tolerance
        try:
            V_inv = np.linalg.inv(V_theta)
        except np.linalg.LinAlgError:
            raise ValueError(
                "The contrast covariance matrix is singular. "
                "Use a smaller subset of contrasts or check for "
                "redundant rows in C."
            )

        stat = float(theta @ V_inv @ theta)
        df = int(np.linalg.matrix_rank(V_theta))
        pvalue = float(1 - stats.chi2.cdf(stat, df))

        return WaldResult(
            stat=stat,
            df=df,
            pvalue=pvalue,
            contrast_matrix=C_arr,
            contrast_estimates=theta,
        )

    def pairwise(self, by: str, ci_method: Optional[str] = None) -> "MarginsResult":
        """All pairwise comparisons of a factor variable's levels.

        Parameters
        ----------
        by : str
            The factor variable name. The current result must contain
            discrete contrasts for this variable (e.g. from ``dydx(by)``).

        Returns
        -------
        MarginsResult
            One row per pairwise comparison.

        Raises
        ------
        ValueError
            If fewer than 2 rows are found for ``by``.
        """
        prefix = f"{by}:"
        indices: List[int] = []
        levels: List[str] = []
        references: List[str] = []
        for i, lab in enumerate(self.labels):
            # Strip any outcome suffix for matching
            base = lab.split(" (")[0]
            if base.startswith(prefix):
                indices.append(i)
                # Parse "by: lvl vs ref"
                rest = base[len(prefix):].strip()
                match = re.match(r"(.+?)\s+vs\s+(.+)", rest)
                if match:
                    levels.append(match.group(1).strip())
                    references.append(match.group(2).strip())
                else:
                    levels.append(rest)
                    references.append("")

        if len(indices) < 2:
            raise ValueError(
                f"Need at least 2 contrasts for {by!r} to build pairwise "
                f"comparisons. Found {len(indices)} matching rows."
            )

        n = len(indices)
        C_rows: List[np.ndarray] = []
        new_labels: List[str] = []
        for i in range(n):
            for j in range(i + 1, n):
                c = np.zeros(len(self.estimate))
                c[indices[i]] = 1.0
                c[indices[j]] = -1.0
                C_rows.append(c)
                new_labels.append(f"{by}: {levels[i]} vs {levels[j]}")

        C = np.vstack(C_rows)
        return self.contrast(C, labels=new_labels, name="pairwise", ci_method=ci_method)

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


@dataclass
class WaldResult:
    """Result of a joint Wald test.

    Attributes
    ----------
    stat : float
        Wald chi-squared statistic.
    df : int
        Degrees of freedom (rank of the contrast covariance).
    pvalue : float
        Two-sided p-value from the chi-squared distribution.
    contrast_matrix : ndarray
        The contrast matrix C used in the test.
    contrast_estimates : ndarray
        The contrasted estimates C @ theta_hat - value.
    """

    stat: float
    df: int
    pvalue: float
    contrast_matrix: np.ndarray
    contrast_estimates: np.ndarray

    def __repr__(self) -> str:
        return (
            f"WaldResult(stat={self.stat:.4f}, df={self.df}, "
            f"pvalue={self.pvalue:.4g})"
        )


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
