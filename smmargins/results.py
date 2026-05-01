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
        The estimated covariance matrix of the margins (delta method).
    se : ndarray
        Standard errors (square root of the diagonal of ``vcov``).
    tvalues : ndarray
        Test statistics (estimate / se).
    pvalues : ndarray
        Two-sided p-values based on either the normal or t distribution.
    labels : list of str
        Labels for each row of the results.
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
    ):
        self.estimate = np.asarray(estimate)
        self.vcov = np.asarray(vcov)
        self.labels = list(labels) if labels is not None else [str(i) for i in range(len(self.estimate))]
        self.level = level
        self.df = df
        self.stat_name = stat_name
        self.outcome_labels = list(outcome_labels) if outcome_labels is not None else None
        self.outcome_index = np.asarray(outcome_index) if outcome_index is not None else None

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.vcov), 0))

    @property
    def tvalues(self) -> np.ndarray:
        return self.estimate / self.se

    @property
    def pvalues(self) -> np.ndarray:
        if self.df is not None:
            return 2 * stats.t.sf(np.abs(self.tvalues), self.df)
        return 2 * stats.norm.sf(np.abs(self.tvalues))

    @property
    def ci_lower(self) -> np.ndarray:
        alpha = 1 - self.level
        if self.df is not None:
            crit = stats.t.ppf(1 - alpha / 2, self.df)
        else:
            crit = stats.norm.ppf(1 - alpha / 2)
        return self.estimate - crit * self.se

    @property
    def ci_upper(self) -> np.ndarray:
        alpha = 1 - self.level
        if self.df is not None:
            crit = stats.t.ppf(1 - alpha / 2, self.df)
        else:
            crit = stats.norm.ppf(1 - alpha / 2)
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
        return MarginsResult(
            estimate=est, vcov=vcov, labels=labels,
            level=self.level, df=self.df, stat_name=name,
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

        return MarginsResult(
            estimate=self.estimate[mask],
            vcov=self.vcov[np.ix_(mask, mask)],
            labels=[self.labels[i] for i in np.where(mask)[0]],
            level=self.level,
            df=self.df,
            stat_name=self.stat_name,
            outcome_labels=self.outcome_labels,
            outcome_index=self.outcome_index[mask],
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
