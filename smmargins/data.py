from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Margins

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
    """

    frame: pd.DataFrame
    collapse_design: bool = False
    collapse_numerics: bool = False

    def prepare_frame(self, frame: pd.DataFrame, margins: Margins) -> pd.DataFrame:
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

    def materialize(self, frame: pd.DataFrame, margins: Margins) -> np.ndarray:
        """Run the full pipeline (numerics collapse → build → design collapse).

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
        """
        frame = self.prepare_frame(frame, margins)
        X = margins._build_exog(frame)
        if self.collapse_design:
            X = X.mean(axis=0, keepdims=True)
        return X
