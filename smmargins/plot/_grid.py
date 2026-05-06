"""Build counterfactual grids for plotting."""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ..core import Margins


@dataclass
class PlotData:
    x_name: str
    x_values: np.ndarray
    by_name: Optional[str]
    by_levels: Optional[List[Any]]
    estimates: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    statistic: str
    label: str


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def build_grid(
    margins: Margins,
    condition: Union[str, Mapping[str, Any]],
    by: Optional[str] = None,
) -> tuple[str, np.ndarray, Optional[str], Optional[List[Any]]]:
    """Build the x-axis grid for plotting.

    Returns (x_name, x_values, by_name, by_levels).
    """
    if isinstance(condition, str):
        x_name = condition
        col = margins.data[x_name]
        if _is_numeric(col):
            x_values = np.linspace(col.min(), col.max(), 50)
        else:
            x_values = np.array(sorted(col.dropna().unique()))
    else:
        # condition is a dict-like mapping
        items = list(condition.items())
        if len(items) != 1:
            raise ValueError(
                "condition dict must have exactly one key for plotting"
            )
        x_name, x_values = items[0]
        x_values = np.asarray(x_values)

    by_levels = None
    if by is not None:
        if by not in margins.data.columns:
            raise ValueError(f"by column {by!r} not found in data")
        by_levels = sorted(margins.data[by].dropna().unique().tolist())

    return x_name, x_values, by, by_levels
