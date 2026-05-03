from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import patsy

from .data import _Profile


@dataclass
class DesignResolver:
    """Owns the model's data frame, design info, and all data-shaping
    helpers (`at=`, `over=`, factor probing).

    Pure data concern — no knowledge of predictions, derivatives,
    scales, or inference.
    """
    results: Any
    frame: Optional[pd.DataFrame]
    design_info: Optional[patsy.DesignInfo]
    exog_names: List[str]

    @property
    def raw_mode(self) -> bool:
        return self.design_info is None

    @classmethod
    def from_results(cls, results, data: Optional[pd.DataFrame] = None) -> "DesignResolver":
        if data is not None:
            frame = data.copy()
        else:
            frame = cls._extract_frame(results)
            if frame is not None:
                frame = frame.copy()
        design_info = cls._extract_design_info(results)
        return cls(
            results=results,
            frame=frame,
            design_info=design_info,
            exog_names=list(results.model.exog_names),
        )

    @staticmethod
    def _extract_frame(results) -> Optional[pd.DataFrame]:
        """Attempt to retrieve the original fitting data frame from the model."""
        try:
            return results.model.data.frame
        except AttributeError:
            pass
        orig_exog = getattr(results.model.data, "orig_exog", None)
        if orig_exog is not None and hasattr(orig_exog, "columns"):
            return pd.DataFrame(orig_exog)
        return None

    @staticmethod
    def _extract_design_info(results):
        """Retrieve the patsy DesignInfo from the model, if available."""
        exog = getattr(results.model.data, "orig_exog", None)
        if exog is not None and hasattr(exog, "design_info"):
            return exog.design_info
        di = getattr(results.model.data, "design_info", None)
        if di is not None:
            return di
        return None

    def build_exog(self, frame: pd.DataFrame) -> np.ndarray:
        """Build a numeric design matrix from a data frame."""
        if self.raw_mode:
            names = list(self.results.model.exog_names)
            n_exog = self.results.model.exog.shape[1]
            cols = names[:n_exog]
            return np.asarray(frame[cols].to_numpy(dtype=float))
        dm = patsy.dmatrix(self.design_info, frame, return_type="matrix")
        return np.asarray(dm)

    @staticmethod
    def expand_at(
        base: pd.DataFrame, at: Optional[Mapping[str, object]]
    ) -> tuple[List[pd.DataFrame], List[str]]:
        """Cartesian-product expansion of an ``at`` specification."""
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

    def single_row(
        self,
        at: str,
        factor_stat: str,
        data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build a one-row representative frame in data space."""
        if data is None:
            data = self.frame
        row = {}
        for c in data.columns:
            col = data[c]
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
                        f"single_row called with at={at!r}; only "
                        f"'mean'/'median'/'zero' are valid here."
                    )
            elif factor_stat == "mode":
                try:
                    row[c] = col.mode().iloc[0]
                except Exception:
                    row[c] = col.iloc[0]
            elif factor_stat == "zero":
                row[c] = self.factor_reference_level(c)
            else:
                raise ValueError(
                    f"single_row called with factor_stat={factor_stat!r}; "
                    f"the 'mean' path uses design-matrix collapse instead."
                )
        return pd.DataFrame([row])

    def factor_reference_level(self, col: str) -> object:
        """Return the patsy reference level for a categorical column."""
        levels = self.frame[col].dropna().unique().tolist()
        try:
            levels = sorted(levels)
        except TypeError:
            pass
        if not levels:
            return self.frame[col].iloc[0]
        if self.raw_mode or self.design_info is None or len(levels) < 2:
            return levels[0]
        probe = pd.DataFrame([{
            c: self.probe_default(c) for c in self.frame.columns
        }])
        best_lvl, best_norm = levels[0], None
        for lvl in levels:
            p = probe.copy()
            p[col] = lvl
            try:
                X = self.build_exog(p)
            except Exception:
                continue
            norm = float(np.abs(np.asarray(X)).sum())
            if best_norm is None or norm < best_norm:
                best_norm = norm
                best_lvl = lvl
        return best_lvl

    def probe_default(self, col: str):
        """Return a default probe value for a column."""
        s = self.frame[col]
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

    def resolve_base(
        self,
        at: str,
        factor_stat: str,
        data: Optional[pd.DataFrame] = None,
    ) -> _Profile:
        """Return the evaluation profile for ``(at, factor_stat)``."""
        if data is None:
            data = self.frame

        if at == "overall":
            return _Profile(frame=data, collapse_design=False)

        if factor_stat == "mean":
            f = data.copy()
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
            return _Profile(frame=f, collapse_design=True)

        return _Profile(frame=self.single_row(at, factor_stat, data=data),
                        collapse_design=False)

    def subgroup_profiles(
        self,
        profile: _Profile,
        over: Union[str, List[str]],
    ) -> tuple[List[_Profile], List[str]]:
        """Split a multi-row profile into subgroup-specific profiles."""
        if isinstance(over, str):
            over = [over]
        cols = list(over)

        for col in cols:
            if col not in self.frame.columns:
                raise ValueError(
                    f"over column {col!r} not found in data. "
                    f"Available: {list(self.frame.columns)}"
                )

        frame = profile.frame
        if frame.shape[0] <= 1:
            raise ValueError(
                "subgroup_profiles requires a multi-row profile. "
                "Single-row profiles must be handled by the caller."
            )

        grouped = frame.groupby(cols[0] if len(cols) == 1 else cols, sort=True)
        profiles = []
        labels = []
        for name, group in grouped:
            if len(cols) == 1:
                label = f"{cols[0]}={name}"
            else:
                label = ", ".join(f"{c}={v}" for c, v in zip(cols, name))
            labels.append(label)
            profiles.append(
                _Profile(
                    frame=group,
                    collapse_design=profile.collapse_design,
                    collapse_numerics=profile.collapse_numerics,
                )
            )
        return profiles, labels

    def expand_over(
        self,
        at: str,
        factor_stat: str,
        atexog: Optional[Mapping[str, object]],
        over: Optional[Union[str, List[str]]],
        weights: Optional[np.ndarray] = None,
        get_weights: Optional[callable] = None,
    ) -> tuple[_Profile, List[pd.DataFrame], List[str], List[Optional[np.ndarray]]]:
        """Resolve profile, expand ``atexog``, and optionally split by ``over=``."""
        if over is None:
            profile = self.resolve_base(at, factor_stat)
            frames, labels = self.expand_at(profile.frame, atexog)
            weights_list = []
            for f in frames:
                n_rows = 1 if profile.collapse_design else len(f)
                weights_list.append(get_weights(n_rows) if get_weights is not None else None)
            return profile, frames, labels, weights_list

        if isinstance(over, str):
            over = [over]

        if at == "overall" or factor_stat == "mean":
            profile = self.resolve_base(at, factor_stat)
            sub_profiles, sub_labels = self.subgroup_profiles(profile, over)
            frames = []
            labels = []
            weights_list = []
            for sp, sl in zip(sub_profiles, sub_labels):
                f_list, at_labels = self.expand_at(sp.frame, atexog)
                if profile.collapse_design:
                    w = None
                else:
                    idx = sp.frame.index
                    w = weights[idx] if weights is not None else None
                for f, al in zip(f_list, at_labels):
                    frames.append(f)
                    weights_list.append(w)
                    labels.append(f"{sl}, {al}" if al else sl)
            return profile, frames, labels, weights_list

        # factor_stat in ("mode", "zero")  → single-row profiles
        cols = list(over)
        grouped = self.frame.groupby(cols[0] if len(cols) == 1 else cols, sort=True)
        frames = []
        labels = []
        weights_list = []
        for name, group in grouped:
            if len(cols) == 1:
                label = f"{cols[0]}={name}"
            else:
                label = ", ".join(f"{c}={v}" for c, v in zip(cols, name))
            profile = self.resolve_base(at, factor_stat, data=group)
            f_list, at_labels = self.expand_at(profile.frame, atexog)
            for f, al in zip(f_list, at_labels):
                frames.append(f)
                weights_list.append(None)
                labels.append(f"{label}, {al}" if al else label)
        return profile, frames, labels, weights_list

    def collapse_numerics_to_mean(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Replace observed (row-varying) numeric columns with their training mean."""
        out = frame.copy()
        for c in self.frame.columns:
            if c not in out.columns:
                continue
            col = self.frame[c]
            if not (pd.api.types.is_numeric_dtype(col)
                    and not pd.api.types.is_bool_dtype(col)):
                continue
            if out[c].nunique(dropna=False) <= 1:
                continue
            out[c] = col.mean()
        return out
