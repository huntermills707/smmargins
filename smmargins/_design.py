from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import patsy

from .data import _Profile, Expr


@dataclass(frozen=True)
class ValueSpec:
    """Normalized per-variable specification."""
    kind: str  # "scalar" | "sequence" | "reducer" | "callable" | "expr"
    payload: Any  # the literal value, list, reducer name, callable, or Expr


_VALID_REDUCERS = {"mean", "median", "mode", "zero", "min", "max", "asobserved"}
_DEFAULT_VALUES_OPTIONS = {"asobserved", "mean", "median", "mode", "zero"}


def _is_percentile(reducer: str) -> bool:
    return bool(re.fullmatch(r"p\d+", reducer))


def _parse_percentile(reducer: str) -> int:
    n = int(reducer[1:])
    if not (0 <= n <= 100):
        raise ValueError(f"Percentile must be between 0 and 100, got {reducer!r}")
    return n


def _apply_reducer(col: pd.Series, reducer: str, *, frame: Optional[pd.DataFrame] = None) -> Any:
    """Compute a scalar from a column for the named reducer."""
    if reducer == "asobserved":
        return None  # sentinel — do nothing

    is_numeric = pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)

    if reducer == "mean":
        if not is_numeric:
            raise TypeError(f"reducer 'mean' requires a numeric column; got {col.dtype}")
        return float(col.mean())

    if reducer == "median":
        if not is_numeric:
            raise TypeError(f"reducer 'median' requires a numeric column; got {col.dtype}")
        return float(col.median())

    if reducer == "min":
        if not is_numeric:
            raise TypeError(f"reducer 'min' requires a numeric column; got {col.dtype}")
        return float(col.min())

    if reducer == "max":
        if not is_numeric:
            raise TypeError(f"reducer 'max' requires a numeric column; got {col.dtype}")
        return float(col.max())

    if reducer == "mode":
        try:
            return col.mode().iloc[0]
        except Exception:
            return col.iloc[0]

    if reducer == "zero":
        if is_numeric:
            return 0.0
        # categorical — caller must resolve reference level
        return None

    if _is_percentile(reducer):
        if not is_numeric:
            raise TypeError(f"reducer {reducer!r} requires a numeric column; got {col.dtype}")
        n = _parse_percentile(reducer)
        return float(np.percentile(col.dropna().to_numpy(dtype=float), n))

    raise ValueError(
        f"Unknown reducer {reducer!r}. Valid reducers: "
        f"{sorted(_VALID_REDUCERS | {'p0..p100'})}"
    )


def _classify_value(value: Any) -> ValueSpec:
    """Classify a single value into a ValueSpec."""
    if isinstance(value, Expr):
        return ValueSpec(kind="expr", payload=value)
    if callable(value) and not isinstance(value, (str, bytes)):
        return ValueSpec(kind="callable", payload=value)
    if isinstance(value, (str, bytes)):
        if value in _VALID_REDUCERS or _is_percentile(value):
            return ValueSpec(kind="reducer", payload=value)
        # A string that is not a reducer is treated as a scalar (e.g. categorical level)
        return ValueSpec(kind="scalar", payload=value)
    if np.isscalar(value):
        return ValueSpec(kind="scalar", payload=value)
    # Sequence-like (list, tuple, ndarray)
    if isinstance(value, (list, tuple, np.ndarray)):
        return ValueSpec(kind="sequence", payload=list(value))
    raise TypeError(f"Unsupported value type in values=: {type(value).__name__}")


def _normalize_values(
    values: Optional[Mapping[str, Any]],
    atexog: Optional[Mapping[str, Any]],
    base_columns: Sequence[str],
) -> Dict[str, ValueSpec]:
    """Merge ``values`` and ``atexog``, validate keys, classify each entry."""
    specs: Dict[str, ValueSpec] = {}

    # Normalize atexog first
    if atexog:
        for k, v in atexog.items():
            if np.isscalar(v) or isinstance(v, (str, bytes)):
                specs[k] = ValueSpec(kind="scalar", payload=v)
            else:
                specs[k] = ValueSpec(kind="sequence", payload=list(v))

    # Merge values on top
    if values:
        for k, v in values.items():
            if k in specs:
                raise ValueError(
                    f"Variable {k!r} specified in both atexog and values. "
                    f"Pass it in only one of them."
                )
            specs[k] = _classify_value(v)

    # Validate keys exist in the data
    invalid = [k for k in specs if k not in base_columns]
    if invalid:
        raise ValueError(
            f"Unknown variable(s) in values/atexog: {invalid}. "
            f"Available: {list(base_columns)}"
        )

    return specs


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

    def expand_values(
        self,
        base: pd.DataFrame,
        specs: Dict[str, ValueSpec],
        default_values: str = "asobserved",
    ) -> tuple[List[pd.DataFrame], List[str]]:
        """Expand a normalized values spec into frames and labels.

        1. Apply default_values policy to columns not in specs.
        2. Apply scalar/reducer/callable/expr specs (one value per row, no axis multiplication).
        3. Cartesian product over sequence specs.
        """
        if default_values not in _DEFAULT_VALUES_OPTIONS:
            raise ValueError(
                f"default_values must be one of {sorted(_DEFAULT_VALUES_OPTIONS)}, "
                f"got {default_values!r}"
            )

        frame = base.copy()
        base_cols = list(frame.columns)

        # 1. Apply default_values policy to unspecified columns
        for col in base_cols:
            if col in specs:
                continue
            if default_values == "asobserved":
                continue
            col_dtype = frame[col].dtype
            is_numeric = pd.api.types.is_numeric_dtype(col_dtype) and not pd.api.types.is_bool_dtype(col_dtype)
            if not is_numeric and default_values in ("mean", "median", "min", "max"):
                continue  # skip non-numeric columns for numeric-only reducers
            if not is_numeric and _is_percentile(default_values):
                continue  # skip non-numeric columns for percentiles
            spec = ValueSpec(kind="reducer", payload=default_values)
            applied = self._apply_spec_to_frame(frame, col, spec)
            if applied is not None:
                frame = applied

        # 2. Apply non-sequence specs (scalar, reducer, callable, expr)
        # Also build a base label from these specs
        base_label_parts = []
        for col, spec in specs.items():
            if spec.kind == "sequence":
                continue
            applied = self._apply_spec_to_frame(frame, col, spec)
            if applied is not None:
                frame = applied
            # Add to base label
            if spec.kind == "scalar":
                base_label_parts.append(f"{col}={spec.payload}")
            elif spec.kind == "reducer":
                base_label_parts.append(f"{col}={spec.payload}")

        base_label = ", ".join(base_label_parts)

        # 3. Cartesian product over sequence specs
        seq_items = [(k, v.payload) for k, v in specs.items() if v.kind == "sequence"]
        if not seq_items:
            return [frame], [base_label] if base_label else [""]

        frames, labels = [], []
        keys = [k for k, _ in seq_items]
        vals = [v for _, v in seq_items]
        for combo in itertools.product(*vals):
            f = frame.copy()
            for k, val in zip(keys, combo):
                f[k] = val
            frames.append(f)
            seq_label = ", ".join(f"{k}={val}" for k, val in zip(keys, combo))
            if base_label:
                labels.append(f"{base_label}, {seq_label}")
            else:
                labels.append(seq_label)
        return frames, labels

    def _apply_spec_to_frame(
        self, frame: pd.DataFrame, col: str, spec: ValueSpec
    ) -> Optional[pd.DataFrame]:
        """Apply a single ValueSpec to a column, returning the modified frame or None if unchanged."""
        if spec.kind == "scalar":
            payload = spec.payload
            # For numeric columns, reject strings that look like invalid reducers
            if isinstance(payload, str):
                col_series = frame[col]
                is_numeric = (
                    pd.api.types.is_numeric_dtype(col_series)
                    and not pd.api.types.is_bool_dtype(col_series)
                )
                if is_numeric and payload not in _VALID_REDUCERS and not _is_percentile(payload):
                    # Try to convert to numeric; if it fails, it might be a mistyped reducer
                    try:
                        float(payload)
                    except ValueError:
                        raise ValueError(
                            f"Invalid value {payload!r} for numeric column {col!r}. "
                            f"If you meant a reducer, valid choices are: "
                            f"{sorted(_VALID_REDUCERS | {'p0..p100'})}"
                        )
                if not is_numeric:
                    # For non-numeric columns, warn if the string doesn't match any level
                    levels = set(col_series.dropna().unique())
                    if payload not in levels:
                        import warnings
                        warnings.warn(
                            f"Value {payload!r} for column {col!r} does not match any "
                            f"observed level ({sorted(levels)}). It will be assigned as-is.",
                            UserWarning,
                            stacklevel=4,
                        )
            frame[col] = payload
            return frame

        if spec.kind == "reducer":
            val = _apply_reducer(frame[col], spec.payload, frame=frame)
            if val is None and spec.payload == "zero":
                # categorical reference level
                val = self.factor_reference_level(col)
            if val is not None:
                frame[col] = val
            return frame

        if spec.kind == "callable":
            fn = spec.payload
            result = fn(frame)
            if not isinstance(result, (pd.Series, np.ndarray)):
                raise TypeError(
                    f"Callable for {col!r} must return a pd.Series or np.ndarray, "
                    f"got {type(result).__name__}"
                )
            arr = np.asarray(result)
            if arr.shape[0] != len(frame):
                raise ValueError(
                    f"Callable for {col!r} returned length {arr.shape[0]}, "
                    f"expected {len(frame)}"
                )
            # Preserve dtype if possible; otherwise let pandas infer
            frame[col] = arr
            return frame

        if spec.kind == "expr":
            expr = spec.payload
            result = expr.evaluate(frame)
            arr = np.asarray(result)
            if arr.shape[0] != len(frame):
                raise ValueError(
                    f"Expr for {col!r} returned length {arr.shape[0]}, "
                    f"expected {len(frame)}"
                )
            frame[col] = arr
            return frame

        return None

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
        values: Optional[Mapping[str, Any]] = None,
        default_values: str = "asobserved",
    ) -> tuple[_Profile, List[pd.DataFrame], List[str], List[Optional[np.ndarray]]]:
        """Resolve profile, expand ``atexog``/``values``, and optionally split by ``over=``."""
        specs = _normalize_values(values, atexog, self.frame.columns)

        if over is None:
            profile = self.resolve_base(at, factor_stat)
            frames, labels = self.expand_values(profile.frame, specs, default_values=default_values)
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
                f_list, at_labels = self.expand_values(sp.frame, specs, default_values=default_values)
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
            f_list, at_labels = self.expand_values(profile.frame, specs, default_values=default_values)
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

    def validate_newdata(self, newdata: pd.DataFrame) -> None:
        """Validate that newdata contains the required columns."""
        if newdata.shape[0] == 0:
            raise ValueError("newdata must have at least one row")
        if self.raw_mode:
            n_exog = self.results.model.exog.shape[1]
            required = list(self.results.model.exog_names)[:n_exog]
            missing = [c for c in required if c not in newdata.columns]
            if missing:
                raise ValueError(
                    f"newdata is missing required columns: {missing}"
                )
            return
        # Formula mode: try building the design matrix; if it fails, surface the error
        try:
            patsy.dmatrix(self.design_info, newdata, return_type="matrix")
        except Exception as exc:
            # Try to extract missing column names from common error patterns
            msg = str(exc)
            import re
            # patsy NameError: "name 'col' is not defined"
            m = re.search(r"name '([^']+)' is not defined", msg)
            if m:
                missing = [m.group(1)]
            else:
                # KeyError from missing column
                m = re.search(r"'([^']+)'", msg)
                missing = [m.group(1)] if m else ["unknown"]
            raise ValueError(
                f"newdata is missing required columns: {missing}"
            ) from exc
