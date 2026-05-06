# How to set covariate profiles with `values=`, `Expr`, and `newdata=`

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing basic predictions
- {doc}`Tutorial: Adjusted predictions </tutorials/adjusted_predictions>` — `at=` and `atexog=`

## Problem statement

You want to evaluate predictions or marginal effects under hypothetical covariate profiles: setting income to its 25th percentile, increasing everyone's age by five years, or using an entirely new data frame. The `values=` keyword provides a per-variable DSL for this; `Expr` lets you write formula-based transformations; and `newdata=` is the escape hatch for arbitrary frames.

## Minimal working solution

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from smmargins import Margins, Expr

rng = np.random.default_rng(7)
N = 5_000
df = pd.DataFrame({
    "age": rng.normal(45, 12, N).clip(18, 90),
    "income": rng.lognormal(10.5, 0.4, N),
    "educ": rng.choice(["hs", "college", "grad"], N, p=[0.4, 0.4, 0.2]),
    "female": rng.integers(0, 2, N),
})
df["voted"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-(
    -4 + 0.05 * df.age + 0.00001 * df.income
    + 0.8 * (df.educ == "college") + 1.4 * (df.educ == "grad")
    + 0.3 * df.female
)))).astype(int)

fit = smf.logit("voted ~ age + income + C(educ) + female", data=df).fit(disp=False)
M = Margins(fit)

# Hold income at its median; everything else stays as observed
print(M.predict(values={"income": "p50"}))

# Increase everyone's age by 5 years
print(M.predict(values={"age": Expr("age + 5")}))

# Evaluate at an out-of-sample profile
new = pd.DataFrame({"age": [30, 50, 70], "income": [30000, 50000, 90000],
                    "educ": ["college", "hs", "grad"], "female": [0, 1, 0]})
print(M.predict(newdata=new))
```

## The `values=` DSL

`values=` is a mapping from variable names to **specifications**. Each specification can be one of several kinds:

| Kind | Example | Result |
|------|---------|--------|
| Scalar | `{"age": 45}` | Column set to 45 for every row |
| Categorical level | `{"educ": "college"}` | Column set to `"college"` |
| Reducer | `{"income": "mean"}` | Column reduced to its mean |
| Percentile | `{"income": "p25"}` | Column reduced to its 25th percentile |
| Sequence | `{"age": [25, 45, 65]}` | Cartesian product over the list |
| Callable | `{"income": lambda d: d["income"] * 1.1}` | Per-row transform via function |
| Expr | `{"income": Expr("income * 1.1")}` | Per-row transform via `df.eval` |

### Built-in reducers

Valid reducer strings are:

- `"mean"`, `"median"`, `"mode"`, `"min"`, `"max"`, `"zero"`
- Percentiles: `"p0"` through `"p100"`
- `"asobserved"` (the default for `default_values=`)

Numeric reducers raise on non-numeric columns; `"mode"` and `"zero"` work for any dtype. `"zero"` on a categorical column sets it to the reference (first observed) level.

### Sequences and the cartesian product

If any value is a list, tuple, or ndarray, `smmargins` computes the cartesian product over all sequence specs:

```python
# 3 ages × 2 sexes = 6 predictions
M.predict(values={"age": [25, 45, 65], "female": [0, 1]})
```

Non-sequence specs (scalars, reducers, callables, `Expr`) are applied first and do not multiply the grid.

### Callables

A callable receives the original data frame and must return a `pd.Series` or `np.ndarray` of the same length:

```python
M.predict(values={"income": lambda d: d["income"] * 1.10})
```

### `Expr`

:class:`~smmargins.Expr` wraps a string that is evaluated via :meth:`pandas.DataFrame.eval` against the original data frame. Use it when the transformation is easier to write as a formula:

```python
M.predict(values={"income": Expr("income * 1.10")})
M.predict(values={"age": Expr("(age - age.mean()) / age.std()")})
```

Strings that are *not* wrapped in `Expr` are interpreted as reducer names or categorical levels, not as formulas.

## `default_values=` — what happens to the rest

By default, columns not mentioned in `values=` or `atexog=` keep their **observed** values (`default_values="asobserved"`). You can change this:

```python
# Hold age at 45; set all other numerics to their mean
M.predict(values={"age": 45}, default_values="mean")

# Set all numerics to their median; factors stay as observed
M.predict(values={"age": 45}, default_values="median")
```

Valid options for `default_values` are `"asobserved"`, `"mean"`, `"median"`, `"mode"`, and `"zero"`.

## `newdata=` — arbitrary frames

`newdata=` accepts any :class:`pandas.DataFrame` with the same columns as the fitting data. It is **mutually exclusive** with `at`, `atexog`, `values`, and `over`:

```python
hypo = pd.DataFrame({"age": [25, 45, 65], "income": [30000, 50000, 80000],
                     "educ": ["college", "hs", "grad"], "female": [0, 1, 0]})
M.predict(newdata=hypo)
```

Use `newdata=` for out-of-sample profiles or when the DSL does not give you enough control.

## Using the DSL with `dydx` and `contrast`

`values=` and `default_values=` work on every post-estimation method, not just `predict`:

```python
# AME of age when income is held at its 25th percentile
M.dydx("age", values={"income": "p25"})

# Marginal effect of education when everyone is female
M.dydx("educ", values={"female": 1})

# Contrast: female=1 vs female=0, holding age at 45
M.contrast(a={"female": 1, "age": 45}, b={"female": 0, "age": 45})
```

## When to use this

Use `values=` for the common case where you want to tweak a few variables and leave the rest at observed or mean values. Use `Expr` when the counterfactual is naturally expressed as a formula (e.g. "income + 10%"). Use `newdata=` when you have a full data frame of profiles, especially out-of-sample ones.

## When NOT to use this

> ⚠️ **Trade-off:** Do not mix `newdata=` with `at`/`atexog`/`values`/`over` — they are mutually exclusive. Pass one complete frame OR use the DSL, not both.
>
> ⚠️ **Trade-off:** `default_values="mean"` reduces unspecified numeric columns to their mean, but factors stay as observed (or at their mode/reference depending on `factor_stat=`). If you want a single representative individual, use `at="mean"` instead.

## See also

- {doc}`How to compute counterfactual predictions with values, Expr, and newdata </howto/counterfactual_predictions>` — additional examples including joint contrasts
- {doc}`How to compute elasticities </howto/elasticities>` — `values=` inside elasticity calls
- {doc}`Reference: Expr </api>` — API reference for the `Expr` wrapper
