# How to compute counterfactual predictions with values, Expr, and newdata

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing basic predictions
- {doc}`Explanation: Counterfactual profiles </explanations/formula_vs_raw_mode>` — the difference between `at=`, `atexog=`, `values=`, and `newdata=`

## Problem statement

You want to evaluate predictions under hypothetical scenarios: setting income to its 25th percentile, increasing everyone's income by 10%, or evaluating at entirely out-of-sample profiles. You need to hold specific variables fixed while letting others vary or default to their means.

## Minimal working solution

Use `values=` for per-variable DSL, `Expr` for formula-based transformations, and `newdata=` for arbitrary frames.

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
    "region": rng.choice(["north", "south", "east", "west"], N),
})
df["voted"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-(
    -4 + 0.05 * df.age + 0.00001 * df.income
    + 0.8 * (df.educ == "college") + 1.4 * (df.educ == "grad")
    + 0.3 * df.female - 0.0004 * df.age * df.female
)))).astype(int)

fit = smf.logit("voted ~ age + income + C(educ) + female + age:female", data=df).fit(disp=False)
M = Margins(fit)

# Predictions at income quartiles, everything else at mean
print("At income quartiles:")
print(M.predict(values={"income": ["p25", "p50", "p75"]}, default_values="mean"))

# 10% income increase via Expr
print("\nCounterfactual: income + 10%:")
print(M.predict(values={"income": Expr("income * 1.10")}))

# Out-of-sample profiles
hypo = pd.DataFrame({"age": [25, 45, 65], "income": [30000, 50000, 80000]})
print("\nOut-of-sample predictions:")
print(M.predict(newdata=hypo))

# Joint contrast: treatment vs control
print("\nJoint contrast (female=1 vs female=0):")
joint = M.contrast(a={"female": 1}, b={"female": 0})
print(joint.summary())
```

## Variations

### Scalar values with default_values

```python
# Hold age at 45, reduce everything else to its mean
print(M.predict(values={"age": 45}, default_values="mean"))
```

### Lambda/callable in values

```python
# Same 10% increase with a callable
print(M.predict(values={"income": lambda d: d["income"] * 1.10}))
```

### Contrast with explicit newdata arms

```python
# Compare two specific profiles with joint SE
arm_a = pd.DataFrame({"age": [45], "income": [60000], "female": [1], "educ": ["college"]})
arm_b = pd.DataFrame({"age": [45], "income": [60000], "female": [0], "educ": ["college"]})
print(M.contrast(a_newdata=arm_a, b_newdata=arm_b).summary())
```

### dydx with values

```python
# AME of age when income is held at its 25th percentile
print(M.dydx("age", values={"income": "p25"}))
```

> ⚠️ **Trade-off:** `values=` is the most ergonomic DSL for counterfactuals — it mixes per-variable specifications with automatic defaults for the rest. `newdata=` is the escape hatch for arbitrary frames but is mutually exclusive with `at`/`atexog`/`values`/`over`. `Expr` evaluates against the original data frame, so it preserves the full distribution when perturbing.

## When to use this

Use `values=` when you want to manipulate specific variables while leaving the rest at observed values or means. Use `Expr(...)` when the counterfactual is a function of the existing data (e.g., "everyone's income + 10%"). Use `newdata=` when you have out-of-sample profiles or need full control over every covariate. Use `contrast()` when you need the difference between two arms with the correct joint SE.

## When NOT to use this

> ⚠️ **Trade-off:** Do not combine `newdata=` with `at`/`atexog`/`values`/`over` — they are mutually exclusive. Pass one complete frame OR use the DSL, not both. `default_values="mean"` reduces unspecified numeric columns to their mean but may produce "fractional people" for factors — use `factor_stat="mode"` on the `Margins` constructor if you want modal levels instead.

## See also

- {doc}`How to set covariate profiles with values=, Expr, and newdata= </howto/expr_and_values>` — full specification of the DSL
- {doc}`Reference: Expr </api>` — API reference for the Expr wrapper
- {doc}`Reference: Margins.contrast </api>` — joint contrast API
- {doc}`How to compute subgroup-specific AMEs </howto/subgroup_analysis>` — `over=` for within-group averaging
