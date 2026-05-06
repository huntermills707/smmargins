# How to compute subgroup-specific average marginal effects with the over parameter

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Explanation: AME vs MEM </explanations/discrete_vs_continuous>` — when averaging over the sample is the right thing to do

## Problem statement

You want marginal effects that are averaged within subgroups defined by one or more categorical variables (e.g., AME of age on voting probability separately for men and women, or by education and region). You need the full joint covariance across subgroups so that cross-subgroup contrasts and Wald tests remain valid.

## Minimal working solution

Pass `over=` to `dydx` or `predict` with a column name (string) or list of column names.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from smmargins import Margins

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

# Subgroup AME by education
print("AME of age by education level:")
print(M.dydx("age", over="educ"))

# Subgroup AME by education and gender (joint covariance preserved)
print("\nAME of age by education and female:")
print(M.dydx("age", over=["educ", "female"]))

# Subgroup predictions
print("\nPredicted probability by region:")
print(M.predict(over="region"))
```

## Variations

### Cross-subgroup contrast

```python
# Test whether AME(age) differs between men and women
res = M.dydx("age", over="female")
contrast = res.contrast(np.array([1, -1]))  # female=1 minus female=0
print(f"Contrast estimate: {contrast.estimate[0]:.6f}")
print(f"Contrast SE:       {contrast.se[0]:.6f}")
print(f"p-value:           {contrast.pvalue[0]:.4f}")
```

### Combining over with atexog

```python
# Subgroup predictions at specific ages
print(M.predict(over="educ", atexog={"age": [25, 45, 65]}))
```

### Subgroup AME on the linear scale

```python
# Linear-scale subgroup AMEs (equal to the coefficient for OLS, varies for logit)
print(M.dydx("age", over="educ", scale="linear"))
```

> ⚠️ **Trade-off:** `over=` partitions the sample and averages within each subgroup, preserving the full joint covariance matrix (not a block-diagonal approximation). This makes cross-subgroup contrasts valid but means the covariance computation involves all subgroups simultaneously. For very large datasets with many subgroups, memory usage scales with the number of subgroups.

## When to use this

Use `over=` when you need heterogeneity analysis — reporting different marginal effects for different subpopulations. It is the correct way to answer "how does the effect of X on Y differ across groups?" while preserving the joint covariance for valid inference on those differences.

## When NOT to use this

> ⚠️ **Trade-off:** Do not use `over=` with continuous variables — it is designed for categorical partitioners. Do not use `over=` when you simply want marginal effects at representative values of a covariate — use `atexog=` instead (e.g., `M.dydx("age", atexog={"female": [0, 1]})`). `over=` and `newdata=` are mutually exclusive.

## See also

- {doc}`Reference: Margins.dydx </api>` — full parameter list including `over=`
- {doc}`How to perform joint tests and pairwise comparisons </howto/joint_tests_pairwise>` — `wald()` and `contrast()` on subgroup results
- {doc}`How to compute counterfactual predictions </howto/counterfactual_predictions>` — `atexog=` for representative-value profiles
