# How to compute elasticities and semi-elasticities for marginal effects

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Explanation: Elasticities </explanations/scales_link_functions>` — the four elasticity methods and their interpretations

## Problem statement

You want to express marginal effects as elasticities — percentage changes rather than level changes. You need the full elasticity (ey/ex), semi-elasticity (dy/ex or ey/dx), or want to combine elasticity methods with custom scales.

## Minimal working solution

Pass `method=` to `dydx` with one of `"eyex"`, `"dyex"`, or `"eydx"`. Continuous variables only — discrete variables raise.

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

# Full elasticity: (dy/dx) * (x/y)
print("Full elasticity ey/ex for age:")
print(M.dydx("age", method="eyex"))

# Semi-elasticity: (dy/dx) * x
print("\nSemi-elasticity dy/ex for age:")
print(M.dydx("age", method="dyex"))

# Semi-elasticity: (dy/dx) / y
print("\nSemi-elasticity ey/dx for age:")
print(M.dydx("age", method="eydx"))
```

## Variations

### Elasticity at means (MEM-style)

```python
print("MEM elasticity of age at covariate means:")
print(M.dydx("age", method="eyex", at="mean"))
```

### Elasticity on the linear scale

```python
# ey/ex on the linear predictor scale (beta * x / eta)
print("Linear-scale elasticity of income:")
print(M.dydx("income", method="eyex", scale="linear"))
```

### Elasticity with counterfactual values

```python
# Elasticity of age when income is at median
print("Elasticity of age at median income:")
print(M.dydx("age", method="eyex", values={"income": "p50"}))
```

See {doc}`How to set covariate profiles with values=, Expr, and newdata= </howto/expr_and_values>` for the full `values=` DSL.

### All elasticity methods on a Poisson model (canonical-link shortcut)

```python
import statsmodels.api as sm

# Poisson: eyex on response scale equals beta_k * xbar_k exactly
df["count"] = rng.poisson(np.exp(0.5 + 0.03 * df.age + 0.00001 * df.income))
fit_pois = smf.glm("count ~ age + income", data=df, family=sm.families.Poisson()).fit()
M_pois = Margins(fit_pois)
eyex_income = M_pois.dydx("income", method="eyex", scale="response")
print(f"eyex(income) = {eyex_income.estimate[0]:.6f}")
print(f"beta * xbar  = {fit_pois.params['income'] * df.income.mean():.6f}")
```

> ⚠️ **Trade-off:** Elasticity methods always use finite differences internally (even when `analytic=True` is set on the constructor). This makes them slightly slower than `method="dydx"` but the difference is negligible for typical datasets. Elasticities on variables where the prediction can be near zero (e.g., rare binary outcomes) can be numerically unstable — a `RuntimeWarning` is issued when `|y| < 1e-12`.

## When to use this

Use `method="eyex"` when you want to interpret the effect as "a 1% increase in X leads to a _% change in Y." Use `method="dyex"` for "a 1% increase in X leads to a _-unit change in Y." Use `method="eydx"` for "a 1-unit increase in X leads to a _% change in Y." Elasticities are especially useful when variables are measured on different scales.

## When NOT to use this

> ⚠️ **Trade-off:** Do not use elasticity methods on discrete variables — they raise `ValueError`. For binary or categorical variables, stick with `method="dydx"` (the default). Do not use `method="eyex"` when predictions can be zero or negative (e.g., OLS with negative fitted values) — the division by y is undefined.

## See also

- {doc}`Reference: Margins.dydx </api>` — full parameter list including `method=`
- {doc}`How to report marginal effects on custom scales </howto/custom_transforms>` — composing elasticity methods with `scale=`
- {doc}`How to compute subgroup-specific AMEs </howto/subgroup_analysis>` — elasticities within subgroups via `over=`
