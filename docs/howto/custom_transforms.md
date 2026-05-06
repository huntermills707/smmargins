# How to report marginal effects on custom scales and with user-defined transforms

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Explanation: Prediction scales </explanations/scales_link_functions>` — the relationship between response scale, linear predictor, and transformed scales

## Problem statement

You want marginal effects on a scale other than the default response scale. You might need the linear predictor (Stata's `xb`), the odds-ratio scale for logit, or a completely custom nonlinear transformation of the prediction. You need to know which built-in scales are available and how to define your own.

## Minimal working solution

Pass a string (`"linear"`, `"or"`, `"exp"`, etc.) or a `Transform` object to the `scale=` parameter of `predict` or `dydx`.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from smmargins import Margins, Transform

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

# Built-in scales
print("Response scale (default):")
print(M.dydx("age", scale="response"))

print("\nLinear predictor scale (Stata 'xb'):")
print(M.dydx("age", scale="linear"))

print("\nOdds-ratio scale (logit only):")
print(M.dydx("age", scale="or"))

# Custom Transform: square of the linear predictor
square = Transform(
    value=lambda e: e ** 2,
    grad=lambda e: 2 * e,
    hess=lambda e: np.full_like(e, 2.0),
    name="square",
)
print("\nCustom 'square' transform:")
print(M.dydx("age", scale=square))
```

## Variations

### All built-in scales on predictions

```python
for s in ["response", "linear", "pr", "or", "exp", "log"]:
    try:
        est = M.predict(scale=s).estimate[0]
        print(f"  {s:10s}: {est:.6f}")
    except ValueError as e:
        print(f"  {s:10s}: raises — {e}")
```

### Custom transform for predictions only (no hessian needed)

```python
# For predict(), only grad= is required
logistic = Transform(
    value=lambda e: 1 / (1 + np.exp(-e)),
    grad=lambda e: np.exp(-e) / (1 + np.exp(-e)) ** 2,
    hess=None,  # OK for predict
    name="logistic",
)
print(M.predict(scale=logistic))
```

### Elasticity on a custom scale

```python
# Compose elasticity methods with custom scales
print(M.dydx("age", method="eyex", scale=square))
print(M.dydx("age", method="eyex", scale="linear"))
```

> ⚠️ **Trade-off:** Built-in scales are validated against the model family (e.g., `scale="or"` raises on Poisson). Custom transforms bypass these checks — you are responsible for ensuring the transform is mathematically valid for your model. Custom transforms for `dydx` require an analytic second derivative (`hess=`); there is no autodiff.

## When to use this

Use `scale="linear"` when you want marginal effects on the latent-index scale (the coefficient scale). Use `scale="or"` for logit odds-ratio interpretation. Use a custom `Transform` when you need a scale that is not built-in, such as a policy-relevant transformation or a link function from a different family.

## When NOT to use this

> ⚠️ **Trade-off:** Do not use `scale="or"` on non-logit models — it raises. Do not use custom transforms without providing `hess=` for `dydx` calls — it raises with a clear error. For prediction-only calls (`predict()`), `hess` can be `None`. Do not define transforms with discontinuous derivatives unless you understand the delta-method implications.

## See also

- {doc}`Reference: Built-in scales </api>` — complete list of built-in scale strings and their definitions
- {doc}`Reference: Transform class </api>` — full API for custom transforms
- {doc}`How to compute elasticities </howto/elasticities>` — composing elasticity methods with scales
