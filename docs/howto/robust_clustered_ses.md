# How to compute robust and cluster-robust standard errors for marginal effects

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Mathematical motivation </math>` — how `cov_type` propagates through the Jacobian

## Problem statement

You need heteroskedasticity-robust or cluster-robust standard errors for your marginal effects. The model was fit with conventional (`nonrobust`) covariance, but your data has grouped structure or you suspect heteroskedasticity. You want to recompute SEs without refitting the model.

## Minimal working solution

Pass `cov_type` to the `Margins` constructor or to any individual `predict` / `dydx` call.

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

# Apply HC3 heteroskedasticity-robust covariance
M_hc3 = Margins(fit, cov_type="HC3")
print(M_hc3.dydx("age"))

# Cluster-robust: synthesize household clusters (~10 obs per cluster)
df["household"] = rng.integers(0, N // 10, N)
fit_c = smf.logit("voted ~ age + income + C(educ) + female + age:female", data=df).fit(disp=False)
M_cluster = Margins(fit_c, cov_type="cluster", cov_kwds={"groups": df["household"]})
print(M_cluster.dydx("age"))

# Per-call override (no need to recreate Margins)
M = Margins(fit)
print(M.dydx("age", cov_type="HC3"))
print(M.dydx("age", cov_type="HC1"))
```

## Variations

### HAC (Newey-West) covariance

```python
M_hac = Margins(fit, cov_type="HAC", cov_kwds={"maxlags": 4})
print(M_hac.dydx("age"))
```

### HC0 through HC3

```python
for ctype in ["HC0", "HC1", "HC2", "HC3"]:
    se = Margins(fit, cov_type=ctype).dydx("age").se[0]
    print(f"  {ctype}: SE = {se:.6f}")
```

### User-supplied covariance matrix

```python
V_custom = fit.cov_params().to_numpy() * 1.5  # 50% wider
M_custom = Margins(fit, vcov=V_custom)
print(M_custom.dydx("age"))
```

> ⚠️ **Trade-off:** `cov_type` recomputes the parameter covariance via statsmodels' sandwich machinery. This is fast (no model refits) but only covers the *parameter* uncertainty. It does not account for uncertainty in the functional form of the marginal effect itself. For highly nonlinear margins or small samples, consider `vce="bootstrap"` or `vce="simulation"` instead.

## When to use this

Use `cov_type="HC3"` (or `"HC0"`–`"HC2"`) when you suspect heteroskedasticity in the latent error. Use `cov_type="cluster"` with `cov_kwds={"groups": ...}` when observations are grouped (panel data, households, villages) and errors are correlated within group. Use `cov_type="HAC"` for time-series data with autocorrelation.

## When NOT to use this

> ⚠️ **Trade-off:** Do not use cluster-robust covariance with too few clusters (rule of thumb: < 50 clusters). The asymptotic approximation breaks down and SEs can be severely downward-biased. Do not use `cov_type` when the model itself was fit with a different family or link — `cov_type` only affects the sandwich, not the point estimates.

## See also

- {doc}`Reference: Margins.cov_type </api>` — full list of supported covariance types
- {doc}`Mathematical motivation </math>` — how the Jacobian sandwiches covariance
- {doc}`How to compute bootstrap standard errors </howto/bootstrap>` — when sandwich approximations are insufficient
