# How to use Krinsky–Robb simulation for standard errors and confidence intervals

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Mathematical motivation </math>` — delta method and alternative VCEs

## Problem statement

You want an alternative to delta-method standard errors that does not rely on the first-order Taylor approximation. You need confidence intervals that are asymmetric when the margin function is nonlinear in parameters, or you want to verify that your delta-method SEs are not misleading.

## Minimal working solution

Pass `vce="simulation"` to any `predict` or `dydx` call. Set `n_sims` for the number of draws and `sim_seed` for reproducibility.

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

# Krinsky–Robb simulation VCE for a single AME
ame_sim = M.dydx("age", vce="simulation", n_sims=2000, sim_seed=42)
print(ame_sim)

# Simulation VCE for predictions at representative ages
pred_sim = M.predict(atexog={"age": [25, 45, 65]}, vce="simulation", n_sims=5000, sim_seed=123)
print(pred_sim)
```

## Variations

### Combine with robust covariance

```python
# HC3 parameter draws -> HC3 simulation SEs
ame_hc3_sim = M.dydx("age", vce="simulation", cov_type="HC3", n_sims=2000, sim_seed=42)
print(ame_hc3_sim)
```

### Large-scale simulation for asymmetric CIs

```python
# 10,000 draws for more precise interval tails
ame_precise = M.dydx("income", vce="simulation", n_sims=10000, sim_seed=7)
print(f"Estimate: {ame_precise.estimate[0]:.6f}")
print(f"SE:       {ame_precise.se[0]:.6f}")
print(f"95% CI:   [{ame_precise.ci_lower[0]:.6f}, {ame_precise.ci_upper[0]:.6f}]")
```

### Simulation for a family of margins (simultaneous inference)

```python
ages = [25, 35, 45, 55, 65]
pred_family = M.predict(
    atexog={"age": ages},
    vce="simulation", n_sims=4000, sim_seed=99,
    ci_method="sup-t"  # simultaneous CIs via sup-t
)
print(pred_family)
```

> ⚠️ **Trade-off:** Krinsky–Robb is fast (no model refits, just parameter draws and evaluations) but assumes the parameter distribution is multivariate normal. In small samples this can be a poor approximation. The point estimate is always the analytic `g(β̂)`; only SEs and CIs come from the simulation draws.

## When to use this

Use `vce="simulation"` when you want a non-parametric alternative to delta-method SEs, when computing simultaneous confidence intervals (`ci_method="sup-t"`, which requires draws), or when you want percentile-based CIs that can be asymmetric. It is the fastest alternative-VCE option because it avoids model refits entirely.

## When NOT to use this

> ⚠️ **Trade-off:** Do not use Krinsky–Robb when the sampling distribution of β̂ is far from normal (very small samples, rare outcomes). The simulation inherits the parametric covariance assumption. Do not use it when you want standard errors that account for model uncertainty (e.g., variable selection) — bootstrap is more appropriate there.

## See also

- {doc}`Reference: Margins.dydx </api>` — full parameter list for simulation VCE
- {doc}`How to compute bootstrap standard errors </howto/bootstrap>` — when you need model refits
- {doc}`How to compute simultaneous confidence intervals </howto/simultaneous_ci>` — `sup-t` and other `ci_method` options
