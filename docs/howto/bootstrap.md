# How to compute bootstrap standard errors with pairs, cluster, or block resampling

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Mathematical motivation </math>` — delta method and alternative VCEs

## Problem statement

You need standard errors that account for model uncertainty, not just parameter sampling variation. Your data has a clustered structure or time-series autocorrelation that makes pairs bootstrap invalid. You want to refit the model on resampled data and recompute marginal effects each time.

## Minimal working solution

Pass `vce="bootstrap"` to any `dydx` or `predict` call. Control the resampling scheme with `boot_method`, the number of replications with `n_boot`, and reproducibility with `boot_seed`.

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

# Pairs bootstrap (default)
ame_boot = M.dydx("age", vce="bootstrap", n_boot=1000, boot_seed=42)
print(ame_boot)

# Cluster bootstrap
print(M.dydx("age", vce="bootstrap", boot_method="cluster",
             cluster=df["region"], n_boot=500, boot_seed=42))

# Block bootstrap (for time series)
print(M.dydx("age", vce="bootstrap", boot_method="block",
             block_size=10, n_boot=500, boot_seed=42))
```

## Variations

### Parallel bootstrap with joblib

```python
# Use n_jobs=-1 for all cores, or specify a number
ame_parallel = M.dydx("age", vce="bootstrap", n_boot=2000, boot_seed=7, n_jobs=-1, verbose=True)
print(ame_parallel)
```

### Bootstrap for predictions at representative values

```python
pred_boot = M.predict(atexog={"age": [25, 45, 65]}, vce="bootstrap", n_boot=1000, boot_seed=42)
print(pred_boot)
```

### Comparing resampling schemes on the same data

```python
for method in ["pairs", "cluster", "block"]:
    kwargs = {"vce": "bootstrap", "boot_method": method, "n_boot": 500, "boot_seed": 7}
    if method == "cluster":
        kwargs["cluster"] = df["region"]
    elif method == "block":
        kwargs["block_size"] = 10
    se = M.dydx("age", **kwargs).se[0]
    print(f"  {method:8s}: SE = {se:.6f}")
```

> ⚠️ **Trade-off:** Bootstrap is the most flexible VCE because it refits the model on each draw, capturing model uncertainty. It is also the slowest — each replication requires a full model fit. Use `n_jobs` to parallelize. The point estimate remains the analytic `g(β̂)` from the original fit; bootstrap draws contribute only to SEs and CIs.

## When to use this

Use `vce="bootstrap"` when the delta-method or Krinsky–Robb assumptions are questionable (small samples, complex models), when you need cluster-aware resampling (`boot_method="cluster"`), or when working with time-series data (`boot_method="block"`). The cluster bootstrap is the gold standard for grouped data when the number of clusters is small.

## When NOT to use this

> ⚠️ **Trade-off:** Bootstrap is computationally expensive — 1000 replications means 1000 model refits. For large models or massive datasets, this can be prohibitive. Do not use pairs bootstrap on clustered data (SEs will be too small). Do not use block bootstrap without choosing `block_size` carefully — too small misses autocorrelation, too large wastes draws.

## See also

- {doc}`Reference: Margins.dydx </api>` — full parameter list for bootstrap VCE
- {doc}`How to use Krinsky–Robb simulation VCE </howto/kr_simulation>` — a faster alternative when model refits are unnecessary
- {doc}`How to compute robust and cluster-robust standard errors </howto/robust_clustered_ses>` — sandwich covariance without refits
