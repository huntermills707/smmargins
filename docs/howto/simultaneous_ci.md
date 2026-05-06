# How to compute simultaneous confidence intervals for families of margins

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing basic predictions
- {doc}`How to use Krinsky–Robb simulation VCE </howto/kr_simulation>` — `vce="simulation"` is required for `sup-t`
- {doc}`How to compute bootstrap standard errors </howto/bootstrap>` — `vce="bootstrap"` also works with `sup-t`

## Problem statement

You are computing multiple marginal effects or predictions (a "family" of margins) and want confidence intervals that control the family-wise error rate. Standard pointwise CIs under-cover the joint event "all intervals contain their true values." You need Bonferroni, Sidak, or sup-t adjustments.

## Minimal working solution

Pass `ci_method` to any `predict` or `dydx` call. Use `"bonferroni"` or `"sidak"` with any VCE; use `"sup-t"` with `vce="simulation"` or `vce="bootstrap"`.

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

# Family of 5 predictions at different ages
ages = [25, 35, 45, 55, 65]
common = dict(atexog={"age": ages}, vce="simulation", n_sims=4000, sim_seed=123)

pw   = M.predict(**common, ci_method="pointwise")
bonf = M.predict(**common, ci_method="bonferroni")
sidk = M.predict(**common, ci_method="sidak")
supt = M.predict(**common, ci_method="sup-t")

# Compare CI widths
for label, res in [("pointwise", pw), ("bonferroni", bonf), ("sidak", sidk), ("sup-t", supt)]:
    widths = res.ci_upper - res.ci_lower
    print(f"{label:12s}: avg width = {widths.mean():.6f}")
```

## Variations

### Simultaneous CIs for multiple marginal effects (delta-method)

```python
# Bonferroni on dydx — works with vce="delta" (the default)
res = M.dydx(["age", "income", "female"], ci_method="bonferroni")
print(res)
```

### Sidak vs Bonferroni comparison

```python
# Sidak is slightly narrower than Bonferroni (equal under independence)
for method in ["bonferroni", "sidak"]:
    r = M.dydx(["age", "income"], ci_method=method)
    width = r.ci_upper - r.ci_lower
    print(f"{method:12s}: age width = {width[0]:.6f}, income width = {width[1]:.6f}")
```

### sup-t with bootstrap draws

```python
# sup-t consumes the draw matrix from bootstrap
supt_boot = M.predict(
    atexog={"age": [25, 45, 65]},
    vce="bootstrap", n_boot=1000, boot_seed=42,
    ci_method="sup-t"
)
print(supt_boot)
```

> ⚠️ **Trade-off:** Bonferroni is conservative (CIs are wider than necessary) but works with any VCE and any correlation structure. Sidak is slightly narrower but assumes independence (safe fallback when you don't know the correlation). sup-t is the sharpest — it exploits correlation in the draw matrix — but requires `vce="simulation"` or `vce="bootstrap"`.

## When to use this

Use `ci_method="bonferroni"` when you want a simple, conservative family-wise correction that works everywhere. Use `"sidak"` when you know the margins are approximately independent. Use `"sup-t"` when you are already using simulation or bootstrap VCE and want the tightest simultaneous bands.

## When NOT to use this

> ⚠️ **Trade-off:** Do not use `"sup-t"` with `vce="delta"` — it raises `ValueError` because sup-t requires simulation or bootstrap draws. Do not use pointwise CIs when presenting a family of margins as jointly significant — the joint coverage rate can be far below the nominal level.

## See also

- {doc}`Reference: MarginsResult.ci_method </api>` — how the CI method is stored and reported
- {doc}`How to use Krinsky–Robb simulation VCE </howto/kr_simulation>` — fast draws for sup-t
- {doc}`How to compute bootstrap standard errors </howto/bootstrap>` — bootstrap draws for sup-t
