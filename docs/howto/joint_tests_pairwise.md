# How to perform joint Wald tests and pairwise comparisons on marginal effects

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`How to compute subgroup-specific AMEs </howto/subgroup_analysis>` — `over=` produces multi-row results suitable for joint testing

## Problem statement

You have computed multiple marginal effects (e.g., AMEs of several variables, or discrete contrasts across levels of a factor) and you want to test joint hypotheses (e.g., "are all AMEs simultaneously zero?") or compare all pairs of factor levels against each other.

## Minimal working solution

Use `.wald()` on a `MarginsResult` for joint tests, and `.pairwise()` for all level-vs-level comparisons on a discrete variable.

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

# Joint Wald test: are all three AMEs simultaneously zero?
res = M.dydx(["age", "income", "female"])
joint = res.wald()
print(f"Joint test: chi2={joint.stat:.4f}, df={joint.df}, p={joint.pvalue:.4f}")

# Pairwise comparisons across education levels
res_educ = M.dydx("educ")
pw = res_educ.pairwise(by="educ")
print("\nPairwise comparisons (unadjusted):")
print(pw.summary())

# Pairwise with Bonferroni-adjusted CIs
pw_bonf = res_educ.pairwise(by="educ", ci_method="bonferroni")
print("\nPairwise comparisons (Bonferroni-adjusted):")
print(pw_bonf.summary())
```

## Variations

### Custom linear restriction

```python
# Test H0: AME(age) == AME(income)
res = M.dydx(["age", "income", "female"])
custom = res.wald(C=[[1, -1, 0]])
print(f"H0: AME(age)=AME(income): chi2={custom.stat:.4f}, p={custom.pvalue:.4f}")
```

### Contrast on a result with arbitrary weights

```python
# Linear contrast on subgroup AMEs: test male vs female AME of age
res = M.dydx("age", over="female")
c = res.contrast(np.array([1, -1]), labels=["male - female"])
print(c.summary())
```

### Pairwise with simultaneous CIs via sup-t

```python
# sup-t CIs for pairwise comparisons (requires simulation draws)
pw_supt = res_educ.pairwise(by="educ", vce="simulation", n_sims=2000, sim_seed=42, ci_method="sup-t")
print(pw_supt.summary())
```

> ⚠️ **Trade-off:** `wald()` returns a `WaldResult` with the test statistic, degrees of freedom, and p-value. It uses the full joint covariance of the `MarginsResult`, so the test is exact under the delta-method approximation. `pairwise()` builds a contrast matrix internally and returns a new `MarginsResult` — you can chain `ci_method=` for multiple-comparison adjustments.

## When to use this

Use `.wald()` when you need an omnibus test (e.g., "does this set of variables have any effect?"). Use `.pairwise()` when you have a categorical variable and want all level-vs-level comparisons. Use `contrast()` when you have a specific linear combination in mind that is not covered by the built-in pairwise machinery.

## When NOT to use this

> ⚠️ **Trade-off:** Do not use `.wald()` on a single-row result — it is degenerate (tests one restriction, equivalent to a z-test). Do not use `.pairwise()` on continuous variables — it is designed for discrete contrasts. `contrast()` does not support nonlinear restrictions; for those, use `vce="bootstrap"` and test on the draw matrix directly.

## See also

- {doc}`Reference: MarginsResult.wald </api>` — full API for Wald tests
- {doc}`Reference: MarginsResult.pairwise </api>` — pairwise comparison API
- {doc}`Reference: MarginsResult.contrast </api>` — linear contrast API
- {doc}`How to compute simultaneous confidence intervals </howto/simultaneous_ci>` — Bonferroni/Sidak/sup-t for pairwise CIs
