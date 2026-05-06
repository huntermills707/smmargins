# How to compute marginal effects for multinomial and ordered outcome models

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Explanation: Multi-outcome models </explanations/scales_link_functions>` — how K-outcome models expand the margin dimensions

## Problem statement

You have a multinomial logit (`MNLogit`) or ordered logit/probit (`OrderedModel`) and you want marginal effects for each outcome class probability. Every statistic returns K values (one per class) with full joint covariance across both variables and classes.

## Minimal working solution

`Margins` auto-detects multi-outcome models. Use `outcome=` to subset to specific classes.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from smmargins import Margins

rng = np.random.default_rng(7)
n = 500
df = pd.DataFrame({
    "x1": rng.standard_normal(n),
    "x2": rng.standard_normal(n),
})
# 3-class outcome
z0 = 0.5 + 1.0 * df["x1"] - 0.5 * df["x2"] + 0.3 * rng.standard_normal(n)
z1 = -0.3 + 0.5 * df["x1"] + 0.5 * df["x2"] + 0.3 * rng.standard_normal(n)
df["y"] = np.argmax(np.column_stack([np.zeros(n), z0, z1]), axis=1)

# Fit multinomial logit
X = sm.add_constant(df[["x1", "x2"]])
fit_mn = sm.MNLogit(df["y"], X).fit(disp=False)

M = Margins(fit_mn)
print(f"Number of outcomes: {M.n_outcomes}")
print(f"Outcome labels: {M.outcome_labels}")

# AAP for all classes (probabilities sum to 1)
aap = M.predict()
print(f"\nAAP (sum = {aap.estimate.sum():.6f}):")
print(aap.summary())

# AME of x1 for all outcome classes
ame = M.dydx("x1")
print("\nAME of x1 per outcome class:")
print(ame.summary())
```

## Variations

### Subset to specific outcomes

```python
# By index
print("Outcome 1 only:")
print(M.predict(outcome=1))

# By label
print("\nOutcomes 0 and 2:")
print(M.predict(outcome=[0, 2]))

# Post-hoc slicing of an existing result
full = M.dydx("x1")
sub = full.outcome(1)
print("\nPost-hoc slice for outcome 1:")
print(sub.summary())
```

### Multiple variables jointly

```python
# 2 variables * 3 outcomes = 6 rows; full joint vcov
joint = M.dydx(["x1", "x2"])
print(f"Shape: {joint.estimate.shape}")
print(f"Vcov shape: {joint.vcov.shape}")
```

### Elasticities on multi-outcome models

```python
# Elasticities work on multi-outcome models too
eyex = M.dydx("x1", method="eyex")
print(eyex.summary())
```

> ⚠️ **Trade-off:** MNLogit uses an analytic softmax-derivative (no extra forward predict calls), making it fast. OrderedModel uses central finite differences for the cumulative-link parameterization, which is slower. Elasticities on multi-outcome models emit a `RuntimeWarning` when any predicted class probability falls below `1e-12` (numerically unstable).

## When to use this

Use multi-outcome margins when your dependent variable has more than two unordered categories (MNLogit) or ordered categories (OrderedModel). The default behavior returns all K outcome classes — subset with `outcome=` when you only care about specific classes. The joint covariance across classes lets you form cross-outcome contrasts (e.g., "effect on P(class=1) minus effect on P(class=0)").

## When NOT to use this

> ⚠️ **Trade-off:** Do not treat the K AMEs as independent — they share the same parameter covariance and are constrained to sum to zero across outcomes for any given variable (in MNLogit). Single-outcome models silently ignore `outcome=`, so the same code paths work uniformly, but be careful not to assume `outcome=` does something on a binary logit.

## See also

- {doc}`Reference: Margins.n_outcomes </api>` — outcome class introspection
- {doc}`Reference: MarginsResult.outcome </api>` — post-hoc outcome slicing
- {doc}`How to perform joint tests and pairwise comparisons </howto/joint_tests_pairwise>` — `wald()` and `contrast()` on multi-outcome results
- {doc}`How to compute elasticities </howto/elasticities>` — elasticity methods on multi-outcome models
