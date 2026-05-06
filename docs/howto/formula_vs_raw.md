# How to choose between formula mode and raw exog mode

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing a basic AME
- {doc}`Explanation: Design matrix reconstruction </explanations/patsy_design_rebuilding>` — how patsy rebuilds the design matrix when variables are perturbed

## Problem statement

You need to decide whether to fit your model with a formula (e.g., `smf.logit("y ~ x1 + x2", data=df)`) or with raw matrices (e.g., `sm.Logit(y, X).fit()`). The choice affects whether interactions, polynomials, and transformations are correctly propagated when `Margins` perturbs variables for marginal effects.

## Minimal working solution

### Formula mode (recommended)

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from smmargins import Margins

rng = np.random.default_rng(7)
n = 500
df = pd.DataFrame({
    "x1": rng.standard_normal(n),
    "x2": rng.standard_normal(n),
    "grp": rng.choice(["A", "B", "C"], n),
})
df["y"] = (0.5 + 1.0 * df.x1 - 0.5 * df.x2 + (df.grp == "B") + 0.3 * rng.standard_normal(n) > 0).astype(int)

# Formula mode: interactions and polynomials are handled correctly
fit_formula = smf.logit("y ~ x1 + I(x1**2) + x1:x2 + C(grp)", data=df).fit(disp=False)
M_formula = Margins(fit_formula)

# All of these correctly update x1**2 and x1:x2 when perturbing x1
print(M_formula.dydx("x1"))
print(M_formula.dydx("grp"))
```

### Raw mode (limited)

```python
import statsmodels.api as sm

# Raw mode: manually build design matrix
X = sm.add_constant(df[["x1", "x2"]])
y = df["y"]
fit_raw = sm.Logit(y, X).fit(disp=False)
M_raw = Margins(fit_raw)

# Works for simple additive models
print(M_raw.dydx("x1"))
```

## Variations

### Raw mode with manually created interaction (incorrect)

```python
# Raw mode with an interaction column — MARGINS WILL BE WRONG
X_bad = sm.add_constant(df[["x1", "x2"]].copy())
X_bad["x1_x2"] = X_bad["x1"] * X_bad["x2"]  # manually created interaction

fit_bad = sm.Logit(y, X_bad).fit(disp=False)
M_bad = Margins(fit_bad)

# This perturbs only the "x1" column, NOT "x1_x2"
# The marginal effect of x1 will be incorrect
print(M_bad.dydx("x1"))  # WARNING: interaction not updated!
```

### Checking which mode is active

```python
# Inspect raw_mode flag
print(f"Formula mode: {not M_formula._raw_mode}")  # False (formula)
print(f"Raw mode:     {M_raw._raw_mode}")           # True (raw)
```

### Forcing raw mode with explicit data

```python
# If model.data.frame is missing, pass data explicitly
X_extra = sm.add_constant(df[["x1", "x2"]])
fit_extra = sm.Logit(y, X_extra).fit(disp=False)
M_extra = Margins(fit_extra, data=df[["x1", "x2"]])
print(M_extra.dydx("x1"))
```

> ⚠️ **Trade-off:** Formula mode uses patsy's `DesignInfo` to rebuild the design matrix whenever a variable is perturbed. This correctly handles `I(x**2)`, `x1:x2`, `C(group)`, splines (`bs(x, df=4)`), and all other patsy transforms. Raw mode perturbs only the literal column with the matching name — interactions and transformations are not automatically updated. Raw mode is faster and uses less memory for simple additive models without interactions.

## When to use formula mode

Use formula mode whenever your model includes interactions (`x1:x2`), polynomial terms (`I(x**2)`), categorical variables (`C(group)`), spline terms (`bs(x)`), or any other patsy transform. Formula mode is the default recommendation for all but the simplest models.

## When to use raw mode

Use raw mode when you have a simple additive model with no interactions or transformations and you want to avoid the patsy overhead. Raw mode is also necessary when working with models that do not support formulas (some custom statsmodels subclasses).

## When NOT to use raw mode

> ⚠️ **Trade-off:** Do not use raw mode when your design matrix contains manually created interaction columns, polynomial columns, or any derived feature that depends on the variable being perturbed. The marginal effects will be wrong because `Margins` does not know about the relationship between columns. If you must use raw mode with interactions, compute the marginal effect by hand or switch to a formula.

## See also

- {doc}`Explanation: Why patsy </explanations/patsy_design_rebuilding>` — the design matrix reconstruction mechanism
- {doc}`Reference: Margins raw_mode </api>` — inspecting the mode flag
- StatsModels documentation on `patsy` an