# How to plot predictions, slopes, and comparisons

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — fitting a model and computing predictions
- {doc}`Tutorial: Counterfactual predictions and plotting </tutorials/counterfactuals_and_plotting>` — notebook walkthrough with rendered figures

## Problem statement

You have fitted a model and want to visualise how predictions, marginal effects, or discrete contrasts vary across a covariate. `smmargins` provides three plotting helpers — `plot_predictions`, `plot_slopes`, and `plot_comparisons` — that grid a conditioning variable, compute the relevant statistic at each grid point, and render the curve with confidence intervals.

## Minimal working solution

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
})
df["voted"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-(
    -4 + 0.05 * df.age + 0.00001 * df.income
    + 0.8 * (df.educ == "college") + 1.4 * (df.educ == "grad")
    + 0.3 * df.female
)))).astype(int)

fit = smf.logit("voted ~ age + income + C(educ) + female", data=df).fit(disp=False)
M = Margins(fit)

# Predicted probability of voting across age
fig, ax = M.plot_predictions(condition="age")

# Marginal effect of age across income
fig, ax = M.plot_slopes("age", condition="income")

# Discrete contrast (female=1 vs female=0) across age
fig, ax = M.plot_comparisons("female", condition="age")
```

## `plot_predictions`

`plot_predictions` grids the conditioning variable over its observed range (50 points for numerics, all unique levels for categorics), calls :meth:`~smmargins.Margins.predict` at each point, and plots the curve.

### Single conditioning variable

```python
fig, ax = M.plot_predictions(condition="age")
```

### Faceting with `by=`

Pass a categorical column to draw separate lines:

```python
fig, ax = M.plot_predictions(condition="age", by="educ")
```

### Custom grid with a dict

```python
fig, ax = M.plot_predictions(condition={"age": np.linspace(18, 90, 100)})
```

### Inference options

All kwargs accepted by :meth:`~smmargins.Margins.predict` are forwarded, so you can change the VCE or confidence level:

```python
fig, ax = M.plot_predictions(
    condition="age",
    vce="simulation",
    n_sims=2000,
    ci=0.90,
    ci_method="pointwise",
)
```

## `plot_slopes`

`plot_slopes` grids the conditioning variable and calls :meth:`~smmargins.Margins.dydx` at each point.

```python
# dydx(age) as a function of age — shows non-linearity
fig, ax = M.plot_slopes("age", condition="age")

# dydx(age) across income, faceted by education
fig, ax = M.plot_slopes("age", condition="income", by="educ")
```

You can pass any `dydx` keyword (e.g. `method="eyex"`, `scale="linear"`):

```python
fig, ax = M.plot_slopes("income", condition="age", method="eyex")
```

## `plot_comparisons`

`plot_comparisons` plots a contrast — either a discrete factor level comparison or a numeric step (+1 vs baseline) — across a conditioning variable.

```python
# Female effect (1 - 0) across age
fig, ax = M.plot_comparisons("female", condition="age")

# Education contrast (grad vs hs) across age
fig, ax = M.plot_comparisons("educ", condition="age")
```

For numeric variables the comparison is `x + 1` versus `x`. For categorical variables it is the second level versus the first.

## Re-using an Axes object

All three functions accept `ax=` so you can overlay curves or compose subplots manually:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
M.plot_predictions(condition="age", ax=axes[0])
M.plot_predictions(condition="income", ax=axes[1])
plt.tight_layout()
```

## When to use this

Use the plotting helpers when you want a quick, correct visualisation that respects the model's non-linearity and uncertainty. Because the grid is computed with the same `values=` DSL used elsewhere, the plots are consistent with tabular outputs.

## When NOT to use this

> ⚠️ **Trade-off:** The plotting API is convenience-only. If you need full control over aesthetics, call :meth:`~smmargins.Margins.predict` or :meth:`~smmargins.Margins.dydx` with a custom grid and plot the resulting DataFrame yourself.
>
> ⚠️ **Trade-off:** `plot_comparisons` chooses the comparison automatically (numeric step or first two factor levels). For more complex contrasts, compute the contrast manually with :meth:`~smmargins.Margins.contrast` and plot the result.

## See also

- {doc}`Tutorial: Counterfactual predictions and plotting </tutorials/counterfactuals_and_plotting>` — executable notebook with all three plot types
- {doc}`Reference: plot_predictions </api>` — full signature and forwarded kwargs
- {doc}`Reference: plot_slopes </api>` — full signature and forwarded kwargs
- {doc}`Reference: plot_comparisons </api>` — full signature and forwarded kwargs
