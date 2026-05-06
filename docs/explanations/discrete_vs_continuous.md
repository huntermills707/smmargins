# Discrete vs. Continuous Variable Detection

## The Problem: What Does "Nudge" Mean?

If you come from Stata, `margins, dydx(x)` uses a finite difference for discrete variables and a derivative for continuous ones. Stata determines which to use based on whether a variable is specified as a factor (`i.x`) or continuous. In Python, `smmargins` must infer this from the data itself — and the distinction matters for both the point estimate and its interpretation.

For a continuous variable $x_j$, the marginal effect is the derivative:

$$\frac{\partial E[y \mid x]}{\partial x_j} = \lim_{\Delta \to 0} \frac{E[y \mid x_j + \Delta] - E[y \mid x_j]}{\Delta}$$

For a discrete variable (binary, categorical, or integer-count), the derivative does not exist. The relevant quantity is the finite difference:

$$E[y \mid x_j = a + 1] - E[y \mid x_j = a]$$

What this means in code: `smmargins` must decide, for each variable, whether to compute a derivative (analytic or finite-difference slope approaching zero) or a discrete jump (difference between two distinct covariate values). Getting this wrong produces either a nonsensical derivative of a step function or an underwhelming finite difference that misses the true effect.

---

## Automatic Detection

`smmargins` uses two criteria to auto-detect variable type:

### Criterion 1: Data Type

Variables with pandas dtype `category`, `bool`, or `object` are automatically treated as discrete. Variables with numeric dtypes (`int64`, `float64`) are candidates for continuous treatment.

### Criterion 2: Number of Unique Values

A numeric variable with $k$ unique values is treated as discrete if $k$ is small relative to the sample size. The default threshold treats variables with 10 or fewer unique values as discrete.

The logic is: a variable with only 2 unique values is clearly binary; a variable with 3-10 unique values is likely ordinal or categorical with numeric coding; a variable with hundreds of unique values is effectively continuous.

> ⚠️ **Trade-off:** Auto-detection based on unique value counts is a heuristic. A variable with 10 unique values might be a finely graded Likert scale (discrete) or a rounded continuous variable (continuous). The threshold of 10 is a sensible default, but you should always verify the classification against your substantive knowledge of the variable.

---

## The Discrete Difference Computation

For a discrete variable $x_j$ taking values $\{a_1, a_2, \ldots, a_k\}$, `smmargins` computes the marginal effect as the change from one level to the next. By default, for a binary variable ($k=2$), this is:

$$\Delta_j = E[y \mid x_j = 1, x_{-j}] - E[y \mid x_j = 0, x_{-j}]$$

What this means in code: the `dydx()` method creates two counterfactual datasets — one with $x_j$ set to its reference level and one with $x_j$ set to its comparison level — computes predictions for both, and differences them. The standard error comes from the delta method applied to the contrast between the two predictions.

For a categorical variable with $k$ levels, `smmargins` computes $k-1$ contrasts against a reference level, analogous to Stata's `margins, dydx(i.cat)`.

---

## The Count Option: Integer Unit Increments

For count variables (number of children, years of education, doctor visits), the relevant comparison is often a one-unit increase rather than a jump from 0 to the maximum. The `count=True` option enforces this:

$$\Delta_j^{\text{count}} = E[y \mid x_j = x_{ij} + 1] - E[y \mid x_j = x_{ij}]$$

What this means in code: `dydx("children", count=True)` computes the effect of an additional child for each observation, then averages. This is a discrete difference with a step size of 1, appropriate for count data where fractional increments are not meaningful.

---

## Elasticity-Style Methods and Discrete Variables

The elasticity-style methods compute proportional changes and are mathematically undefined for discrete variables:

- `"eyex"`: $\frac{\partial \log E[y]}{\partial \log x} = \frac{\partial E[y]}{\partial x} \cdot \frac{x}{E[y]}$ — elasticity of $y$ with respect to $x$
- `"dyex"`: $\frac{\partial E[y]}{\partial \log x} = \frac{\partial E[y]}{\partial x} \cdot x$ — semi-elasticity
- `"eydx"": $\frac{\partial \log E[y]}{\partial x} = \frac{\partial E[y]}{\partial x} \cdot \frac{1}{E[y]}$ — semi-elasticity (log outcome)

These formulas all involve $\partial E[y] / \partial x$, which assumes $x$ is differentiable. For a discrete variable, division by zero or the logarithm of a non-positive value can occur.

What this means in code: if you call `dydx(x, method="eyex")` on a variable that `smmargins` has classified as discrete, it raises a clear error:

```
ValueError: method="eyex" requires continuous variables. Variable "x" is discrete.
Use discrete=False to override if appropriate.
```

> ⚠️ **Trade-off:** Elasticity methods provide intuitive "percent change" interpretations but are only valid for strictly positive continuous variables. For discrete or zero-inflated variables, the percentage change framing is misleading. Use standard discrete differences instead.

---

## Manual Override

You can override auto-detection with the `discrete` parameter:

- `discrete=True`: Treat the variable as discrete, compute finite differences
- `discrete=False`: Treat the variable as continuous, compute derivatives

This is essential when the heuristic misclassifies a variable. For example, a variable coded as integers 0-5 representing a continuous quantity (rounded income in thousands) should be treated as continuous:

What this means in code: `dydx("income_rounded", discrete=False)` forces derivative-based computation even though the variable has only 6 unique values. Conversely, a variable stored as `float64` with values $\{0.0, 1.0\}$ should be treated as discrete: `dydx("flag", discrete=True)`.

---

## Summary Table

| Variable Type | Auto-Detected As | Default Computation | Override |
|---|---|---|---|
| Binary (0/1) | Discrete | $E[y \mid 1] - E[y \mid 0]$ | `discrete=True/False` |
| Categorical | Discrete | Contrasts vs. reference level | N/A |
| Integer count | Discrete (if $k \leq 10$) | Unit increment difference | `count=True` |
| Numeric, many unique | Continuous | Derivative $\partial E[y]/\partial x$ | `discrete=True` |
| Numeric, few unique | Discrete | Finite difference | `discrete=False` |

---

## Related Documentation

- **Tutorial:** {doc}`Marginal Effects for Discrete and Continuous Variables </tutorials/marginal_effects>` — compare point estimates across detection modes.
- **Reference:** {doc}`Margins.dydx() </api>` for the `discrete`, `count`, and `method` parameters.
