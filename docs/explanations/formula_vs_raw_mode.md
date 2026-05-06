# Formula Mode vs. Raw Exog Mode

## Two Paths to the Same Destination

If you come from Stata, you are accustomed to a single path: you load a dataset, specify variables, and Stata handles the rest. There is no distinction between "formula mode" and "raw matrix mode" because Stata has no equivalent of a design matrix abstraction. Python does, and the choice between formula mode and raw exog mode is one of the most consequential decisions in `smmargins`.

If you use `statsmodels.get_margeff`, you are always in raw mode. The function operates on the fitted model's `exog` matrix with no knowledge of how that matrix was constructed from the original data. This is a fundamental limitation that `smmargins` resolves through formula mode.

---

## Formula Mode: Preserving the Full Data Pipeline

When you fit a statsmodels model with a formula string:

```python
model = smf.logit("y ~ x1 + I(x1**2) + x1:x2 + C(group)", data=df)
result = model.fit()
```

statsmodels (via patsy) stores the `DesignInfo` object — a complete recipe for transforming the data frame into the design matrix. This recipe includes:

- Which columns to extract from the data frame
- What transformations to apply (polynomials, interactions, splines)
- What contrast coding to use for categorical variables
- The order and names of the resulting design matrix columns

Formula mode in `smmargins` uses this `DesignInfo` object to rebuild the design matrix whenever a covariate is perturbed. The process:

$$\tilde{X} = \text{DesignInfo.build}(\text{data with } x_j \text{ perturbed})$$

What this means in code: `smmargins(result)` detects that `result` was fit with a formula, extracts the `DesignInfo`, and uses it for all subsequent predictions and marginal effects. When computing the marginal effect of `x1`, it modifies `x1` in the data frame and lets patsy regenerate `x1`, `I(x1**2)`, and `x1:x2` consistently.

---

## Raw Mode: Working with the Design Matrix Directly

When you fit a model with an explicit design matrix:

```python
X = df[["x1", "x1_sq", "x1_x2", "group_B", "group_C"]]
model = sm.Logit(df["y"], X)
result = model.fit()
```

there is no `DesignInfo`. The mapping from data columns to design matrix columns is lost. `smmargins` operates in raw mode, perturbing design matrix columns directly:

$$\tilde{X} = X + \Delta \cdot e_j$$

where $e_j$ is the unit vector for the $j$-th design matrix column.

What this means in code: `smmargins(result, exog=X)` operates on the matrix directly. When computing the marginal effect corresponding to the `x1` column, it adds a small perturbation to that column only. The `x1_sq` and `x1_x2` columns remain frozen at their original values.

---

## Where Raw Mode Fails

Raw mode produces incorrect marginal effects whenever the design matrix columns are functions of each other. Consider the polynomial example:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon$$

The true marginal effect of $x$ is $\beta_1 + 2\beta_2 x$. In formula mode, perturbing $x$ rebuilds both the $x$ and $x^2$ columns, and the marginal effect captures both terms. In raw mode, perturbing only the $x$ column gives $\beta_1$, missing the $2\beta_2 x$ contribution entirely.

The error magnitude depends on $\beta_2$ and $x$:

$$\text{Error}_{\text{raw}} = 2\beta_2 x \cdot \Delta$$

What this means in code: for a model with $x^2$ where $\beta_2 = 0.5$ and observations have $x \approx 5$, the raw-mode marginal effect is wrong by approximately $5.0 \cdot \Delta$ — a substantial bias that grows with $x$.

### Interaction Terms

For a model with $y \sim x_1 \cdot x_2$, the marginal effect of $x_1$ is:

$$\frac{\partial E[y]}{\partial x_1} = \beta_1 + \beta_{12} x_2$$

In raw mode, if `x1_x2` is a manually constructed column, perturbing `x1` does not update `x1_x2`. The $\beta_{12} x_2$ term is omitted, and the marginal effect is understated by $\beta_{12} x_2$.

### Splines

For $y \sim \text{bs}(x, \text{df}=5)$, patsy generates 5 B-spline basis columns that are all nontrivial functions of $x$. Perturbing one basis column while holding the others fixed has no meaningful interpretation. The marginal effect requires re-evaluating all 5 basis functions at $x + \Delta$.

---

## Where Raw Mode Is Correct

Raw mode is correct — and often preferred — when:

1. **The design matrix columns are independent.** Each column represents a distinct variable with no functional relationship to the others. A model like `y ~ x1 + x2 + x3` fit with raw exog has no hidden dependencies.

2. **You do not have the original data frame.** If you are working with published results (a coefficient vector and covariance matrix), you may only have the design matrix. Raw mode is the only option.

3. **You are conducting numerical experiments.** If you are manually constructing design matrices to test edge cases, raw mode gives you full control.

> ⚠️ **Trade-off:** Raw mode is faster (no patsy rebuilding) and works without the original data frame, but it requires you to guarantee that the design matrix columns are independently perturbable. Formula mode is slower but mathematically correct for any model structure.

---

## The Practical Recommendation

| Situation | Recommended Mode | Reason |
|---|---|---|
| Model fit with `smf.*` formula | Formula | DesignInfo available, all transformations tracked |
| Interactions in model | Formula | Raw mode cannot update interaction columns |
| Polynomial/spline terms | Formula | Raw mode cannot re-derive transformed columns |
| Categorical variables with contrasts | Formula | Raw mode cannot reconstruct contrast coding |
| Simple linear model, no transformations | Either | Columns are independent; raw mode is fine |
| Working with published results only | Raw | No data frame or formula available |
| Performance-critical applications | Raw | Avoids patsy rebuild overhead |

**The rule of thumb:** If your model formula contains anything beyond `+` — that is, if it uses `*`, `:`, `I()`, `C()`, `bs()`, `cr()`, or any other patsy transform — use formula mode. The cost of being wrong (silent incorrect marginal effects) far exceeds the cost of patsy rebuilding.

---

## What Happens at the API Boundary

`smmargins` auto-detects the mode from the fitted results object:

- If `result.model.data.design_info` exists (set by patsy during formula fitting), formula mode is used.
- If not, raw mode is used with the `exog` matrix stored in `result.model.exog`.

You can override the design used at evaluation time by passing `newdata=` explicitly to `predict()` or `dydx()`.

What this means in code: in most cases, you do not need to think about mode selection. Fit your model with a formula, pass the result to `smmargins`, and the correct mode is chosen automatically. Mode selection only becomes a conscious decision when (1) you fit without a formula, or (2) you are passing a custom `exog` to a predictions/margins call.

---

## Related Documentation

- **Tutorial:** {doc}`Formula Mode with Complex Transformations </tutorials/counterfactuals_and_plotting>` — side-by-side correct and incorrect marginal effects for polynomial, interaction, and spline models.
- **Reference:** {doc}`Margins.predict() </api>` and {doc}`Margins.dydx() </api>` for the `newdata` parameter; mode is auto-detected from the fitted results object passed to the {doc}`Margins </api>` constructor.
- **Explanation:** {doc}`Why Patsy Design Matrix Rebuilding Matters </explanations/patsy_design_rebuilding>` — the mathematical details of how formula mode preserves correct marginal effects.
