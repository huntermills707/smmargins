# Why Patsy Design Matrix Rebuilding Matters

## The Problem: Formulas Are Not Just Column Names

If you come from Stata, you are used to thinking about variables as atomic entities. When Stata's `margins` nudges `x1`, it operates on the variable `x1` in the dataset — end of story. This works because Stata does not have a formula system that transforms variables before they enter the model.

Python is different. A statsmodels formula like:

```
y ~ x1 + I(x1**2) + C(cat) + x1:x2 + bs(x3, df=5)
```

creates a design matrix where the columns are not the raw variables but *transformed* versions of them: a squared term, categorical dummy contrasts, an interaction column, and B-spline basis functions. If you nudge the design matrix column corresponding to `x1` while leaving the `I(x1**2)` and `x1:x2` columns untouched, you violate the mathematical structure of the model.

---

## The Wrong Way: Perturbing Design Matrix Columns Directly

Consider a model with a quadratic term:

$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \varepsilon$$

The marginal effect of $x$ is:

$$\frac{\partial E[y \mid x]}{\partial x} = \beta_1 + 2\beta_2 x$$

What this means in code: if you perturb the design matrix column for $x$ by $\Delta$ but leave the $x^2$ column unchanged, you are computing:

$$\frac{\partial E[y]}{\partial x_{\text{col}}} = \beta_1$$

and completely ignoring the $2\beta_2 x$ contribution from the quadratic term. The marginal effect is wrong by an amount that grows with $|x|$ and $|\beta_2|$.

---

## The Right Way: Perturb the Data, Rebuild the Design Matrix

The correct approach is to go back to the source data, perturb the raw variable $x$ by $\Delta$, and let patsy rebuild the entire design matrix from the modified data frame. This ensures that all derived terms — squares, interactions, splines, categorical contrasts — update consistently.

Mathematically, the marginal effect estimand is:

$$\frac{\partial}{\partial x_j} f\left(X(x_j)\beta\right)$$

where $X(x_j)$ emphasizes that the *entire* design matrix is a function of $x_j$, not just one column. By the chain rule:

$$\frac{\partial f}{\partial x_j} = \sum_{k=1}^{p} \frac{\partial f}{\partial X_k} \cdot \frac{\partial X_k}{\partial x_j}$$

What this means in code: `smmargins` stores the patsy `DesignInfo` object from the original model fit. When computing the marginal effect of `x1`, it creates a copy of the data frame, adds a small perturbation $\Delta$ to `x1`, and calls `DesignInfo.build_matrix(data_perturbed)` to regenerate the full design matrix. The derivative is then computed from the difference between predictions at the original and perturbed design matrices.

---

## What Gets Preserved

Patsy design rebuilding correctly handles:

### Polynomial Terms

For $y \sim x + I(x^2) + I(x^3)$, perturbing $x$ in the data frame updates all three terms. The marginal effect includes contributions from the linear, quadratic, and cubic components exactly as the chain rule demands.

### Interactions

For $y \sim x_1 \cdot x_2$ (which expands to $x_1 + x_2 + x_1:x_2$), perturbing $x_1$ updates both the $x_1$ main effect and the $x_1:x_2$ interaction column. The marginal effect of $x_1$ is:

$$\frac{\partial E[y]}{\partial x_1} = \beta_1 + \beta_{12} x_2$$

What this means in code: the interaction coefficient $\beta_{12}$ contributes to the marginal effect of $x_1$ in proportion to $x_2$. This contribution is only captured if the interaction column is rebuilt when $x_1$ is perturbed.

### Splines

For $y \sim \text{bs}(x, \text{df}=5)$, patsy generates 5 B-spline basis columns. Perturbing $x$ in the data frame causes patsy to recompute all 5 basis function evaluations. The marginal effect is the sum of the 5 basis derivatives weighted by their coefficients — a nontrivial function of $x$ that no column-wise perturbation could capture.

### Categorical Contrasts

For $y \sim C(x, \text{Treatment})$, patsy creates dummy columns with a specified contrast encoding. Design rebuilding preserves the contrast structure; raw exog manipulation would require the user to manually reconstruct the treatment contrast matrix.

---

## Raw Mode: When Design Rebuilding Is Impossible

> ⚠️ **Trade-off:** Formula mode is mathematically correct but requires that the model was fit with a patsy formula and that the `DesignInfo` object is available. Raw mode (passing `exog` directly) cannot track interactions or transformations because the mapping from data columns to design matrix columns is lost.

If you fit your model with `model.fit(exog=X, endog=y)` instead of a formula, `smmargins` has no `DesignInfo` to work with. It operates on `exog` columns directly. This means:

- Manually included interaction columns (e.g., `x1_x2 = x1 * x2` added to the matrix) will **not** update when `x1` is perturbed.
- Polynomial terms added as explicit columns will **not** update consistently.
- Splines evaluated manually and added as columns will **not** re-evaluate.

The marginal effect in raw mode is computed column by column, which is correct only if the design matrix columns are linearly independent functions of distinct underlying variables.

> ⚠️ **Trade-off:** Raw mode is necessary when you do not have the original data frame (e.g., you are working with a published covariance matrix and design matrix). In this case, you are responsible for ensuring that the design matrix structure supports column-wise perturbation.

---

## Formula Mode vs. Raw Mode: The Verdict

| Feature | Formula Mode | Raw Mode |
|---|---|---|
| Polynomial terms | Automatically tracked | Must be manually managed |
| Interactions | Rebuilt on perturbation | Frozen at fit-time values |
| Splines | Re-evaluated by patsy | Frozen at fit-time values |
| Categorical contrasts | Preserved via `DesignInfo` | User must reconstruct |
| Requires `DesignInfo` | Yes | No |
| Works with published matrices | No | Yes |

**Recommendation:** Use formula mode for any model with interactions or transformations. Use raw mode only when you have no access to the original data frame and formula.

---

## Related Documentation

- **Tutorial:** {doc}`Formula Mode with Interactions and Splines </tutorials/getting_started>` — worked examples showing correct vs. incorrect marginal effects.
- **Reference:** {doc}`Margins.predict() </api>` for the `newdata` parameter and formula/raw mode handling.
- **Explanation:** {doc}`Formula Mode vs. Raw Exog Mode </explanations/formula_vs_raw_mode>` — deeper comparison of the two input paths.
