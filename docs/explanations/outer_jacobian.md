# Analytic vs. Finite-Difference Jacobians

## The Problem: Building the $G$ Matrix

The delta method requires the Jacobian matrix $G = \partial g / \partial \beta$. For marginal effects, this is an *outer* Jacobian: the derivative of a scalar summary statistic with respect to the $p$ model parameters. There are two ways to compute it: analytically (symbolic derivatives) and by finite differences (numerical differentiation).

If you come from Stata, you have only ever used finite-difference Jacobians. Stata's `margins` computes all standard errors by numeric differentiation — there is no analytic path. `smmargins` offers both, and the choice matters for speed and numerical stability.

---

## The Analytic Path: Symbolic Derivatives of the Mean Function

For generalized linear models (GLMs), the mean function $f(\eta)$ and its derivative $f'(\eta)$ are known in closed form. The statsmodels `family` object exposes these through `link.inverse` and `link.inverse_deriv`.

For a logit model, the mean function and its derivative are:

$$f(\eta) = \Lambda(\eta) = \frac{1}{1 + e^{-\eta}}$$

$$f'(\eta) = \Lambda(\eta) \cdot (1 - \Lambda(\eta))$$

What this means in code: when you call `dydx()` on a GLM or GLM-like model (Logit, Probit, Poisson, NegativeBinomial), `smmargins` calls `family.link.inverse_deriv(eta)` to evaluate $f'(\eta)$ at the linear predictor for every observation. This is a vectorized operation with no additional model evaluations.

For the second derivative (needed for marginal effects, since the marginal effect itself is $f'(\eta) \cdot \beta_j$), the logit case is:

$$f''(\eta) = \Lambda(\eta) \cdot (1 - \Lambda(\eta)) \cdot (1 - 2\Lambda(\eta))$$

What this means in code: the outer Jacobian row for covariate $j$ is computed as the mean of $f''(x_i'\hat{\beta}) \cdot \hat{\beta}_j \cdot x_i$ across observations. This is one pass over the data after $\hat{\beta}$ is known.

---

## Special Cases in the Analytic Path

### OLS, WLS, GLS: The Identity Link

For linear models, $f(\eta) = \eta$, so $f'(\eta) = 1$ and $f''(\eta) = 0$. The marginal effect of $x_j$ is simply $\hat{\beta}_j$, a constant. The Jacobian is trivial:

$$\frac{\partial g}{\partial \beta} = e_j'$$

where $e_j$ is the unit vector selecting the $j$-th coefficient. The delta-method variance collapses to $\text{Var}(\hat{\beta}_j)$, which is exactly the regression standard error squared.

What this means in code: for linear models, `smmargins` bypasses all finite differencing entirely. The standard error comes directly from `cov_params()[j, j]`.

### MNLogit: The Softmax Jacobian

For multinomial logit with $K$ outcome categories, the mean function is the softmax:

$$P(y = k \mid x) = \frac{e^{x'\beta_k}}{\sum_{j=1}^{K} e^{x'\beta_j}}$$

The Jacobian of this $K$-dimensional output with respect to the stacked parameter vector requires the softmax derivative matrix. `smmargins` implements this directly using the known closed form:

$$\frac{\partial P_k}{\partial \beta_m} = P_k \left( \mathbb{1}_{k=m} - P_m \right) x$$

What this means in code: multinomial logit gets a fully analytic path, with no finite differencing even for cross-category marginal effects.

---

## The Finite-Difference Fallback

When an analytic derivative is not available, `smmargins` falls back to finite differences. This triggers in three situations:

1. **Non-GLM models** with no closed-form mean function derivative (e.g., custom models).
2. **Offset or exposure terms** present, which compose nonlinearly with the link function.
3. **OrderedModel** (ordered probit/logit), where the cumulative probability structure makes analytic Jacobians substantially more complex.

The finite-difference step size follows the cube-root rule:

$$h = \varepsilon^{1/3} \cdot \max(|x|, 1)$$

where $\varepsilon$ is machine epsilon (approximately $2.2 \times 10^{-16}$ for float64), giving $h \approx 10^{-5}$.

What this means in code: for each parameter $\beta_j$, `smmargins` perturbs $\beta_j$ by $\pm h$, re-evaluates the statistic $g(\beta)$, and computes:

$$\frac{\partial g}{\partial \beta_j} \approx \frac{g(\beta + h \cdot e_j) - g(\beta - h \cdot e_j)}{2h}$$

This requires $2p$ statistic evaluations for a central difference, or $p$ evaluations for a forward difference.

---

## The Speed Difference

> ⚠️ **Trade-off:** Analytic Jacobians eliminate $p$ forward `predict()` calls per statistic, but they require that the mean function and its derivatives be correctly implemented for every model family. Finite differences are slower but universal — they work for any model that can make predictions.

Consider a logit model with $p = 50$ parameters and $m = 10$ covariates of interest:

- **Analytic path:** one pass over the data to compute $f''(\eta) \cdot x$, then $m \times p$ multiplications. Negligible overhead.
- **Finite-difference path:** for each of $m$ statistics, perturb each of $p$ parameters and re-predict. This is $m \times p = 500$ additional model evaluations.

The analytic path is typically 100-1000x faster for moderate-sized models.

> ⚠️ **Trade-off:** Finite differences can suffer from truncation error (if $h$ is too large) or roundoff error (if $h$ is too small). The cube-root rule balances these, but for parameters near zero or with extreme scale differences, the optimal $h$ may vary. Analytic derivatives avoid this entirely.

---

## Stata's Approach vs. `smmargins`

Stata's `margins` uses finite differences for everything. This is a defensible design choice — it means `margins` works with any model that Stata can fit, including user-written estimators, with no family-specific code. The cost is speed.

`smmargins` takes the opposite default: use analytic derivatives whenever possible, fall back to finite differences only when necessary. For the vast majority of applied work (logit, probit, Poisson, OLS), this means delta-method standard errors are effectively instantaneous.

---

## Related Documentation

- **Tutorial:** {doc}`Speed Comparison: Analytic vs. Finite-Difference Jacobians </tutorials/inference>` — benchmarks on large models.
- **Reference:** {doc}`Margins </api>` for the `analytic` constructor flag that selects between the analytic chain rule and the central-difference fallback.
