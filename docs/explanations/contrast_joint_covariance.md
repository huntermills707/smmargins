# Why `Margins.contrast` exists: joint covariance for two arms

## The problem

A common task is to compute a *contrast* between two counterfactual specifications — for example, the population average treatment effect (PATE):

$$
\Delta \;=\; \mathbb{E}\bigl[g_A(\beta)\bigr] \;-\; \mathbb{E}\bigl[g_B(\beta)\bigr],
$$

where $g_A$ averages the predicted response with `treat=1` and $g_B$ does the same with `treat=0`. A user could compute this naïvely:

```python
r_a = M.predict(values={"treat": 1})
r_b = M.predict(values={"treat": 0})
diff = r_a.estimate - r_b.estimate           # correct
naive_se = np.sqrt(r_a.se**2 + r_b.se**2)    # WRONG
```

The point estimate is fine. **The standard error is wrong**, often badly.

## Why the naïve SE is wrong

The two predictions $g_A$ and $g_B$ depend on the *same* $\hat\beta$. Their sampling errors are correlated — strongly, since they share every coefficient. The variance of the difference is:

$$
\operatorname{Var}\bigl(g_A(\hat\beta) - g_B(\hat\beta)\bigr)
\;=\;
\operatorname{Var}(g_A) + \operatorname{Var}(g_B) - 2\,\operatorname{Cov}(g_A, g_B).
$$

The naïve formula sets the covariance to zero. For PATE in a logit, where $g_A$ and $g_B$ differ only by one regressor, the covariance term is typically **the same order of magnitude as the variances**, and the naïve SE can be off by a factor of 2 or more.

## The delta-method derivation

Let $J_A$ and $J_B$ be the Jacobians of $g_A$ and $g_B$ with respect to $\beta$, evaluated at $\hat\beta$. Stack them:

$$
J \;=\;
\begin{bmatrix} J_A \\ J_B \\ J_A - J_B \end{bmatrix},
\qquad
g(\hat\beta) \;=\;
\begin{bmatrix} g_A(\hat\beta) \\ g_B(\hat\beta) \\ g_A(\hat\beta) - g_B(\hat\beta) \end{bmatrix}.
$$

The first-order delta-method approximation gives a single joint covariance matrix:

$$
\operatorname{Var}\bigl(g(\hat\beta)\bigr) \;\approx\; J\,\widehat V\,J^\top,
$$

where $\widehat V$ is the estimated covariance of $\hat\beta$. The (3, 3) block of this matrix is the *correct* variance of the difference — and it automatically picks up the off-diagonal $\operatorname{Cov}(g_A, g_B)$ because of the way $J_A - J_B$ is built into the stack.

Equivalently, observe that

$$
\bigl[J\,\widehat V\,J^\top\bigr]_{33}
\;=\;
J_A \widehat V J_A^\top + J_B \widehat V J_B^\top - 2\,J_A \widehat V J_B^\top,
$$

so the joint computation is simply doing the bookkeeping the user would otherwise have to do by hand.

## What the implementation does

`Margins.contrast` (`smmargins/core.py:1442`) builds exactly that stacked statistic and Jacobian:

1. Resolve arm A's frames via the same path `predict()` uses (with `values=a` or `a_newdata=`).
2. Resolve arm B's frames the same way.
3. Form $g(\beta) = [g_A(\beta), g_B(\beta), g_A(\beta) - g_B(\beta)]$ as a single vector-valued statistic.
4. Form $J(\beta)$ by stacking $J_A$, $J_B$, and $J_A - J_B$.
5. Pass through the standard `_delta` runner. Inference kwargs (`vce=`, `cov_type=`, `ci_method=`) compose unchanged.

The result is a standard `MarginsResult` with three rows. The third row's SE is the joint-covariance SE. The full $3 \times 3$ block is available as `result.vcov` for users who want to construct further linear combinations.

## Composition with non-delta inference

For `vce="bootstrap"` and `vce="simulation"`, the joint computation is *automatic*: a single resample (or a single $\hat\beta$ draw) produces both $g_A^{(b)}$ and $g_B^{(b)}$, and the contrast at draw $b$ is just $g_A^{(b)} - g_B^{(b)}$. The empirical covariance of those draws gives the joint covariance with no extra work — the bookkeeping is implicit in the resampling design.

For `ci_method="sup-t"`, the simultaneous CI uses the joint draw matrix across all three rows, so the band on the difference is calibrated against $g_A$ and $g_B$ jointly.

## Tradeoffs and pitfalls

- **`a_newdata` and `b_newdata` should usually be the same length.** They don't have to be — `M.contrast(a_newdata=df_treated, b_newdata=df_control)` is well-defined when treated and control populations differ — but the contrast then estimates a *target-population-specific* effect, not PATE on the original sample. Document which population you mean.
- **The joint $\widehat V$ assumption requires the two arms to be evaluated against the same $\hat\beta$.** If you fit two separate models and want to contrast them, this machinery is the wrong tool — you need a different covariance estimator (typically a stacked-GMM or seemingly-unrelated approach).
- **Bootstrap arm consistency.** Each resample refits once and computes both arms against the refit's coefficients. Don't bootstrap arms independently — that throws away the very correlation the joint computation is designed to capture.

## See also

- Tutorial: contrast walkthrough (forthcoming).
- API: `smmargins.Margins.contrast` (see {doc}`../api`).
- Theory: {doc}`../math` for the general delta-method derivation underlying every `MarginsResult`.
