# Comparing Inference Methods: Delta vs. KR vs. Bootstrap

## One Estimand, Three Routes to Uncertainty

If you come from Stata, inference after `margins` is the delta method — period. Stata offers no built-in alternative for marginal effect standard errors. If you use `statsmodels.get_margeff`, the same limitation applies: delta method only, take it or leave it.

`smmargins` offers three inference methods, each with different assumptions, computational costs, and robustness properties. Choosing among them is a genuine statistical decision, not a matter of taste.

---

## The Delta Method

The delta method linearizes the estimator via Taylor expansion and uses the model's estimated covariance matrix:

$$\widehat{\text{Var}}[g(\hat{\beta})] = G \hat{V} G'$$

where $G = \partial g(\beta) / \partial \beta' |_{\hat{\beta}}$ and $\hat{V}$ is the model's `cov_params()`.

### Strengths
- **Fast:** A single matrix multiplication after the Jacobian is computed. No resampling, no simulation, no refitting.
- **Asymptotically exact:** Under standard regularity conditions, delta-method standard errors converge to the true sampling variance at rate $O(n^{-1})$.
- **Deterministic:** The same input always produces the same standard error. Reproducibility is trivial.

### Weaknesses
- **Finite-sample bias:** The Taylor approximation can understate variance when $g$ is highly nonlinear or the sample is small.
- **Normality assumption:** Relies on asymptotic normality of $\hat{\beta}$. For small samples with binary outcomes, the sampling distribution may be skewed.
- **No cluster awareness:** Unless $\hat{V}$ itself was computed with cluster-robust covariance (e.g., `cov_type="cluster"`), the delta method ignores cluster structure. Even with cluster-robust $\hat{V}$, it assumes the cluster structure only affects covariance estimation, not the sampling distribution of $g(\hat{\beta})$.

> ⚠️ **Trade-off:** Delta method is the right default for well-behaved models with $n > 200$. For small samples or extreme predicted probabilities, consider KR simulation or bootstrap.

---

## Kenward-Roger (KR) Simulation

KR simulation takes a different approach: instead of approximating the distribution of $g(\hat{\beta})$, it simulates from the approximate sampling distribution of $\hat{\beta}$ and evaluates $g$ at each draw.

The procedure:

1. Draw $\beta^{(b)} \sim N(\hat{\beta}, \hat{V})$ for $b = 1, \ldots, B$.
2. Compute $g^{(b)} = g(\beta^{(b)})$ for each draw.
3. The standard error is the empirical standard deviation of $\{g^{(b)}\}$.

Mathematically, the KR variance estimator is:

$$\widehat{\text{Var}}_{\text{KR}}[g(\hat{\beta})] = \frac{1}{B-1} \sum_{b=1}^{B} \left(g^{(b)} - \bar{g}\right)^2$$

What this means in code: `dydx(..., vce="simulation", n_sims=B)` draws $B$ parameter vectors from a multivariate normal with mean $\hat{\beta}$ and covariance $\hat{V}$, re-evaluates the marginal effect statistic at each draw, and reports the empirical standard deviation. No model is refit.

### Strengths
- **No model refitting:** Much faster than bootstrap. The cost is $B$ statistic evaluations, not $B$ model fits.
- **Handles moderate nonlinearity:** By evaluating $g$ exactly (not linearized) at each draw, KR captures some curvature that the delta method misses.
- **Deterministic given seed:** With a fixed random seed, results are reproducible.

### Weaknesses
- **Still assumes normality:** The draws come from $N(\hat{\beta}, \hat{V})$. If the true sampling distribution is non-normal (small sample, rare events), KR inherits that misspecification.
- **Assumes $\hat{V}$ is correct:** If the covariance matrix is misspecified, KR simulates from the wrong distribution.
- **No cluster resampling:** Like the delta method, KR does not resample clusters. It simulates parameter uncertainty, not sampling uncertainty.

> ⚠️ **Trade-off:** KR simulation is a middle ground. It is nearly as fast as the delta method (no refitting) but more accurate for moderately nonlinear $g$. It is the best choice when the delta method feels too approximate but bootstrap is too slow.

---

## Bootstrap

The bootstrap resamples the data (or clusters) and refits the model at each iteration. The procedure:

1. For $b = 1, \ldots, B$:
   - Resample the data (with replacement) to create dataset $D^{(b)}$.
   - Refit the model on $D^{(b)}$ to obtain $\hat{\beta}^{(b)}$.
   - Compute $g^{(b)} = g(\hat{\beta}^{(b)})$.
2. The standard error is the empirical standard deviation of $\{g^{(b)}\}$.

The bootstrap variance estimator is:

$$\widehat{\text{Var}}_{\text{boot}}[g(\hat{\beta})] = \frac{1}{B-1} \sum_{b=1}^{B} \left(g^{(b)} - \bar{g}\right)^2$$

What this means in code: `dydx(..., vce="bootstrap", n_boot=200)` resamples the data 200 times, fits the model from scratch each time, computes the marginal effect on each fitted model, and reports the empirical standard deviation. This is the most computationally intensive method.

### Strengths
- **Robust to non-normality:** By resampling and refitting, the bootstrap approximates the true sampling distribution without assuming normality. It is the most reliable method for small samples.
- **Cluster/block resampling:** Supports `boot_method="cluster"` (with a `cluster=` ID array) to resample clusters instead of observations, preserving within-cluster correlation structure. `boot_method="block"` does moving-block resampling for time series. This is essential for correct inference with clustered or serially correlated data.
- **Double robustness:** Even if the model is slightly misspecified, the bootstrap standard error reflects the empirical sampling variability of the estimator.

### Weaknesses
- **Computationally expensive:** Refitting the model $B$ times dominates the cost. For complex models (mixed effects, large datasets), this can be prohibitive.
- **Convergence failures:** Some bootstrap samples may cause the model to fail to converge (separation in logit, rank deficiency). `smmargins` handles these gracefully but they reduce effective $B$.
- **Asymptotically less efficient:** Bootstrap SEs converge to the true SE at rate $O(B^{-1/2})$, while delta method converges at $O(n^{-1})$. For very large $n$, delta method is more precise if its assumptions hold.

> ⚠️ **Trade-off:** Bootstrap is the gold standard for robustness but comes at a steep computational cost. Use it when (1) the sample is small, (2) the data is clustered and you need cluster-aware inference, or (3) you suspect the sampling distribution is far from normal. For large, well-behaved samples, the delta method is faster and equally accurate.

---

## The Decision Framework

| Method | Speed | Robust to Non-Normality | Cluster-Aware | Best For |
|---|---|---|---|---|
| Delta | Fastest | No | Only via `cov_type` | Large samples, well-behaved models |
| KR Simulation | Fast | Moderately | No | Moderate nonlinearity, medium samples |
| Bootstrap | Slowest | Yes | Yes (explicit) | Small samples, clustered data, complex models |

> ⚠️ **Trade-off:** There is no universally "best" method. The delta method is the right default for most applied work. KR simulation improves on it for nonlinear statistics at modest cost. Bootstrap is the safety net when assumptions fail. The cost of bootstrap is time; the cost of delta method in the wrong setting is invalid inference.

---

## Related Documentation

- **Tutorial:** {doc}`Inference Method Comparison on Clustered Data </tutorials/inference>` — side-by-side delta, KR, and bootstrap on a real dataset.
- **Reference:** {doc}`Margins.dydx() </api>` for the `vce` parameter (`"delta"`, `"simulation"`, `"bootstrap"`), along with `n_sims`, `n_boot`, `boot_method`, and `cluster`.
