# Multiple Comparison Adjustments

## The Problem: Familywise Error Inflation

If you come from Stata, you may have noticed that `margins, dydx(x1 x2 x3)` reports a separate confidence interval for each marginal effect, but these intervals are not jointly valid. If each interval has 95% pointwise coverage, the probability that *all* intervals simultaneously cover their true values is substantially less than 95%. This is the multiple comparisons problem.

When you compute marginal effects for $m$ covariates, you are conducting $m$ simultaneous hypothesis tests. Under the null, the probability of at least one false rejection is:

$$P(\text{at least one Type I error}) = 1 - (1 - \alpha)^m$$

For $\alpha = 0.05$ and $m = 10$, this familywise error rate (FWER) is approximately $1 - 0.95^{10} = 0.40$. You have a 40% chance of falsely rejecting at least one null — far above the nominal 5% level.

What this means in code: if you call `dydx(["x1", "x2", ..., "x10"])` with the default `ci_method="pointwise"`, each marginal effect's confidence interval is valid individually, but you should not interpret "all 10 intervals exclude zero" as strong evidence that all 10 effects are nonzero.

---

## Pointwise Inference: The Default

The `"pointwise"` option computes standard marginal effect confidence intervals with no adjustment:

$$\text{CI}_j = \hat{\theta}_j \pm z_{1 - \alpha/2} \cdot \text{SE}(\hat{\theta}_j)$$

Coverage is $1 - \alpha$ for each individual interval, but the joint coverage over all $m$ intervals is lower.

> ⚠️ **Trade-off:** Pointwise intervals are the right choice when you care about each marginal effect separately and are not making joint claims. They are narrower than adjusted intervals and have higher statistical power for individual comparisons. They are the wrong choice when your research claim depends on the joint behavior of multiple marginal effects.

---

## Bonferroni Correction: Uniform Inflation

The `"bonferroni"` adjustment inflates every interval uniformly to control the familywise error rate:

$$\text{CI}_j^{\text{Bonferroni}} = \hat{\theta}_j \pm z_{1 - \alpha/(2m)} \cdot \text{SE}(\hat{\theta}_j)$$

The critical value changes from $z_{0.975} = 1.96$ to $z_{1 - 0.025/m}$. For $m = 10$, this is $z_{0.9975} = 2.81$, widening each interval by a factor of approximately 1.43.

What this means in code: `dydx(..., ci_method="bonferroni")` divides the significance level $\alpha$ by the number of comparisons $m$ and recomputes the critical value. The standard error itself is unchanged; only the multiplier changes.

### Strengths
- **Guarantees FWER control:** By Boole's inequality, $P(\text{at least one rejection}) \leq \sum_{j=1}^{m} P(\text{reject } H_{0j}) = m \cdot (\alpha/m) = \alpha$. This holds regardless of the correlation structure among the test statistics.
- **Simple and intuitive:** Divide $\alpha$ by $m$. No simulation, no correlation modeling.

### Weaknesses
- **Conservative:** When the test statistics are positively correlated (common for marginal effects of related covariates), Bonferroni is strictly conservative — the true FWER is less than $\alpha$, meaning intervals are wider than necessary.
- **Power loss:** The uniform inflation can make intervals so wide that genuinely nonzero effects become statistically insignificant.

> ⚠️ **Trade-off:** Bonferroni is the safest choice when you need a guaranteed FWER bound and know nothing about the correlation among your marginal effects. It is unnecessarily conservative when the statistics are strongly correlated, which is the typical case for marginal effects.

---

## Sidak Correction: Slightly Narrower

The `"sidak"` adjustment assumes independence among the test statistics and computes:

$$\alpha_{\text{adjusted}} = 1 - (1 - \alpha)^{1/m}$$

The critical value is $z_{1 - \alpha_{\text{adjusted}}/2}$. For $m = 10$ and $\alpha = 0.05$, this gives $\alpha_{\text{adjusted}} = 1 - 0.95^{0.1} = 0.0051$, compared to Bonferroni's $\alpha/m = 0.005$. The difference is small but Sidak intervals are always slightly narrower.

What this means in code: `dydx(..., ci_method="sidak")` replaces the Bonferroni divisor with the Sidak exponent. The standard error is unchanged.

The Sidak-adjusted confidence interval is:

$$\text{CI}_j^{\text{Sidak}} = \hat{\theta}_j \pm z_{1 - (1 - (1-\alpha)^{1/m})/2} \cdot \text{SE}(\hat{\theta}_j)$$

> ⚠️ **Trade-off:** Sidak is slightly more powerful than Bonferroni but assumes independence. For marginal effects of correlated covariates, this assumption is violated, so Sidak does not guarantee FWER control. It is a reasonable heuristic but not a rigorous bound.

---

## Sup-T Simulation: Exploiting Correlation

The `"sup-t"` method takes a fundamentally different approach. Instead of adjusting the critical value analytically, it simulates the joint distribution of the test statistics and computes data-specific critical values that account for their empirical correlation structure.

The procedure:

1. Compute the estimated covariance matrix $\hat{\Sigma}$ of the $m$ marginal effects (from the delta-method sandwich).
2. Draw $B$ vectors $Z^{(b)} \sim N(0, \hat{\Sigma})$.
3. For each draw, compute $\max_j |Z_j^{(b)} / \text{SE}_j|$.
4. The critical value $c_{1-\alpha}$ is the $(1 - \alpha)$ quantile of this empirical distribution.

The confidence intervals are:

$$\text{CI}_j^{\text{sup-t}} = \hat{\theta}_j \pm c_{1-\alpha} \cdot \text{SE}(\hat{\theta}_j)$$

What this means in code: `dydx(..., ci_method="sup-t", vce="simulation")` (or `vce="bootstrap"`) draws from the joint distribution of all marginal effect estimates, finds the critical value that controls the probability that *any* interval fails to cover, and uses that critical value for all intervals. The critical value depends on the correlation structure — highly correlated statistics yield smaller $c_{1-\alpha}$ than independent ones. `sup-t` requires draws, so it cannot compose with the default `vce="delta"`.

### Strengths
- **Accounts for correlation:** By simulating from the joint distribution, sup-t automatically adapts to the correlation among marginal effects. When statistics are highly correlated, $c_{1-\alpha}$ is close to the pointwise critical value. When they are independent, it approaches the Sidak value.
- **Narrowest valid intervals:** Among methods that control FWER, sup-t produces the tightest intervals because it uses the maximal information from $\hat{\Sigma}$.
- **Exact in finite samples under normality:** If the marginal effects are exactly normal, sup-t controls FWER exactly.

### Weaknesses
- **Requires $\hat{\Sigma}$:** The method depends on an accurate estimate of the joint covariance matrix. If the delta-method covariance is poor (small sample, misspecified model), sup-t inherits that inaccuracy.
- **Simulation variability:** The critical value depends on the number of simulation draws $B$. Large $B$ (e.g., 10,000) is needed for stability.
- **Not valid under non-normality:** Like the delta method, sup-t assumes normality of the test statistic distribution.

> ⚠️ **Trade-off:** Sup-t is the most powerful FWER-controlling method when the marginal effects are approximately normal and their covariance is well-estimated. It is the recommended choice for joint inference on marginal effects. However, it does not solve non-normality — for small samples, bootstrap joint intervals may be preferred.

---

## The Coverage Picture

For $m = 5$ marginal effects with moderate positive correlation ($\rho = 0.3$):

| Method | Nominal | True Joint Coverage | Interval Width |
|---|---|---|---|
| Pointwise | 95% | ~83% | Narrowest |
| Sidak | 95% | ~95% (if independent) | Moderate |
| Bonferroni | 95% | > 95% | Wide |
| Sup-t | 95% | ~95% | Narrowest valid |

> ⚠️ **Trade-off:** The choice among these methods is a trade-off between power (interval width) and validity guarantee. Pointwise is most powerful but invalid for joint claims. Bonferroni is least powerful but universally valid. Sup-t occupies the sweet spot for most applications.

---

## Related Documentation

- **Tutorial:** {doc}`Joint Confidence Intervals for Multiple Marginal Effects </tutorials/inference>` — visualize how sup-t produces tighter joint intervals than Bonferroni.
- **Reference:** {doc}`Margins.dydx() </api>` for the `ci_method` and `ci_alpha` parameters and supported adjustment methods (`"pointwise"`, `"bonferroni"`, `"sidak"`, `"sup-t"`).
