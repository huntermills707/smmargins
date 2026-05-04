# Prediction Scales and Link Functions

## The Same Quantity, Different Lenses

If you come from Stata, you know that `predict, pr` gives predicted probabilities, `predict, xb` gives linear predictions, and `margins` defaults to the response scale. What Stata does not make explicit is that these are all the same underlying model evaluated through different lenses — and choosing the wrong lens for your estimand can produce nonsensical results.

If you use `statsmodels.get_margeff`, you get marginal effects on the response (probability) scale with no option to change. `smmargins` exposes the full set of scales and enforces valid scale-model combinations.

---

## The Link Function Chain

Every GLM consists of three layers:

$$\underbrace{y}_{\text{response}} \leftarrow \underbrace{\mu = f(\eta)}_{\text{mean function}} \leftarrow \underbrace{\eta = X\beta}_{\text{linear predictor}}$$

The link function $g(\mu) = \eta$ connects the linear predictor to the mean. Its inverse $f = g^{-1}$ maps back to the response. A "scale" in `smmargins` is simply the point in this chain at which you choose to report.

### The Response Scale

The default scale is `"response"` — the mean function applied to the linear predictor:

$$\hat{\mu}_i = f(x_i'\hat{\beta})$$

For logit, this is the predicted probability $\Lambda(x_i'\hat{\beta})$. For Poisson, this is the predicted count $\exp(x_i'\hat{\beta})$.

What this means in code: `predict(scale="response")` evaluates `family.link.inverse(eta)` for each observation and returns the mean of these values. For a counterfactual prediction, it evaluates the same function at the modified covariate values.

### The Linear Scale

The `"linear"` scale reports the linear predictor directly:

$$\hat{\eta}_i = x_i'\hat{\beta}$$

This is Stata's `predict, xb`. It is on the scale of the link function, not the response.

What this means in code: `predict(scale="linear")` skips the inverse link entirely and returns $\hat{\eta}$. The delta-method Jacobian is simply $X/n$, and the standard errors come from the linear model's covariance structure directly.

> ⚠️ **Trade-off:** Linear-scale predictions are easier to interpret in terms of "one-unit changes in $x$," but they live on the log-odds scale (for logit) or the log-count scale (for Poisson), which is not where policy decisions are made. Response-scale predictions are on the probability or count scale, which is more intuitive but nonlinear in $\beta$.

---

## Model-Specific Scales

### Odds Ratios: `"or"`

For logit models only, the `"or"` scale reports exponentiated linear predictions:

$$\text{OR}_i = \exp(x_i'\hat{\beta})$$

The odds ratio compares the odds of success $(P/(1-P))$ between two covariate profiles. A coefficient $\beta_j$ of 0.5 means a one-unit increase in $x_j$ multiplies the odds by $e^{0.5} \approx 1.65$.

What this means in code: `predict(scale="or")` computes $\exp(\hat{\eta}_i)$ for each observation. The delta-method Jacobian uses the chain rule: $\partial \exp(\eta)/\partial \beta = \exp(\eta) \cdot x_i'$.

> ⚠️ **Trade-off:** Odds ratios are convenient for reporting (no probability-scale nonlinearity), but they are not collapsible over covariates. The average odds ratio is not the odds ratio at average covariates. Use with care in heterogeneous populations.

### Incidence Rates: `"ir"`

For Poisson and negative binomial models, the `"ir"` scale is the predicted rate:

$$\hat{\lambda}_i = \exp(x_i'\hat{\beta})$$

This is mathematically identical to `"response"` for count models (since the mean of a Poisson is its rate), but the naming convention signals that the quantity is an incidence rate per unit exposure.

What this means in code: `predict(scale="ir")` on a Poisson model returns the same numeric values as `scale="response"`, but the labeling and validation logic ensure the scale-model pairing is documented.

### Exponential and Log Scales

The `"exp"` scale is $\exp(\eta)$ — generic and available for any model where it is mathematically defined. The `"log"` scale is $\log(\mu)$, which maps a response-scale prediction back to the linear predictor.

$$\text{"exp"}: \hat{y}_i = \exp(x_i'\hat{\beta})$$

$$\text{"log"}: \hat{y}_i = \log(f(x_i'\hat{\beta}))$$

What this means in code: `scale="exp"` on an OLS model gives exponentiated linear predictions (useful for log-linear models fit with OLS). `scale="log"` on a Poisson model gives $\log(\hat{\lambda})$, which is the linear predictor — effectively undoing the inverse link.

---

## Invalid Combinations

Not all scales make sense for all models. `smmargins` validates scale-model pairings and raises a clear error for invalid combinations:

| Scale | Logit | Probit | Poisson | OLS | MNLogit |
|---|---|---|---|---|---|
| `"response"` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `"linear"` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| `"pr"` | $\checkmark$ | $\checkmark$ | $\times$ | $\times$ | $\times$ |
| `"ir"` | $\times$ | $\times$ | $\checkmark$ | $\times$ | $\times$ |
| `"or"` | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| `"exp"` | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |

For example, `predict(scale="or")` on a Poisson model raises:

```
ValueError: scale="or" (odds ratio) is only valid for logit models.
```

What this means in code: the validation happens before any computation, so you get a clear message rather than a silently wrong result.

---

## Why Scale Matters for Marginal Effects

The `scale` parameter in `dydx()` controls the scale of the *prediction* that the derivative is taken with respect to, not the scale of the derivative itself. For a logit model:

- `dydx(scale="response")`: $\partial \Lambda(\eta) / \partial x_j$ — change in probability per unit change in $x_j$
- `dydx(scale="linear")`: $\partial \eta / \partial x_j = \beta_j$ — change in log-odds per unit change in $x_j$

The relationship between them is:

$$\frac{\partial \Lambda(\eta)}{\partial x_j} = \Lambda(\eta)(1 - \Lambda(\eta)) \cdot \beta_j$$

What this means in code: `scale="response"` marginal effects vary with $x$ (they are largest near $\Lambda = 0.5$ and approach zero in the tails), while `scale="linear"` marginal effects are constant. The choice is substantive, not cosmetic.

---

## Related Documentation

- **Tutorial:** {doc}`Comparing Scales in Logit and Poisson Models </tutorials/adjusted_predictions>` — visualize how marginal effects differ across scales.
- **Reference:** {doc}`Margins.predict() </api>` for the complete list of valid scale-model combinations.
