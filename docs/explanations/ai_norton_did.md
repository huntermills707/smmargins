# The Ai & Norton Difference-in-Differences Problem

## The Trap: Reading DiD Off the Interaction Coefficient

If you come from a linear regression background, you are accustomed to interpreting interaction coefficients as difference-in-differences (DiD) estimators. In OLS, the coefficient on $D \times T$ *is* the DiD:

$$Y = \beta_0 + \beta_1 D + \beta_2 T + \beta_3 (D \times T) + \varepsilon$$

$$\text{DiD}_{\text{OLS}} = \beta_3$$

This is a remarkable convenience: a single coefficient gives the causal estimand of interest. Many applied researchers carry this intuition into nonlinear models — and it fails catastrophically.

Ai and Norton (2003) showed that in a logit model, the interaction coefficient $\beta_3$ is on the log-odds scale, not the probability scale. It is not the DiD. It is not even proportional to the DiD. The DiD on the probability scale is a nonlinear function of all coefficients and all covariates, and it varies across observations.

---

## The Numbers: A Concrete Example

Consider a simple DiD setup with binary treatment $D$ and binary post-period $T$. Fit the same data with OLS and logit.

**OLS results:**

The interaction coefficient is $\hat{\beta}_3 = -0.676$. This is the DiD on the probability scale:

$$\text{DiD}_{\text{OLS}} = E[Y \mid D=1, T=1] - E[Y \mid D=1, T=0] - E[Y \mid D=0, T=1] + E[Y \mid D=0, T=0] = -0.676$$

**Logit results:**

The interaction coefficient is also $\hat{\beta}_3 = -0.676$, but this is on the log-odds scale. The actual DiD on the probability scale is $-0.147$ — a factor of 4.6 smaller in magnitude.

What this means in code: if you fit `logit(y ~ D * T)` and report $\hat{\beta}_3$ as your DiD estimate, you are reporting the wrong number by a large factor. The correct DiD requires computing predicted probabilities at all four $(D, T)$ combinations and differencing them:

$$\text{DiD}_{\text{logit}} = P(Y=1 \mid D=1, T=1) - P(Y=1 \mid D=1, T=0) - P(Y=1 \mid D=0, T=1) + P(Y=1 \mid D=0, T=0)$$

where each probability is $\Lambda(\beta_0 + \beta_1 d + \beta_2 t + \beta_3 (d \cdot t))$ evaluated at the relevant $(d, t)$ pair.

---

## Why the Interaction Coefficient Fails

In a logit model, the marginal effect of the interaction is not the interaction of the marginal effects. Formally:

$$\frac{\partial^2 \Lambda(\eta)}{\partial D \partial T} \neq \beta_3 \cdot \Lambda(\eta)(1 - \Lambda(\eta))$$

The left-hand side is the cross-partial derivative of the predicted probability — the quantity of interest for DiD. The right-hand side is what you might naively compute from the coefficient. They differ because the logit mean function $\Lambda(\eta)$ is nonlinear, so the effect of $D$ depends on $T$ (and vice versa) in a way that the log-odds interaction coefficient does not capture.

Expanding the cross-derivative:

$$\frac{\partial^2 \Lambda(\eta)}{\partial D \partial T} = \Lambda(\eta)(1 - \Lambda(\eta)) \cdot \beta_3 \cdot (1 - 2\Lambda(\eta)) \cdot \beta_3 + \Lambda(\eta)(1 - \Lambda(\eta)) \cdot \text{(other terms)}$$

The expression includes $\beta_3$ multiplied by $\Lambda(\eta)(1 - \Lambda(\eta))(1 - 2\Lambda(\eta))$, which is a function of all covariates through $\eta$. There is no constant "DiD effect" — it varies across the covariate distribution.

---

## The Correct Approach: Predict and Difference

The solution is conceptually simple: compute predicted probabilities at each of the four $(D, T)$ combinations, then difference them exactly as the DiD definition requires.

$$\text{DiD}_i = \Lambda(\eta_i \mid D_i=1, T_i=1) - \Lambda(\eta_i \mid D_i=1, T_i=0) - \Lambda(\eta_i \mid D_i=0, T_i=1) + \Lambda(\eta_i \mid D_i=0, T_i=0)$$

The *average* DiD is then:

$$\overline{\text{DiD}} = \frac{1}{n} \sum_{i=1}^{n} \text{DiD}_i$$

What this means in code: `M.did()` in `smmargins` performs exactly this computation. It sets up four counterfactual predictions (treatment and control, pre and post), differences them observation by observation, and averages. The standard error is computed via the delta method on the full four-prediction contrast.

> ⚠️ **Trade-off:** Computing the DiD via counterfactual predictions requires four prediction passes over the data (one for each $(D, T)$ combination) rather than reading a single coefficient. This is more computation but gives the correct answer. The alternative — reading $\beta_3$ as the DiD — is fast and wrong.

---

## Generalizing Beyond Binary $D$ and $T$

The problem is not limited to binary variables. For any two covariates $x_j$ and $x_k$ in a nonlinear model, the cross-partial derivative $\partial^2 f(\eta) / \partial x_j \partial x_k$ is not given by any single coefficient or simple combination thereof. It requires evaluating the second derivative of the mean function at the covariate values and combining it with the coefficient structure.

For a general nonlinear mean function $f$:

$$\frac{\partial^2 f(\eta)}{\partial x_j \partial x_k} = f''(\eta) \cdot \beta_j \cdot \beta_k$$

What this means in code: `smmargins` computes cross-partials by evaluating $f''(\eta)$ (the second derivative of the mean function) at each observation's linear predictor, then multiplying by the relevant coefficients. For logit, $f''(\eta) = \Lambda(\eta)(1 - \Lambda(\eta))(1 - 2\Lambda(\eta))$.

---

## Related Documentation

- **Tutorial:** {doc}`Correct DiD Estimation in Logit Models </tutorials/did>` — replicate the Ai & Norton example with `smmargins`.
- **Reference:** {doc}`did() </api>` for the exact API specification and counterfactual construction.
