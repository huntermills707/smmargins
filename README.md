# `smmargins` — Stata-style `margins` for StatsModels

A small module that fills in the marginal-effects gaps in StatsModels:
adjusted predictions and marginal effects at user-specified covariate
profiles, with delta-method standard errors, for *any* fitted model that
exposes `params`, `cov_params()`, and a `predict(params, exog)` method.

## Installation

```bash
pip install smmargins
```

Requires Python ≥3.9. Dependencies (`numpy`, `pandas`, `statsmodels`, `scipy`, `patsy`, `matplotlib>=3.5`) are installed automatically.

## Math

The delta method: for a parameter vector $\hat\beta$ with estimated
covariance $\widehat V$, and a (possibly vector-valued) statistic
$g(\beta)$,

$$
\widehat{\mathrm{Var}}\bigl[g(\hat\beta)\bigr]
\;\approx\; G\,\widehat V\,G^\top,
\qquad
G = \left.\frac{\partial g}{\partial \beta}\right|_{\hat\beta}.
$$

This is exactly the approach documented in
[Stata's margins-delta-method FAQ](https://www.stata.com/support/faqs/statistics/compute-standard-errors-with-margins/).
The cases in [Richard Williams' *Margins01* notes](https://academicweb.nd.edu/~rwilliam/stats/Margins01.pdf)
all fit the $g(\beta)$ schema:

| Statistic | $g(\beta)$                                                              |
|-----------|-------------------------------------------------------------------------|
| AAP       | $\frac{1}{n}\sum_i f(x_i^\top\beta)$                                    |
| APM       | $f(\bar x^\top\beta)$                                                   |
| APR       | $\frac{1}{n}\sum_i f(x_i^\top\beta)$ with some $x$ fixed                |
| AME       | $\frac{1}{n}\sum_i \partial f(x_i^\top\beta)/\partial x_{ij}$           |
| MEM       | $\partial f(\bar x^\top\beta)/\partial x_j$                             |
| MER       | AME / AAP with some $x$ held at representative values                   |
| Contrast  | $E[f(\cdot)\mid x_j=\text{level}] - E[f(\cdot)\mid x_j=\text{ref}]$     |

where $f$ is the response-scale map (identity for OLS, inverse link for
GLMs, etc.). Every statistic above is a linear combination of
mean-predictions $\frac{1}{n}\sum_i f(x_i^\top\beta)$ on different
design matrices, so the single Jacobian primitive needed is

$$
\frac{\partial}{\partial\beta}\,\frac{1}{n}\sum_i f(x_i^\top\beta)
\;=\; \frac{1}{n}\sum_i f'(x_i^\top\beta)\,x_i .
$$

The module computes $f'$ analytically when the model exposes it (any
GLM via `family.link.inverse_deriv` — Logit, Probit, Poisson, NegBin,
Gamma, Gaussian — plus OLS/WLS/GLS via the identity link), and falls
back to central finite differences when it doesn't. See design choice
4 below.

## Why patsy

When we want the marginal effect of `x1` and the formula is
`y ~ x1 + I(x1**2) + x1:x2 + C(group)`, we can't just nudge one column
of the design matrix — `x1` enters three columns. What we *can* nudge
is the `x1` column of the **original data frame**, and then ask patsy to
rebuild the design matrix using the stored `DesignInfo`:

```python
patsy.dmatrix(design_info, perturbed_frame, return_type="matrix")
```

That preserves polynomial terms, interactions, splines (`bs(x, df=4)`),
and categorical contrasts automatically. It's also the right abstraction
for "hold `age=45`" or "set `group='b'`" — you mutate the data frame,
not the design matrix.

## API

```python
from smmargins import Margins

M = Margins(fit)                 # fit is a statsmodels results object

# --- adjusted predictions ---
M.predict()                                       # AAP             (at="overall")
M.predict(at="mean")                              # APM             (at="mean")
M.predict(at="median")                            # APM at medians
M.predict(at="zero")                              # APM at zero
M.predict(atexog={"x1": [0, 1, 2]})               # APR (others at observed values)
M.predict(atexog={"x1": [0, 1, 2]}, at="mean")    # APR holding others at means

# --- marginal effects ---
M.dydx("x1")                                      # AME (continuous auto-detected)
M.dydx(["x1", "x2"])                              # multiple variables stacked jointly
M.dydx("*")                                       # all RHS columns
M.dydx("x1", at="mean")                           # MEM (factors at proportions)
M.dydx("x1", at="median")                         # ME at medians
M.dydx("x1", atexog={"x2": [0, 1]})               # MER
M.dydx("kids", count=True)                        # unit increment x -> x+1 (for integers)
M.dydx("group")                                   # discrete contrasts vs reference level
M.dydx("group", reference="b")                    # discrete contrasts vs "b"

# --- elasticities (continuous only) ---
M.dydx("x1", method="eyex")                       # full elasticity:    ey/ex = (dy/dx)·x/y
M.dydx("x1", method="dyex")                       # semi-elasticity:    dy/ex = (dy/dx)·x
M.dydx("x1", method="eydx")                       # semi-elasticity:    ey/dx = (dy/dx)/y
# Combine with at=/atexog= as usual; raises on discrete variables.
# On multi-outcome models returns one elasticity per outcome class.

# --- 0.4: scales, weights, subgroups, joint tests ---
M.dydx("x1", scale="linear")                      # AME on Xβ (Stata's xb)
M.dydx("x1", over="region")                       # subgroup AMEs (joint vcov preserved)
Margins(fit, weights=w).dydx("x1")                # observation weights (sampling/frequency)
res = M.dydx(["x1", "x2"]); res.wald()            # joint Wald test of margins == 0
res = M.dydx("group"); res.pairwise(by="group")   # all level-vs-level comparisons

# --- factor handling at evaluation points ---
M.predict(at="mean")                              # factor dummies at observed proportions (Stata default)
M.predict(at="mean", factor_stat="mode")          # factors held at modal level
M.predict(at="zero", factor_stat="mode")          # numerics at 0, factors at mode
```

The `at=` parameter mirrors statsmodels' `get_margeff(at=...)`:
`"overall"` (default), `"mean"`, `"median"`, `"zero"`. The `atexog=` dict
fixes specific variables at specific values (Stata's `margins, at(...)`);
unlike statsmodels' `atexog`, it's keyed by **variable name** rather than
column index.

### Formula vs. Raw Exog Mode

`Margins` supports models fit without formulas (e.g. `sm.OLS(y, X).fit()`).
In this "raw mode", variable names are taken from `model.exog_names` (often
`x1, x2, ...` or `const`).

**Important Limitation:** In raw mode, `Margins` does not know about
relationships between columns in the design matrix. If you manually
included an interaction column (e.g. `X["x1_x2"] = X["x1"] * X["x2"]`),
perturbing `x1` for a marginal effect will **not** automatically update
`x1_x2`. This will lead to incorrect marginal effects for that variable.
If your model has interactions or transformations, you should fit it using
a formula to ensure `Margins` can correctly rebuild the design matrix.

Each call returns a `MarginsResult` with `.estimate`, `.se`, `.vcov`,
`.ci_lower`, `.ci_upper`, `.pvalue`, plus `.summary()` returning a
DataFrame. Use `use_t=True` on the `Margins` constructor if you want
t-distribution inference (reads `results.df_resid`).

```python
M = Margins(fit, analytic=True)   # default: analytic outer Jacobian when possible
M = Margins(fit, analytic=False)  # force central finite differences everywhere
```

## Design choices worth knowing

1. **`at="mean"` averages the design matrix by default.** This matches
   Stata's `margins, atmeans` and statsmodels' `get_margeff(at='mean')`:
   numeric columns become their sample mean, factor dummies become their
   observed proportions (a "person" who is 0.33 female). The test suite
   verifies a match to `statsmodels.get_margeff(at='mean')` to machine
   precision. Pass `factor_stat="mode"` for the modal-individual variant
   (numerics at their mean, factors at their modal level) — gives a
   "typical individual" rather than a fictional fractional one.
   `factor_stat="zero"` puts factors at their reference level instead.
   Williams (Margins01) notes that atmeans-on-dummies is usually a worse
   choice than AME regardless of which convention you pick, so prefer
   AME for factor-heavy models.

2. **`dydx` is response-scale by default.** For GLMs this means
   $\partial\hat\mu / \partial x$, not $\partial\hat\eta / \partial x$.
   This matches Stata's default and is almost always what's wanted.

3. **Discrete vs continuous is auto-detected**, based on dtype and number
   of unique values. Override with `discrete=True`/`False` if needed.

4. **Outer Jacobian is analytic when possible, FD otherwise.** Every
   statistic in this module is a linear combination of mean-predictions
   $\frac{1}{n}\sum_i f(x_i^\top\beta)$, whose gradient is
   $\frac{1}{n}\sum_i f'(x_i^\top\beta)\,x_i$. We obtain $f'$
   analytically for any GLM via `family.link.inverse_deriv` (Logit,
   Probit, Poisson, NegBin, Gamma, Gaussian) and via the identity link
   for OLS/WLS/GLS. MNLogit uses an analytic softmax-derivative.
   For continuous AME we still perturb the **data
   column** with central differences — patsy doesn't expose symbolic
   derivatives of basis transforms (`I(x**2)`, `bs(...)`, `cr(...)`,
   interactions) — but that's a single 2-rebuild step, independent of
   $p$. When an analytic path is missing, an offset/exposure is
   present (so $\eta\neq X\beta$), or the model is neither GLM nor a
   linear-regression model, the entire Jacobian falls back to central
   finite differences with $h = \epsilon^{1/3}\cdot\max(|x|, 1)$ — the
   truncation-vs-rounding sweet spot. Stata's `margins` uses FD throughout;
   our analytic path removes $p$ forward `predict` calls per statistic
   without changing any answer to within FD tolerance. Pass `analytic=False`
   to force FD on every model.

5. **Everything goes through `model.predict(params, exog)`.** That way
   the inverse link, offsets, exposures, and any other model-specific
   wrinkles handled by StatsModels come along for the ride for free.

## Verification

The test suite checks the module against:

- An **OLS with polynomial + interaction**, where the analytic AME is
  $\beta_1 + 2\beta_2\,\overline{x_1} + \beta_4\,\overline{x_2}$ and its
  SE is $\sqrt{g^\top V g}$. Matches to machine precision.
- **Logit AME**, compared against `statsmodels.get_margeff(at='overall')`.
  Matches to $10^{-5}$ on estimate and SE.
- **Logit MEM**, compared against `get_margeff(at='mean')` on a formula
  with a `C(grp)` factor. Matches to $10^{-5}$ on estimate and SE under
  the default `factor_stat="mean"`. A separate hand-computed check covers
  the `factor_stat="mode"` path.
- **Poisson AAP** with hand-derived $g$ gradient. Matches exactly; and
  sanity-checks that AAP equals the sample mean for canonical-link
  Poisson.
- **Analytic-vs-FD parity matrix.** Every public API call (AAP, APM,
  APR, AME, MEM, MER — continuous and discrete) is run twice on each
  of OLS+poly, Logit+`C(grp)`, Poisson, and MNLogit, with `analytic=True`
  and `analytic=False`, and the two paths must agree on both estimate and
  SE. This guards against the analytic chain-rule formula drifting
  from the FD answer that the other tests pin to Stata /
  `get_margeff`.
- **Elasticities (`eyex`, `dyex`, `eydx`)** matched against
  `statsmodels.get_margeff(method=...)` for both AME-style (`at='overall'`)
  and MEM-style (`at='mean'`), on Logit and Poisson, to $10^{-5}$ on
  estimates and $10^{-3}$ on SEs. Plus a hand check that Poisson MEM
  $\mathrm{ey/ex}(x_1)=\beta_1\,\overline{x_1}$ for the canonical-link
  case. Elasticities on a discrete variable raise.

### External parity vs R `marginaleffects`

`smmargins` is also pinned to the de-facto cross-language reference,
[`marginaleffects`](https://marginaleffects.com), via a small set of
checked-in R reference outputs. The R script is `tests/comparison/generate_r.R`
(requires `marginaleffects`, `readr`, and `sandwich`) and the matching
pytest assertions live in `tests/comparison/test_r.py`. Six cases are
covered:

| Case                                       | Estimate tol | SE tol           |
| ------------------------------------------ | ------------ | ---------------- |
| Logit AME with HC3                         | 1e-6         | 5e-5             |
| Poisson AME with HC3                       | 1e-6         | 5e-4             |
| OLS pairs bootstrap (n_boot=1000)          | 1e-10        | Monte-Carlo (5σ) |
| OLS with HC1, polynomial + interaction     | 1e-6         | 1e-6             |
| Logit cluster-robust AME                   | 1e-5         | 1e-4             |
| Logit AME at representative `x2` values    | 1e-5         | 1e-4             |

These tests are auto-skipped if the CSV reference files are missing, so
the rest of the suite still runs without a working R install. Regenerate
the references with:

```bash
cd tests/comparison && Rscript generate_r.R
```

## Multi-outcome models

`smmargins` supports `statsmodels.MNLogit` (multinomial logit) and
`statsmodels.miscmodels.ordinal_model.OrderedModel` (ordered logit/probit).
For these models every statistic returns one value per outcome class — `K`
values in place of the usual scalar — with full joint covariance across
both rows and classes.

```python
import statsmodels.api as sm
from smmargins import Margins

fit = sm.MNLogit(y, X).fit()
M = Margins(fit)

M.n_outcomes        # K — detected automatically
M.outcome_labels    # class labels from the fit (or numeric fallback)

res = M.predict()           # AAP per class; K rows
res.summary()               # long-format DataFrame with `outcome` column
res.estimate.sum()          # 1.0 — class probabilities at AAP sum to one

ame = M.dydx("x1")          # AME of x1 on each class probability; K rows
ame = M.dydx(["x1", "x2"])  # 2*K rows; full joint vcov across vars × classes
```

Subset to specific outcomes via the `outcome=` kwarg, or post-hoc with
`MarginsResult.outcome(k)` (which preserves the corresponding sub-block of
the joint covariance):

```python
M.predict(outcome=1)               # only class 1
M.predict(outcome=[0, 2])          # classes 0 and 2
M.dydx("x1", outcome="versicolor") # by label, if labeled

res = M.predict()
res.outcome(2)                     # slice an existing result
```

Single-outcome models silently ignore `outcome=`, so the same code paths
work uniformly across model classes.

**Implementation.** MNLogit uses an analytic softmax-derivative gradient
(no extra forward `predict` calls per parameter), exact to within FD
tolerance against `statsmodels.get_margeff` on AME/MEM. OrderedModel uses
central finite differences — the cumulative-link parameterization with the
log-difference threshold deltas is left to a future release. Elasticity
methods (`method="eyex" | "dyex" | "eydx"`) work on multi-outcome models
and emit a `RuntimeWarning` when any predicted class probability falls
below `1e-12` (where elasticities are numerically unstable).

## Inference options (0.3)

`smmargins` ships four tightly-related inference features behind one unified
`vce=` API:

### 1. Custom covariance (`cov_type=`, `vcov=`)

Override the default covariance matrix used for delta-method SEs:

```python
# Recompute with robust HC3
M = Margins(fit, cov_type="HC3")
M.dydx("x1")

# Cluster-robust (pass groups via cov_kwds)
M = Margins(fit, cov_type="cluster", cov_kwds={"groups": df["cluster_id"]})
M.dydx("x1")

# User-supplied matrix (mutually exclusive with cov_type)
M = Margins(fit, vcov=my_vcov_matrix)
```

Supported `cov_type` values include `"nonrobust"`, `"HC0"`–`"HC3"`,
`"cluster"`, and `"HAC"`. When neither is passed, the covariance falls
back to `results.cov_params()` (unchanged pre-0.3 behaviour).

### 2. Krinsky–Robb simulation (`vce="simulation"`)

Draw parameters from their sampling distribution and evaluate the margin
function for each draw. Fast — no model refits.

```python
M.dydx("x1", vce="simulation", n_sims=2000, sim_seed=42)
M.predict(atexog={"age": [25, 45, 65]}, vce="simulation", n_sims=5000)
```

The reported point estimate stays the analytic `g(β̂)`; draws are used
only for SEs and percentile CIs. Composes with `cov_type=` naturally
(e.g. `vce="simulation", cov_type="HC1"`).

### 3. Bootstrap (`vce="bootstrap"`)

Refit the model on each bootstrap sample. Supports pairs, cluster, and
moving-block resampling:

```python
# Pairs bootstrap (default)
M.dydx("x1", vce="bootstrap", n_boot=1000, boot_seed=42)

# Cluster bootstrap
M.dydx("x1", vce="bootstrap", boot_method="cluster",
       cluster=df["firm_id"], n_boot=1000)

# Block bootstrap (time series)
M.dydx("x1", vce="bootstrap", boot_method="block",
       block_size=10, n_boot=1000)
```

Optional `verbose=True` shows a progress bar; `n_jobs` enables parallel
refits via `joblib`.

### 4. Simultaneous confidence intervals (`ci_method=`)

Four methods work with any VCE:

```python
M.dydx(["x1", "x2", "x3"], ci_method="bonferroni")
M.dydx(["x1", "x2", "x3"], ci_method="sidak")
M.predict(atexog={"x1": [-1, 0, 1]}, vce="simulation",
          n_sims=2000, ci_method="sup-t")
```

- `"pointwise"` (default) — standard marginal CIs.
- `"bonferroni"` — conservative family-wise adjustment.
- `"sidak"` — slightly narrower than Bonferroni under independence.
- `"sup-t"` — simulation-based simultaneous critical value from the
  maximum standardized absolute deviation across the margin family.
  Requires `vce="simulation"` or `"bootstrap"`.

## Counterfactuals and plotting (0.5)

### Per-variable DSL (`values=`)

Hold specific variables at fixed values, grid points, or computed statistics without restating every column:

```python
# Fix x1 at 1, reduce everything else to its mean
M.predict(values={"x1": 1}, default_values="mean")

# Grid over income quartiles with everything else at observed values
M.predict(values={"income": ["p25", "p50", "p75"]})

# Scale a variable per row with a callable or Expr
from smmargins import Expr
M.predict(values={"income": lambda df: df["income"] * 1.10})
M.predict(values={"income": Expr("income * 1.10")})
```

Accepted value kinds per key: scalar, sequence, reducer string (`"mean"`, `"p25"`, ...), callable, or `Expr(...)`. `default_values="asobserved"` (default) leaves unspecified columns row-varying.

### `newdata=` escape hatch

Pass an arbitrary frame for out-of-sample evaluation:

```python
hypo = pd.DataFrame({"age": [25, 45, 65], "income": [30_000, 50_000, 80_000]})
M.predict(newdata=hypo)
M.dydx("age", newdata=hypo)
```

Mutually exclusive with `at`/`atexog`/`values`/`over`.

### Joint contrasts (`Margins.contrast`)

Compute two counterfactual arms and their difference with the correct joint SE:

```python
joint = M.contrast(a={"treat": 1}, b={"treat": 0})
joint.estimate   # [g_A, g_B, g_A - g_B]
joint.se         # SEs that include the A-B covariance
```

Also accepts `a_newdata=` / `b_newdata=` for frame-based arms.

### Plotting

```python
from smmargins import plot_predictions, plot_slopes, plot_comparisons

# Predicted probability vs age, with 95% CI band
plot_predictions(M, "age")

# AME of age vs income, faceted by sex
plot_slopes(M, "age", condition="income", by="sex")

# Treatment effect vs age
plot_comparisons(M, "treat", condition="age")
```

`condition` can be a variable name (auto-gridded) or a dict with explicit grid points. `ci=None` removes the band; `ci_method="bonferroni" | "sidak" | "sup-t"` passes through to the underlying margin call.

## Scales, weights, subgroups, joint tests (0.4)

Six features layered on top of the 0.3 inference machinery. Each
composes with `vce=`, `cov_type=`, and `ci_method=`.

### 1. Prediction scale (`scale=`)

Choose what `predict()` and `dydx()` operate on. Default is response
scale (`Λ(η)` for logit, `exp(η)` for Poisson, etc.).

```python
M.predict(scale="linear")              # linear predictor η = Xβ (Stata's xb)
M.dydx("x1", scale="or")               # AME on the odds-ratio scale (logit only)
M.dydx("x1", scale="exp")              # AME of exp(η) — generic across families
```

Built-in values: `"response"` (default), `"linear"`, `"pr"`, `"ir"`,
`"or"`, `"exp"`, `"log"`. Invalid combinations error clearly
(e.g. `scale="or"` on Poisson). Stata users: `predict(xb)` ↔
`scale="linear"`.

### 2. Custom transforms (`Transform`)

Pass a `Transform` instance to `scale=` for a user-defined λ. `dydx`
requires an analytic second derivative (`hess=`); `predict` only
needs `grad=`. **No autodiff** — analytic derivatives are a deliberate
package contract.

```python
from smmargins import Transform
import numpy as np

square = Transform(value=lambda e: e**2,
                   grad=lambda e: 2*e,
                   hess=lambda e: np.full_like(e, 2.0),
                   name="square")

M.dydx("x1", scale=square)
```

Built-ins (`Identity`, `Linear`, `Exp`, `Log`, `Logit`, `Probit`) are
exported from `smmargins.transforms`.

### 3. Elasticities × scales

Every elasticity method composes with every scale. No new methods —
just routing.

```python
M.dydx("x1", method="eyex", scale="linear")    # ey/ex on the linear predictor
M.dydx("x1", method="eyex", scale=square)      # ey/ex through a custom transform
```

### 4. Observation weights (`weights=`)

Pass weights at construction. `weight_type="sampling"` (default) or
`"frequency"`. WLS-fitted results respect their fit-time weights when
`weights=None`; explicit `weights=` overrides and warns.

```python
M = Margins(fit, weights=w)                                 # sampling
M = Margins(fit, weights=counts, weight_type="frequency")   # replication
M.dydx("x1", vce="bootstrap", n_boot=1000)                  # weighted resampling
```

### 5. Subgroup AMEs (`over=`)

Partition the sample by one or more columns and average within each
subgroup. The full joint covariance is preserved (not block-diagonal),
so cross-subgroup contrasts and Wald tests stay valid.

```python
M.dydx("x1", over="region")
M.dydx("x1", over=["region", "sex"])
M.predict(over="region")
```

### 6. Joint tests and pairwise comparisons (`wald()`, `pairwise()`)

```python
res = M.dydx(["x1", "x2", "x3"])
res.wald()                          # joint test that all margins == 0
res.wald(C=[[1, -1, 0]])            # H0: AME(x1) == AME(x2)
res.wald(C=np.eye(3)[:2], value=0)  # custom restriction

res = M.dydx("group")                          # discrete contrasts vs reference
pw = res.pairwise(by="group")                  # all level-vs-level pairs
pw_bonf = res.pairwise(by="group", ci_method="bonferroni")
```

`wald()` returns a `WaldResult(stat, df, pvalue, contrast_matrix,
contrast_estimates)`. `pairwise()` returns a `MarginsResult` whose
`ci_method=` passes through the simultaneous-CI machinery (`sup-t`
under `vce="simulation"` is narrower than Bonferroni when contrasts
are correlated).

## Difference-in-differences

Two small additions turn the module into a full DiD estimator:

- `MarginsResult.contrast(c, labels=None)` forms any linear combination
  of the estimates, using $c^\top V_m c$ directly on the already-computed
  joint covariance. Exact under the same delta-method approximation; no
  extra differentiation.
- `Margins.did(group, condition, ...)` sets up the 2×2 grid and returns
  a `DiDResult` bundling the four cell predictions, the two simple
  effects, and the DiD — all sharing the same joint covariance (so you
  can test them jointly through `result.joint.vcov`).

```python
M = Margins(fit)
res = M.did("group", "preexist_Y",
            group_levels=["A", "B"], condition_levels=[0, 1],
            atexog={"age": 60, "female": 0})  # optional: fix covariates
print(res)          # shows cells + simple effects + DiD
res.did.estimate    # the DiD point estimate
res.did.ci_lower    # lower 95% CI
res.cells.vcov      # 4x4 joint covariance of the cell predictions
```

For **multi-outcome models** (MNLogit, OrderedModel), `did()` returns a
`DiDResult` where every field carries the K-outcome axis.  The four cells
contain K probabilities each (summing to 1 per profile), the simple effects
contain 2*K estimates, and the DiD contains K estimates whose sum is exactly
zero.  The `joint` field preserves the full (3*K, 3*K) covariance, so you can
form cross-outcome contrasts such as "DiD on P(class=1) minus DiD on
P(class=0)" via `res.joint.contrast(...)`.  Use `res.outcome(k)` to slice the
entire bundle to a single outcome class.

**Why this matters for nonlinear models (Ai & Norton, 2003).** In a
logit, the coefficient on the `group × condition` interaction is on the
*log-odds* scale. On the *probability* scale the DiD is a nonlinear
function of every parameter and every covariate profile — you cannot
read it off the interaction coefficient. The test suite verifies:

- On **OLS**, `did()` returns exactly the interaction coefficient with
  exactly its SE (matches to 1e-8).
- On **logit**, `did()` returns something *very different* from the
  interaction coefficient (e.g. −0.676 on log-odds vs −0.147 on
  probability in one test run), and matches a by-hand four-cell
  computation to 1e-10.

See [CHANGELOG.md](CHANGELOG.md).

## Files

- `smmargins/`        — the package directory.
  - `core.py`         — the `Margins` orchestrator (predict/dydx/did).
  - `_design.py`      — `DesignResolver`: frame, design info, `at=`/`over=` expansion.
  - `_engine.py`      — `PredictionEngine`: predictions, gradients, scale dispatch, weights.
  - `_derivs.py`      — `DerivativeEngine`: continuous/count/discrete dydx component builders.
  - `data.py`         — data profiling and `patsy` integration.
  - `transforms.py`   — `Transform` class and built-in scales (0.4).
  - `inference.py`    — KR simulation and bootstrap VCE workers.
  - `results.py`      — `MarginsResult`, `DiDResult`, `WaldResult` containers.
  - `utils.py`        — math helpers, Jacobian primitives, validators.
- `tests/`            — test suite (split by functionality).
  - `tests/comparison/` — R `marginaleffects` reference parity tests
    (`generate_r.R` + `test_r.py`).
- `demo_margins.py`    — Williams-style walkthrough on a simulated
  dataset, including robust SEs, KR simulation, bootstrap, and
  simultaneous-CI examples.
- `demo_did.py`        — healthcare-style 2×2 DiD walkthrough.
