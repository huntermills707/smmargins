# Implementation Prompt — MNLogit / OrderedModel Multi-Outcome Support for `smmargins`

> Hand this to a fresh Sonnet-level Claude Code session running in `/home/hunter/Workspace/smmargins`. The prompt is self-contained: do not assume any prior conversation context. Edit the repo path in the "Reference reading" section if you move it.

## Your role

You are a software engineer adding multi-outcome model support to `smmargins`, a Python package providing Stata-style marginal effects for StatsModels. Read the existing source carefully, write code matching the package's style, and verify against `statsmodels.get_margeff` to high precision. Commit at each phase boundary.

## Context

`smmargins` provides a class-based API (`Margins(fit).predict()`, `M.dydx(...)`) for adjusted predictions and marginal effects with delta-method standard errors. It works on any fitted StatsModels result that exposes `params`, `cov_params()`, and `predict(params, exog)`.

**Value props you must preserve**:
- Class-based API (no functional shift).
- Pandas-native (no Polars dependency).
- DiD with joint covariance (`M.did(...)`).
- Generic dispatch via `predict(params, exog)`.
- Analytic outer Jacobian for any model with a known link derivative; FD fallback otherwise.
- Patsy-DesignInfo formula handling, with a "raw exog" fallback.

**Current limitation**: every statistic in the package assumes 1-D output per row from `_predict`. Models like `statsmodels.discrete.discrete_model.MNLogit` and `statsmodels.miscmodels.ordinal_model.OrderedModel` return `(n, K)` probabilities — one column per outcome class. Right now this would silently misbehave or error. Your task is to add proper multi-outcome support end-to-end.

## Reference reading (do this first; do not skip)

1. Read `smmargins.py` in full (~2500 lines, well-commented). Focus areas:
   - `Margins._predict` (~smmargins.py:874) — wraps `model.predict(params, exog)`.
   - `Margins._grad_mean_predict` (~smmargins.py:967) — analytic gradient primitive.
   - `Margins._link_deriv` (~smmargins.py:919) — link-derivative dispatch.
   - `Margins._delta` (~smmargins.py:1412) — the delta-method engine.
   - `Margins.predict` (~smmargins.py:1472) — statistic closure for AAP/APM/APR.
   - `Margins.dydx` (~smmargins.py:1617) and helpers `_dydx_continuous_components` (~1920), `_dydx_count_components` (~2031), `_dydx_discrete_components` (~2090).
   - `MarginsResult` class (~smmargins.py:196).
   - `Margins.did` (~smmargins.py:2169) — leave alone but understand its shape contract.
2. Read `README.md` for the math and design choices (delta-method-on-mean-prediction is the unifying primitive).
3. Read `tests/test_marginal_effects.py`, `tests/test_predictions.py`, `tests/test_elasticities.py`, `tests/conftest.py` to understand the test style and tolerance bars.
4. **Verify the current state before designing changes**: run `grep -i "mnlogit\|ordered\|multinomial\|softmax" smmargins.py tests/`. If any of this is already partially done, integrate with it rather than rewriting.

## Math (the hardest part — do not skip)

### MNLogit (multinomial logit)

For a K-class multinomial logit with class 0 as reference, the model parameterizes:

```
P(Y = k | x) = exp(eta_k) / sum_{j=0..K-1} exp(eta_j)
where eta_0 = 0,  eta_k = x' beta_k  for k = 1..K-1.
```

`statsmodels.MNLogit.params` is shape `(p, K-1)`; `MNLogit.predict(params, exog)` returns shape `(n, K)`.

**Verify before coding** how the flat parameter vector maps to the matrix — fit a small MNLogit, compare `fit.params.shape` and `fit.params.values.ravel()` against `cov_params().shape`. Likely column-major (`order='F'`), but confirm. The flat-vector layout is the contract you build the Jacobian against.

**Softmax derivative** (canonical):

```
d P_k(x) / d beta_l  =  P_k(x) * (delta_{kl} - P_l(x)) * x
```

with class 0 contributing nothing because `beta_0 = 0`.

**Gradient of mean prediction** `(1/n) sum_i P_k(x_i)` w.r.t. `beta_l`:

```
(1/n) sum_i  P_k(x_i) * (delta_{kl} - P_l(x_i)) * x_i
```

For a statistic that's a vector of `K` mean predictions (one per class), the Jacobian assembled across all `(K-1)` parameter blocks is `(K, p*(K-1))`.

### OrderedModel

For an ordered K-category model:

```
P(Y = k | x) = F(tau_{k+1} - x' beta) - F(tau_k - x' beta)
```

with thresholds `-inf = tau_0 < tau_1 < ... < tau_{K-1} < tau_K = +inf` and `F` the link CDF (logit or probit). statsmodels parameterizes thresholds via unconstrained deltas: `alpha_1 = tau_1`, `alpha_k = log(tau_k - tau_{k-1})` for `k > 1`.

**Recommendation for v0**: implement OrderedModel via the **finite-difference fallback path only**. The link-derivative path is intricate enough that getting it right and tested adds substantial scope; FD on the mean-prediction primitive is one extra rebuild step regardless of K. Document this in code and README.

## Architecture changes

These changes must be backwards-compatible: every existing test must still pass with K = 1 (the implicit default for single-outcome models).

### Phase A — plumb a K-outcome axis through the pipeline

**A.1.** Add `n_outcomes` property on `Margins`. Detect K at construction:
- Most models: `K = 1`.
- `MNLogit`: `K = self.model.J` (verify on a real fit).
- `OrderedModel`: derive from `model.endog_names` length or `params` indexing.
- Last-resort: run a 1-row probe prediction and check `.shape`.

**A.2.** Add `outcome_labels` property: list of length K (e.g., MNLogit class labels via `model._ynames_map`); `None` for single-outcome.

**A.3.** Refactor `_predict` (smmargins.py:~874) to always return shape `(n, K)`:
- 1-D output → reshape to `(n, 1)`.
- 2-D output → return as-is.
- Update docstring.
- Update **all** callers; do not silently flatten anywhere.

**A.4.** Refactor `_grad_mean_predict` (smmargins.py:~967) to return `(K, p_full)`:
- Single-outcome with known link derivative: `(1, p)`.
- MNLogit: build the softmax-derivative gradient (math above).
- `p_full` is the size of the *flat* parameter vector — for MNLogit, `p * (K-1)`.

**A.5.** Refactor `_link_deriv` (smmargins.py:~919) to dispatch on model type:
- GLMs: existing `family.link.inverse_deriv` path; reshape its `(n,)` output to `(n, 1)`.
- MNLogit: introduce a sibling primitive `_softmax_grad_mean(X, beta)` that directly returns `(K, p_full)`. Have `_grad_mean_predict` dispatch to it. (Alternative: return `None` and fall back to FD — acceptable for v0 but slower.)
- OrderedModel: return `None` (FD fallback) for v0; comment the deferral.

**A.6.** Refactor `_delta` (smmargins.py:~1412) to accept statistics returning `(m, K)` and produce a `(m*K, p_full)` Jacobian:
- Statistic returns 2-D `(m, K)`. Flatten row-major (`.ravel()`) for the result `estimate`.
- Jacobian `(m, K, p)` → `(m*K, p)` row-major.
- Pass `outcome_labels` and a per-row `outcome_index` through to `MarginsResult`.

**A.7.** Refactor `MarginsResult` (smmargins.py:196):
- Add `outcome_labels: Optional[Sequence[str]]` and `outcome_index: Optional[ndarray]` to `__init__`.
- When set, `summary()` returns long-format with an `outcome` column and one row per `(label, outcome)` pair.
- When not set: behavior byte-identical to today.
- `contrast()` semantics: contrast vector is over the `(m * K)` flat rows. Document this.
- Add `result.outcome(k)` helper: slice rows belonging to outcome `k` (by label or index) and return a new `MarginsResult` (with its rows of `vcov` carved out — keep joint covariance among the kept rows).

### Phase B — update every statistic closure

These closures currently return `(m,)`. After your refactor each must return `(m, K)`:

- `Margins.predict` statistic (smmargins.py:1595–1599)
- `Margins.dydx` continuous path statistic (smmargins.py:~2009)
- `Margins.dydx` count path
- `Margins.dydx` discrete path
- Multi-variable `dydx` stacking (smmargins.py:~1857–1898) — the stacking must respect K (stack along the leading "variable" axis, keep K trailing)

Use `_predict(beta, X)` returning `(n, K)`, then mean over axis 0 → `(K,)` per profile, then stack to `(m, K)`. Resist re-flattening anywhere — `(m, K)` is an invariant, not a convenience.

### Phase C — discrete & count `dydx` for multi-outcome

For discrete and `count=True` `dydx`, the statistic is `E[f(X_a)] - E[f(X_b)]`. With `(K,)` outcomes per profile this becomes a `(K,)` difference per pair. No further math — just propagate the shape.

For continuous `dydx` with central differences on the data column, you compute `(predict(X_+h) - predict(X_-h)) / (2h)` averaged over rows. With `(n, K)` predictions this becomes `(K,)` per variable.

### Phase D — scope boundaries (DO NOT IMPLEMENT IN THIS PR)

- **DiD on multi-outcome**: have `Margins.did(...)` raise `NotImplementedError("DiD on multi-outcome models is not yet supported")` when `n_outcomes > 1`. Reference this prompt's filename in the message. The 2x2 grid × K outcomes design needs more thought.
- **Elasticities × multi-outcome**: same — raise on `n_outcomes > 1` with a clear message. (Adding it later is straightforward; not in this PR.)
- **Analytic Jacobian for OrderedModel**: defer. FD only.

## API surface

After your changes, MNLogit usage looks like:

```python
import statsmodels.api as sm
from smmargins import Margins

fit = sm.MNLogit(y, X).fit()
M = Margins(fit)

# all outcomes by default
res = M.predict()         # AAP for each class
res.summary()             # long-format DataFrame: rows × outcomes
res.outcome(2)            # MarginsResult sliced to class 2

res = M.dydx("x1")        # AME of x1 on each class probability
res.summary()
```

Add a new `outcome=` keyword on `predict` and `dydx` that subsets to specific class label(s):

```python
M.predict(outcome=1)              # only class 1
M.predict(outcome=[0, 2])         # classes 0 and 2
M.dydx("x1", outcome="versicolor")
```

Default is "all outcomes." Single-outcome models silently ignore `outcome=`.

## Tests (acceptance bar)

Add `tests/test_mnlogit.py`:

1. **Parity vs `statsmodels.get_margeff` on a 3-class MNLogit fit**:
   - AME (overall) on a continuous variable, both estimate and SE, atol=1e-5 / rtol=1e-3.
   - MEM (at='mean') on a continuous variable.
   - Parametrize over the 3 outcome classes.
2. **Hand AAP check**: K AAP values must sum to 1.0 (per-row probabilities sum to 1, average preserves this) within 1e-10.
3. **Analytic-vs-FD parity** for AME: `Margins(fit, analytic=True)` and `Margins(fit, analytic=False)` agree on estimate and SE within 1e-5 / 1e-3.
4. **Discrete contrast** on a binary covariate in MNLogit: spot-check against a hand-computed two-row `predict` difference.
5. **`outcome=` subsetting**: `M.predict(outcome=1).estimate` matches the corresponding rows of `M.predict().estimate`. Joint vcov of the slice equals the corresponding submatrix of the full vcov.

Add `tests/test_ordered.py`:

1. **Parity vs `get_margeff` on `OrderedModel`** (logit link, 4 categories) for AME, atol=1e-5 / rtol=1e-3. FD path only.
2. Hand AAP sums to 1.0.

Existing tests must all pass. Do not modify any existing test unless an existing test relied on `_predict` returning 1-D — in which case update it and call out the diff in the PR description.

## Implementation phasing — commit at each boundary

1. **A.1–A.2**: `n_outcomes` and `outcome_labels` detection. Tests asserting these are correct on OLS, Logit, MNLogit, OrderedModel fixtures. → commit.
2. **A.3**: `_predict` returns `(n, K)`. All callers updated. Existing tests still pass. → commit.
3. **A.4–A.5**: `_grad_mean_predict` returns `(K, p)` (with K=1 elsewhere). Existing analytic-Jacobian tests still pass. → commit.
4. **A.6–A.7**: `_delta` and `MarginsResult` carry the outcome axis. Single-outcome behavior byte-identical. → commit.
5. **Phase B**: statistic closures return `(m, K)`. MNLogit now works end-to-end via FD. → commit.
6. **MNLogit analytic softmax gradient** (`_softmax_grad_mean`) wired into the analytic path. Analytic-vs-FD parity test green. → commit.
7. **Phase C**: discrete + count for multi-outcome (mostly free at this point). → commit.
8. **OrderedModel**: confirm FD-only path works. → commit.
9. **Tests**: parity tests against `get_margeff`, iterate to green. → commit.
10. **README + CHANGELOG**: "Multi-outcome models" subsection; note OrderedModel is FD-only; update supported-models list. → commit.

If you finish a phase and discover Phase A.X was already partially done in the existing source (something the prompt didn't anticipate), integrate rather than redo, and note it in the PR description.

## Quality bar

- All existing tests pass: `python -m pytest tests/ -v`.
- New parity tests pass at the tolerances above.
- No new public API surface beyond `outcome=` kwargs and `MarginsResult.outcome(k)`.
- Code style matches the rest of `smmargins.py`: numpydoc docstrings, type hints, no emojis.
- No new dependencies.
- README has one paragraph per new model class with a runnable example.
- `CHANGELOG.md` gets an `## [Unreleased]` block describing the additions.

## Anti-goals

- Don't refactor anything beyond what's needed for multi-outcome. Resist "while I'm in here" cleanup.
- Don't add MNLogit/OrderedModel-specific code in `MarginsResult` — keep the K-axis fully generic so future K-output models (zero-inflated count, etc.) work without changes.
- Don't add new vcov options, plotting, or formatters — out of scope.
- Don't break raw-exog mode. MNLogit is commonly fit without a formula.
- Don't introduce abstractions for hypothetical future K-output models beyond what MNLogit + OrderedModel actually need today.

## Deliverable

A single PR (or branch ready for one) that:
- Adds multi-outcome support end-to-end for MNLogit (analytic + FD) and OrderedModel (FD only).
- Adds `tests/test_mnlogit.py` and `tests/test_ordered.py`.
- Updates README and CHANGELOG.
- Leaves `Margins.did(...)` raising `NotImplementedError` on multi-outcome models with a TODO referencing this work.

When you finish, summarize: which files changed (with line counts), which tests were added, where you made FD-vs-analytic tradeoffs, and any places where the existing API had to bend to fit the K-axis.

## Kickoff

Start by running:

```
grep -n -i "mnlogit\|ordered\|multinomial\|softmax\|n_outcomes\|outcome_labels" smmargins.py tests/
wc -l smmargins.py tests/*.py
python -m pytest tests/ -v
```

to baseline current state, then read `smmargins.py` end-to-end before touching anything.
