# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-05-01

### Added

- Internal structure refactored into a package (`smmargins/`) for better maintainability. Logic is now split across `core.py`, `data.py`, `results.py`, and `utils.py`.
- `pyproject.toml` updated to support package-based installation.

### Fixed

- Removed redundant `DiDResult` definition in `core.py`.

### Changed

- `smmargins.py` (monolith) is now `smmargins/` (package).

- Multi-outcome model support for `statsmodels.MNLogit` and
  `statsmodels.miscmodels.ordinal_model.OrderedModel`. `Margins.predict()`
  and `Margins.dydx()` now return one estimate per outcome class with full
  joint covariance across classes and rows.
- Analytic outer Jacobian for MNLogit via the softmax derivative
  (`_softmax_grad_mean`); OrderedModel uses central finite differences.
- New `outcome=` keyword on `Margins.predict()` and `Margins.dydx()` for
  subsetting results to specific class label(s) or index/indices. Single-
  outcome models silently ignore it.
- `MarginsResult.outcome(k)` helper for post-hoc slicing of multi-outcome
  results, preserving the corresponding submatrix of the joint covariance.
- `Margins.n_outcomes` and `Margins.outcome_labels` properties for
  introspecting a fitted model's outcome structure.
- `Margins.did()` now works on multi-outcome models (MNLogit, OrderedModel).
  The 2×2 contrast matrix lifts to a block-diagonal (3*K, 4*K) matrix via
  Kronecker product with `eye(K)`, preserving the K-outcome axis on
  `cells`, `simple_effects`, `did`, and `joint`. New `DiDResult.outcome(k)`
  slicer delegates to `MarginsResult.outcome()` on each component. The
  `joint` field exposes the (3*K, 3*K) covariance for cross-outcome
  contrasts.
- Elasticity methods (`eyex`, `dyex`, `eydx`) now work on multi-outcome
  models (MNLogit, OrderedModel). A `RuntimeWarning` is emitted when
  predicted probabilities fall below 1e-12 for near-zero outcome classes.
- Parity tests against `statsmodels.get_margeff()` on MNLogit (3-class) for
  AME and MEM, plus analytic-vs-FD parity, hand-computed AAP-sums-to-one,
  discrete-contrast hand check, and `outcome=` slicing tests.
- Basic correctness tests for OrderedModel (4-category logit) on AAP, AME,
  MEM, and `outcome=` slicing.
- DiD multi-outcome tests: shape invariants, K-sum-to-zero across outcomes,
  hand-computed DiD per outcome (1e-10), cross-outcome contrast via
  `joint`, and `DiDResult.outcome(k)` slicing.
- Elasticity multi-outcome tests: hand-derived references via
  `dyex = dydx*x`, `eydx = dydx/y`, `eyex = dydx*x/y` cross-checks at FD
  tolerance, single-outcome regression, and blow-up warning behaviour.

### Changed

- `Margins._predict` now always returns shape `(n, K)`. Single-outcome
  models return `(n, 1)` instead of the previous `(n,)`. All internal
  callers updated; external behaviour is unchanged for single-outcome
  models.
- `MarginsResult.summary()` returns a long-format DataFrame with a leading
  `outcome` column when applied to a multi-outcome result.
- Parameter vectors from models with 2-D `params` (MNLogit's `(p, K-1)`)
  are now flattened in column-major order to match `statsmodels`'s
  `cov_params()` layout.

## [0.1.0] - 2026-04-30

### Added

- Initial release on PyPI.
- `Margins.predict()` — adjusted predictions (AAP, APM, APR) with `at=` and name-keyed `atexog=`.
- `Margins.dydx()` — marginal effects (AME, MEM, MER) for continuous and discrete variables, plus elasticities (`eyex`, `dyex`, `eydx`).
- `Margins.did()` — 2×2 difference-in-differences on the response scale with joint covariance.
- `MarginsResult.contrast()` — exact linear combinations of estimates using the already-computed joint covariance.
- Delta-method standard errors for any fitted model exposing `params`, `cov_params()`, and `predict(params, exog)`.
- Analytic outer Jacobian for GLM and linear-regression models; automatic fallback to central finite differences otherwise.
- Full test coverage against `statsmodels.get_margeff()` and hand-derived analytic results.
