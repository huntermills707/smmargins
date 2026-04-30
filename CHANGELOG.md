# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
