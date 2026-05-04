# How to verify smmargins results against R marginaleffects

## Prerequisites

- {doc}`Tutorial: First steps with smmargins </tutorials/getting_started>` — basic model fitting and margins
- R installed locally with the `marginaleffects`, `readr`, and `sandwich` packages

## Problem statement

You want to confirm that `smmargins` produces results consistent with the de-facto cross-language reference, R's `marginaleffects` package. You need to know which cases are covered, what tolerances are expected, and how to regenerate reference outputs.

## Minimal working solution

The test suite in `tests/comparison/` contains six parity cases. These tests auto-skip if the reference CSVs are missing, so the rest of the suite still runs without R.

```python
import pathlib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from numpy.testing import assert_allclose

from smmargins import Margins

HERE = pathlib.Path("tests/comparison")

def _read_r_csv(path):
    return pd.read_csv(path)

def _match_label(ame_labels, r_term):
    labels = np.array(ame_labels)
    for prefix in ("d", "e"):
        stripped = np.array([lab[len(prefix):] if lab.startswith(prefix) else lab for lab in labels])
        mask = stripped == r_term
        if np.any(mask):
            return np.where(mask)[0][0]
    return None

# Load shared fixture data
df = pd.read_csv(HERE / "data" / "data_seed42.csv")

# Case 1: Logit AME with HC3
fit = smf.logit("y_logit ~ x1 + x2 + C(grp)", data=df).fit(disp=False)
M = Margins(fit, cov_type="HC3")
ame = M.dydx(["x1", "x2"])

ref = _read_r_csv(HERE / "r_logit_ame_hc3.csv")
for _, row in ref.iterrows():
    if row["term"] == "grp":
        continue
    idx = _match_label(ame.labels, row["term"])
    assert_allclose(ame.estimate[idx], row["estimate"], atol=1e-6)
    assert_allclose(ame.se[idx], row["std.error"], atol=5e-5)
print("Case 1 (Logit AME HC3): PASS")
```

## Variations

### Case 2: Poisson AME with HC3

```python
df = pd.read_csv(HERE / "data" / "data_seed42.csv")
fit = smf.glm("count ~ x1 + x2", data=df, family=sm.families.Poisson()).fit()
M = Margins(fit, cov_type="HC3")
ame = M.dydx(["x1", "x2"])

ref = _read_r_csv(HERE / "r_poisson_ame.csv")
for _, row in ref.iterrows():
    idx = _match_label(ame.labels, row["term"])
    assert_allclose(ame.estimate[idx], row["estimate"], atol=1e-6)
    assert_allclose(ame.se[idx], row["std.error"], atol=5e-4)
print("Case 2 (Poisson AME HC3): PASS")
```

### Case 3: OLS pairs bootstrap

```python
df = pd.read_csv(HERE / "data" / "data_seed12345.csv")
fit = smf.ols("y_ols ~ x1 + x2", data=df).fit()
M = Margins(fit)
ame = M.dydx(["x1", "x2"], vce="bootstrap", n_boot=1000, boot_seed=42)

ref = _read_r_csv(HERE / "r_ols_bootstrap.csv")
for _, row in ref.iterrows():
    idx = _match_label(ame.labels, row["term"])
    assert_allclose(ame.estimate[idx], row["estimate"], atol=1e-10)
    r_se = row["std.error"] if pd.notna(row.get("std.error")) else (row["conf.high"] - row["conf.low"]) / (2 * 1.96)
    mc_tol = 5 * max(ame.se[idx], r_se) / np.sqrt(1000)
    assert abs(ame.se[idx] - r_se) < mc_tol
print("Case 3 (OLS bootstrap): PASS")
```

### Regenerating reference outputs

```bash
cd tests/comparison && Rscript generate_r.R
```

## Tolerance summary

| Case | Estimate tol | SE tol |
|------|-------------|--------|
| Logit AME with HC3 | 1e-6 | 5e-5 |
| Poisson AME with HC3 | 1e-6 | 5e-4 |
| OLS pairs bootstrap (n=1000) | 1e-10 | Monte Carlo (5 sigma) |
| OLS with HC1, polynomial + interaction | 1e-6 | 1e-6 |
| Logit cluster-robust AME | 1e-5 | 1e-4 |
| Logit AME at representative x2 values | 1e-5 | 1e-4 |

> ⚠️ **Trade-off:** The R parity tests are the strongest validation of the package's numerical correctness, but they require a working R installation. They auto-skip when reference files are missing, so CI pipelines without R still pass. The bootstrap SE tolerance is Monte Carlo — it scales with the bootstrap SE itself, not an absolute threshold.

## When to use this

Run the parity test suite whenever you modify the core inference machinery (Jacobian computation, delta method, VCE workers) or upgrade statsmodels. It is also useful when porting analyses from R to Python — the same data and model specification should match to the tolerances above.

## When NOT to use this

> ⚠️ **Trade-off:** Do not treat these tests as a substitute for understanding your model — they verify numerical parity, not substantive correctness. The factor-contrast handling differs between `smmargins` and R `marginaleffects` (see the `grp` skip in Case 1), so categorical variable comparisons may not align one-for-one.

## See also

- `tests/comparison/generate_r.R` — the R script that generates reference CSVs
- `tests/comparison/test_r.py` — the full pytest parity suite
- {doc}`Mathematical motivation </math>` — the mathematical foundation being verified
