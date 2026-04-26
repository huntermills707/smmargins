"""
demo_margins.py
===============

Walkthrough of the core analyses in Richard Williams' *Margins01* notes
(https://academicweb.nd.edu/~rwilliam/stats/Margins01.pdf), implemented
on top of StatsModels + patsy + the ``marginal_effects`` module.

We'll reproduce, in turn:

  1. Adjusted predictions at specific values (APR / "margins, at(...)")
  2. Adjusted predictions at means (APM / "margins, atmeans")
  3. Average adjusted predictions (AAP / "margins")
  4. Marginal effects at representative values (MER / "margins, dydx(..) at(..)")
  5. Marginal effects at means (MEM / "margins, dydx(..) atmeans")
  6. Average marginal effects (AME / "margins, dydx(..)")
  7. Discrete changes for categorical variables
  8. The interaction example that Williams uses to motivate AME
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from marginal_effects import Margins

pd.options.display.width = 120
pd.options.display.float_format = "{: .4f}".format

# ---------------------------------------------------------------------------
# Simulate a binary-outcome dataset with structure similar to Williams' notes
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
N = 5_000
df = pd.DataFrame(
    {
        "age":    rng.normal(45, 12, N).clip(18, 90),
        "income": rng.lognormal(10.5, 0.4, N),          # ~36k median
        "educ":   rng.choice(["hs", "college", "grad"], N, p=[0.4, 0.4, 0.2]),
        "female": rng.integers(0, 2, N),
    }
)
eta = (
    -4.0
    + 0.05 * df["age"]
    + 0.00001 * df["income"]
    + 0.8 * (df["educ"] == "college")
    + 1.4 * (df["educ"] == "grad")
    + 0.3 * df["female"]
    - 0.0004 * df["age"] * (df["female"])        # interaction
)
df["voted"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-eta))).astype(int)

print("Sample:")
print(df.head(3), "\n")

# ---------------------------------------------------------------------------
# Fit a logit with an interaction, like the Williams example
# ---------------------------------------------------------------------------
fit = smf.logit(
    "voted ~ age + income + C(educ) + female + age:female",
    data=df,
).fit(disp=False)
print("=" * 80)
print("Fitted logit")
print("=" * 80)
print(fit.summary().tables[1])
print()

# `analytic=True` is the default: the outer ∂g/∂β goes through
# `family.link.inverse_deriv` for any GLM (Logit/Probit/Poisson/...) and
# the identity link for OLS/WLS/GLS, falling back to central finite
# differences only when the link derivative isn't available. Set
# `analytic=False` to force FD; you'll get the same answers (see the
# parity check at the bottom of this file) but pay p extra forward
# predict() calls per statistic.
M = Margins(fit)

# ---------------------------------------------------------------------------
# 1. Adjusted predictions at representative values (APR)
#    Stata: margins, at(age=(25 45 65))
# ---------------------------------------------------------------------------
print("=" * 80)
print("1. APR  (predict at age=25,45,65; everything else at sample values)")
print("=" * 80)
print(M.predict(at={"age": [25, 45, 65]}))
print()

# ---------------------------------------------------------------------------
# 2. Adjusted prediction at means (APM)  vs  average adjusted prediction (AAP)
# ---------------------------------------------------------------------------
print("=" * 80)
print("2. APM  (margins, atmeans)   vs   AAP  (margins)")
print("=" * 80)
print("APM:"); print(M.predict(atmeans=True))
print("\nAAP:"); print(M.predict())
print()

# ---------------------------------------------------------------------------
# 3. Marginal effect: MER vs MEM vs AME for `age`
#    (Williams points out these three can differ meaningfully in nonlinear
#    models with interactions)
# ---------------------------------------------------------------------------
print("=" * 80)
print("3. d Pr(voted)/d age : MER (at age=25,45,65),  MEM, and AME")
print("=" * 80)
print("MER (at age=25,45,65):")
print(M.dydx("age", at={"age": [25, 45, 65]}))
print("\nMEM (at means of everything):")
print(M.dydx("age", atmeans=True))
print("\nAME (averaged over the sample):")
print(M.dydx("age"))
print()

# ---------------------------------------------------------------------------
# 4. Discrete contrast for the categorical variable `educ`
# ---------------------------------------------------------------------------
print("=" * 80)
print("4. Discrete AME for educ  (each level vs 'college' as reference)")
print("=" * 80)
print(M.dydx("educ", reference="college"))
print()

# ---------------------------------------------------------------------------
# 5. Discrete change for the dummy `female`  (auto-detected as discrete)
# ---------------------------------------------------------------------------
print("=" * 80)
print("5. AME for female (0/1 dummy):  Pr(voted|female=1) - Pr(voted|female=0)")
print("=" * 80)
print(M.dydx("female"))
print()

# ---------------------------------------------------------------------------
# 6. Interaction-sensitivity: marginal effect of age, separately for men/women
#    This is Williams' classic motivating example: the interaction coefficient
#    alone tells you little about what the marginal effect actually is for any
#    given subpopulation.
# ---------------------------------------------------------------------------
print("=" * 80)
print("6. AME of age, separately by sex  (Williams' interaction illustration)")
print("=" * 80)
print(M.dydx("age", at={"female": [0, 1]}))
print()

# ---------------------------------------------------------------------------
# 7. Adjusted predictions, age by sex — table suitable for plotting
# ---------------------------------------------------------------------------
print("=" * 80)
print("7. Predicted Pr(voted) over age, for each sex")
print("=" * 80)
tbl = M.predict(at={"age": list(range(20, 91, 10)), "female": [0, 1]})
print(tbl)

# ---------------------------------------------------------------------------
# 8. Analytic vs FD: same answers, faster path
#    Logit exposes `family.link.inverse_deriv`, so the analytic outer
#    Jacobian is used by default. Toggling `analytic=False` reroutes
#    every statistic through central finite differences — useful as a
#    sanity check or when working with a custom Link subclass that
#    doesn't implement inverse_deriv.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("8. Analytic vs FD — same numbers, taken via different paths")
print("=" * 80)
M_fd = Margins(fit, analytic=False)
ame_an = M.dydx("age")
ame_fd = M_fd.dydx("age")
print(f"AME(age) analytic : est={ame_an.estimate[0]: .8f}  se={ame_an.se[0]: .8f}")
print(f"AME(age) FD       : est={ame_fd.estimate[0]: .8f}  se={ame_fd.se[0]: .8f}")
print(f"max abs diff      : "
      f"est {abs(ame_an.estimate[0] - ame_fd.estimate[0]): .2e}, "
      f"se {abs(ame_an.se[0] - ame_fd.se[0]): .2e}")
