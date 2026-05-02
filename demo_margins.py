"""
demo_margins.py
===============

Walkthrough of the core analyses in Richard Williams' *Margins01* notes
(https://academicweb.nd.edu/~rwilliam/stats/Margins01.pdf), implemented
on top of StatsModels + patsy + the ``smmargins`` package.

Sections
--------
  1.  Adjusted predictions at specific values (APR / ``margins, at(...)``)
  2.  APM vs AAP (``margins, atmeans`` vs ``margins``)
  3.  MER vs MEM vs AME for a continuous covariate
  4.  Discrete contrast for a categorical variable
  5.  Discrete change for a 0/1 dummy
  6.  AME by interaction subgroup (Williams' motivating example)
  7.  Predicted probability over age, by sex (table for plotting)
  8.  Analytic vs FD parity check
  9.  Robust covariance (``cov_type="HC3"``)
  10. Krinsky–Robb simulation VCE
  11. Pairs bootstrap VCE
  12. Simultaneous CIs via sup-t
  13. Cluster-robust SEs (``cov_type="cluster"``)
  14. Multiple-comparison adjustments side-by-side (Bonferroni / Sidak / sup-t)
  15. User-supplied parameter covariance (``vcov=``)
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from smmargins import Margins

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
print(M.predict(atexog={"age": [25, 45, 65]}))
print()

# ---------------------------------------------------------------------------
# 2. Adjusted prediction at means (APM)  vs  average adjusted prediction (AAP)
# ---------------------------------------------------------------------------
print("=" * 80)
print("2. APM  (margins, atmeans)   vs   AAP  (margins)")
print("=" * 80)
print("APM:"); print(M.predict(at="mean"))
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
print(M.dydx("age", atexog={"age": [25, 45, 65]}))
print("\nMEM (at means of everything):")
print(M.dydx("age", at="mean"))
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
print(M.dydx("age", atexog={"female": [0, 1]}))
print()

# ---------------------------------------------------------------------------
# 7. Adjusted predictions, age by sex — table suitable for plotting
# ---------------------------------------------------------------------------
print("=" * 80)
print("7. Predicted Pr(voted) over age, for each sex")
print("=" * 80)
tbl = M.predict(atexog={"age": list(range(20, 91, 10)), "female": [0, 1]})
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

# ---------------------------------------------------------------------------
# 9. Robust covariance (Feature 1)
#    Recompute SEs with HC3 heteroskedasticity-consistent covariance.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("9. Robust covariance — HC3")
print("=" * 80)
M_hc3 = Margins(fit, cov_type="HC3")
print(M_hc3.dydx("age"))

# ---------------------------------------------------------------------------
# 10. Krinsky–Robb simulation VCE (Feature 2)
#     Draw parameters from their sampling distribution and evaluate margins.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("10. Krinsky–Robb simulation VCE")
print("=" * 80)
print(M.dydx("age", vce="simulation", n_sims=2000, sim_seed=42))

# ---------------------------------------------------------------------------
# 11. Bootstrap VCE (Feature 3)
#     Pairs bootstrap with 500 replications.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("11. Bootstrap VCE")
print("=" * 80)
print(M.dydx("age", vce="bootstrap", n_boot=500, boot_seed=42))

# ---------------------------------------------------------------------------
# 12. Simultaneous CIs — sup-t (Feature 4)
#     Use simulation draws to compute simultaneous CIs for a family of margins.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("12. Simultaneous CIs (sup-t)")
print("=" * 80)
print(M.predict(atexog={"age": [25, 45, 65]},
                vce="simulation", n_sims=2000, sim_seed=42,
                ci_method="sup-t"))

# ---------------------------------------------------------------------------
# 13. Cluster-robust SEs
#     Synthesize a clustering structure (e.g., households of ~10 voters who
#     share unobserved local effects). Cluster-robust SEs propagate that
#     correlation through the Jacobian to the AME.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("13. Cluster-robust SEs vs nonrobust  (synthetic household clusters)")
print("=" * 80)
df_c = df.copy()
df_c["household"] = rng.integers(0, N // 10, N)  # ~10 obs per cluster
fit_c = smf.logit(
    "voted ~ age + income + C(educ) + female + age:female",
    data=df_c,
).fit(disp=False)
M_nonrobust = Margins(fit_c)
M_cluster = Margins(fit_c, cov_type="cluster",
                    cov_kwds={"groups": df_c["household"]})
ame_nr = M_nonrobust.dydx("age").se[0]
ame_cl = M_cluster.dydx("age").se[0]
print(f"AME(age) SE — nonrobust : {ame_nr: .6f}")
print(f"AME(age) SE — cluster    : {ame_cl: .6f}   (ratio {ame_cl / ame_nr:.2f}x)")

# ---------------------------------------------------------------------------
# 14. Multiple-comparison adjustments
#     A family of 5 marginal effects at different ages. Pointwise CIs
#     under-cover the joint event "all 5 contain the truth"; Bonferroni
#     and Sidak inflate the critical value uniformly; sup-t uses the
#     simulation draws to exploit correlation across the family.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("14. Family-wise CI methods at age=25,35,45,55,65")
print("=" * 80)
ages = [25, 35, 45, 55, 65]
common = dict(atexog={"age": ages}, vce="simulation",
              n_sims=4000, sim_seed=123)
pw   = M.predict(**common, ci_method="pointwise")
bonf = M.predict(**common, ci_method="bonferroni")
sidk = M.predict(**common, ci_method="sidak")
supt = M.predict(**common, ci_method="sup-t")

widths = pd.DataFrame({
    "age":        ages,
    "pointwise":  pw.ci_upper   - pw.ci_lower,
    "bonferroni": bonf.ci_upper - bonf.ci_lower,
    "sidak":      sidk.ci_upper - sidk.ci_lower,
    "sup-t":      supt.ci_upper - supt.ci_lower,
}).set_index("age")
print("CI widths:")
print(widths)
print("\nBonferroni >= Sidak (always); for correlated margins sup-t is "
      "typically narrower than both.")

# ---------------------------------------------------------------------------
# 15. User-supplied vcov
#     Drop in any (k, k) covariance matrix you trust — e.g. a sandwich
#     computed offline, a Bayesian posterior covariance, or the output
#     of a custom resampling scheme — and smmargins will sandwich it
#     through the Jacobian without recomputing anything else.
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("15. User-supplied parameter covariance (vcov=)")
print("=" * 80)
V_default = fit.cov_params().to_numpy()
V_inflated = V_default * 1.5     # toy example: assume 50% wider sampling cov
M_v = Margins(fit, vcov=V_inflated)
ame_default = M.dydx("age").se[0]
ame_user    = M_v.dydx("age").se[0]
print(f"AME(age) SE — default cov_params() : {ame_default: .6f}")
print(f"AME(age) SE — vcov = 1.5 x default : {ame_user: .6f}   "
      f"(ratio {ame_user / ame_default:.3f}, expect ≈ sqrt(1.5)={np.sqrt(1.5):.3f})")
