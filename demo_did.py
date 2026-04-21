"""
demo_did.py
===========

Difference-in-differences example directly matching the question:

    "Is there a rate difference of condition X between group A and B,
     with or without preexisting condition Y?"

We fit a logit for P(X=1) on group (A/B), preexisting Y (0/1), their
interaction, and control covariates. Then we use Margins.did() to get,
on the *probability* scale:

  * 4 cell predictions      P(X | group, Y)
  * 2 simple effects        P(X|B,Y) - P(X|A,Y)   at each Y
  * 1 DiD                   (simple effect at Y=1) - (simple effect at Y=0)

All with delta-method standard errors and CIs.

The DiD here is the "does the A-vs-B gap depend on Y?" question.  It is
NOT the coefficient on group×Y (that's on the log-odds scale); on the
probability scale you have to go through the inverse link — which is
exactly what Margins.did() does.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from marginal_effects import Margins

pd.options.display.width = 140
pd.options.display.float_format = "{: .4f}".format

# ---------------------------------------------------------------------------
# Simulate patient-level data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N = 6_000
df = pd.DataFrame({
    "group":        rng.choice(["A", "B"], N, p=[0.55, 0.45]),
    "preexist_Y":   rng.integers(0, 2, N),             # 0 = no Y, 1 = has Y
    "age":          rng.normal(55, 15, N).clip(18, 95),
    "female":       rng.integers(0, 2, N),
})

# True data-generating process:
#   * baseline rate of X depends on age, sex, and Y
#   * group B has a modest additive bump in the log-odds
#   * the group effect is AMPLIFIED among patients with preexisting Y
#     (this is the thing we want to detect)
eta = (
    -3.5
    + 0.04 * df["age"]
    - 0.3 * df["female"]
    + 0.5 * (df["group"] == "B")
    + 1.1 * df["preexist_Y"]
    + 0.8 * (df["group"] == "B") * df["preexist_Y"]   # interaction
)
df["condition_X"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-eta))).astype(int)

print("Raw sample rates of condition X by cell:")
print(df.groupby(["group", "preexist_Y"])["condition_X"].mean().round(4))
print()

# ---------------------------------------------------------------------------
# Fit the logit with the group × preexist_Y interaction + controls
# ---------------------------------------------------------------------------
fit = smf.logit(
    "condition_X ~ C(group) + preexist_Y + C(group):preexist_Y + age + female",
    data=df,
).fit(disp=False)

print("=" * 84)
print("Logit model (coefficients are on the log-odds scale)")
print("=" * 84)
print(fit.summary().tables[1])
print()

# ---------------------------------------------------------------------------
# DiD on the *probability* (response) scale — what the clinical question asks
# ---------------------------------------------------------------------------
M = Margins(fit)
did = M.did("group", "preexist_Y",
            group_levels=["A", "B"],
            condition_levels=[0, 1])

print("=" * 84)
print("DiD on the probability scale, averaged over age and sex distribution")
print("=" * 84)
print(did)

# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------
pA0 = did.cells.estimate[0]   # group=A, Y=0
pA1 = did.cells.estimate[1]   # group=A, Y=1
pB0 = did.cells.estimate[2]   # group=B, Y=0
pB1 = did.cells.estimate[3]   # group=B, Y=1
se_simple_Y0 = did.simple_effects.se[0]
se_simple_Y1 = did.simple_effects.se[1]
did_est, did_se = did.did.estimate[0], did.did.se[0]

print()
print("=" * 84)
print("Plain-language summary")
print("=" * 84)
print(f"Condition X rate, group A, no preexisting Y : {pA0:.3%}")
print(f"Condition X rate, group A, with Y           : {pA1:.3%}")
print(f"Condition X rate, group B, no preexisting Y : {pB0:.3%}")
print(f"Condition X rate, group B, with Y           : {pB1:.3%}")
print()
print(f"Rate difference (B - A) among NO-Y patients  : "
      f"{(pB0 - pA0):+.3%}  (SE {se_simple_Y0:.3%})")
print(f"Rate difference (B - A) among WITH-Y patients: "
      f"{(pB1 - pA1):+.3%}  (SE {se_simple_Y1:.3%})")
print()
print(f"Difference-in-differences                   : "
      f"{did_est:+.3%}  (SE {did_se:.3%})")
print(f"  -> the B-vs-A gap is {abs(did_est):.3%} larger among patients "
      f"with preexisting Y.")
print(f"  -> 95% CI: ({did.did.ci_lower[0]:+.3%}, {did.did.ci_upper[0]:+.3%})")
print(f"  -> p-value: {did.did.pvalue[0]:.4g}")

# ---------------------------------------------------------------------------
# Sensitivity: DiD at a specific patient profile (e.g. 60-year-old male)
# ---------------------------------------------------------------------------
print()
print("=" * 84)
print("DiD at a specific profile: 60-year-old male")
print("=" * 84)
did_profile = M.did(
    "group", "preexist_Y",
    group_levels=["A", "B"], condition_levels=[0, 1],
    at={"age": 60, "female": 0},
)
print(did_profile.did)

# ---------------------------------------------------------------------------
# Bonus: plottable table of cell predictions with CIs
# ---------------------------------------------------------------------------
print()
print("=" * 84)
print("Cells with 95% CIs (suitable for a plot)")
print("=" * 84)
tbl = did.cells.summary().copy()
print(tbl)

# If you wanted to plot:
#   import matplotlib.pyplot as plt
#   fig, ax = plt.subplots()
#   for g in ["A", "B"]:
#       sub = tbl[tbl.index.str.contains(f"group={g}")]
#       ax.errorbar([0, 1],
#                   sub["prediction"].values,
#                   yerr=(sub["prediction"] - sub["[95% CI lo]"]).values,
#                   marker="o", label=f"group {g}", capsize=4)
#   ax.set_xticks([0, 1]); ax.set_xticklabels(["no Y", "with Y"])
#   ax.set_ylabel("P(condition X)"); ax.legend(); plt.show()
