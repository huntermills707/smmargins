"""
Tests: verify delta-method margins against hand-derived truth,
and cross-check AME/MEM vs StatsModels' built-in ``get_margeff``
where applicable.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from marginal_effects import Margins

rng = np.random.default_rng(12345)
N = 2000
df = pd.DataFrame(
    {
        "x1": rng.normal(0, 1, N),
        "x2": rng.normal(0, 1, N),
        "grp": rng.choice(["a", "b", "c"], N),
        "dummy": rng.integers(0, 2, N),
    }
)

# ---------------------------------------------------------------------------
# 1. OLS with interaction and polynomial: verify delta-method against
#    hand-derived analytic AME, APR.
# ---------------------------------------------------------------------------
print("=" * 72)
print("TEST 1 : OLS with interaction + I(x1**2)")
print("=" * 72)
df["y"] = (
    1.0 + 0.5 * df["x1"] - 0.25 * df["x1"] ** 2 + 0.3 * df["x2"]
    + 0.4 * df["x1"] * df["x2"]
    + rng.normal(0, 0.3, N)
)
ols = smf.ols("y ~ x1 + I(x1**2) + x2 + x1:x2", data=df).fit()
M = Margins(ols, use_t=True)

# AME for x1 : d/dx1 [b0 + b1 x1 + b2 x1^2 + b3 x2 + b4 x1 x2]
#            = b1 + 2 b2 x1 + b4 x2
#   -> AME  = b1 + 2 b2 mean(x1) + b4 mean(x2)
p = ols.params
hand_ame = (p["x1"] + 2 * p["I(x1 ** 2)"] * df["x1"].mean()
            + p["x1:x2"] * df["x2"].mean())
# Gradient wrt beta = [0, 1, 2*mean(x1), 0, mean(x2)]  (order: Intercept,
#  x1, I(x1**2), x2, x1:x2). We just use delta internally — verify match.
ame_x1 = M.dydx("x1")
print("AME(x1) hand-computed :", hand_ame)
print(ame_x1)
assert np.isclose(ame_x1.estimate[0], hand_ame, atol=1e-7), "AME mismatch"

# Verify SE against analytic form
# SE^2 = g' V g where g = [0, 1, 2*mean(x1), 0, mean(x2)]
names = list(ols.params.index)
g = np.zeros(len(names))
g[names.index("x1")] = 1.0
g[names.index("I(x1 ** 2)")] = 2 * df["x1"].mean()
g[names.index("x1:x2")] = df["x2"].mean()
V = np.asarray(ols.cov_params())
hand_se = np.sqrt(g @ V @ g)
print(f"AME(x1) hand SE        : {hand_se:.8f}")
print(f"AME(x1) delta-method SE: {ame_x1.se[0]:.8f}")
assert np.isclose(ame_x1.se[0], hand_se, rtol=1e-5), "AME SE mismatch"

# ---------------------------------------------------------------------------
# 2. APR: predictions at representative values
# ---------------------------------------------------------------------------
print()
print("-" * 72)
print("APR: predict at x1 in {-1, 0, 1}, averaging over x2 as observed")
print("-" * 72)
apr = M.predict(at={"x1": [-1.0, 0.0, 1.0]})
print(apr)

# Hand compute first row: predict at x1=-1, x2=x2_i, average.
b = ols.params
i, a, b1, b2, b12 = (b["Intercept"], b["I(x1 ** 2)"],
                    b["x1"], b["x2"], b["x1:x2"])
x1v = -1.0
pred_hand = (i + b1 * x1v + a * x1v ** 2
             + b2 * df["x2"].mean() + b12 * x1v * df["x2"].mean())
print("First row hand-computed:", pred_hand)
assert np.isclose(apr.estimate[0], pred_hand, atol=1e-7)

# ---------------------------------------------------------------------------
# 3. Logit: AME / MEM vs StatsModels' built-in get_margeff
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("TEST 2 : Logit AME and MEM vs statsmodels.get_margeff")
print("=" * 72)
df["z"] = (
    -0.5 + 0.8 * df["x1"] - 0.6 * df["x2"]
    + 0.4 * (df["grp"] == "b").astype(int)
    - 0.3 * (df["grp"] == "c").astype(int)
)
df["yb"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-df["z"]))).astype(int)

logit = smf.logit("yb ~ x1 + x2 + C(grp)", data=df).fit(disp=False)
ML = Margins(logit)

ame = ML.dydx("x1")
print("delta-method AME(x1):")
print(ame)

# Cross-check with statsmodels get_margeff (AME = at='overall', effects are dp/dx
# for continuous, average discrete for dummy)
mfx_overall = logit.get_margeff(at="overall", method="dydx")
print("\nstatsmodels get_margeff (overall):")
print(mfx_overall.summary())

# Extract x1 row from statsmodels output
sm_x1_idx = list(logit.params.index).index("x1")
# get_margeff returns effects in order of exog columns (dropping intercept)
exog_names = logit.model.exog_names
# position of x1 in exog minus the intercept offset
x1_pos = [n for n in exog_names if n != "Intercept"].index("x1")
sm_x1_me = mfx_overall.margeff[x1_pos]
sm_x1_se = mfx_overall.margeff_se[x1_pos]
print(f"\nx1 from statsmodels: me={sm_x1_me:.8f}  se={sm_x1_se:.8f}")
print(f"x1 from delta      : me={ame.estimate[0]:.8f}  se={ame.se[0]:.8f}")
assert np.isclose(ame.estimate[0], sm_x1_me, atol=1e-5)
assert np.isclose(ame.se[0], sm_x1_se, rtol=1e-3)

# MEM: evaluate at means.  Default factor_stat="mean" matches Stata /
# statsmodels: design-matrix means (i.e. observed proportions for dummies).
mem_x1 = ML.dydx("x1", atmeans=True)
mfx_mean = logit.get_margeff(at="mean", method="dydx")
sm_x1_mem = mfx_mean.margeff[x1_pos]
sm_x1_mem_se = mfx_mean.margeff_se[x1_pos]
print(f"\nMEM(x1) statsmodels: {sm_x1_mem:.8f} (se {sm_x1_mem_se:.8f})")
print(f"MEM(x1) delta      : {mem_x1.estimate[0]:.8f} (se {mem_x1.se[0]:.8f})")
assert np.isclose(mem_x1.estimate[0], sm_x1_mem, atol=1e-5)
assert np.isclose(mem_x1.se[0], sm_x1_mem_se, rtol=1e-3)

# factor_stat="mode" path: factors held at their modal level. This is the
# old behavior and should differ from statsmodels whenever a non-modal
# factor level has nontrivial proportion. Sanity-check it against a hand
# computation on the modal factor row.
mem_x1_mode = ML.dydx("x1", atmeans=True, factor_stat="mode")
modal_grp = df["grp"].mode().iloc[0]
row = pd.DataFrame([{
    "x1": df["x1"].mean(), "x2": df["x2"].mean(),
    "grp": modal_grp, "dummy": df["dummy"].mean(),
}])
h = (np.finfo(float).eps ** (1.0 / 3.0)) * max(
    df["x1"].std(ddof=0), abs(df["x1"].mean()), 1.0
)
rp = row.copy(); rp["x1"] = rp["x1"] + h
rm = row.copy(); rm["x1"] = rm["x1"] - h
Xp = ML._build_exog(rp); Xm = ML._build_exog(rm)
hand_mode = float(
    (logit.model.predict(logit.params, Xp)[0]
     - logit.model.predict(logit.params, Xm)[0]) / (2.0 * h)
)
print(f"MEM(x1, mode) hand : {hand_mode:.8f}")
print(f"MEM(x1, mode) delta: {mem_x1_mode.estimate[0]:.8f}")
assert np.isclose(mem_x1_mode.estimate[0], hand_mode, atol=1e-6)

# ---------------------------------------------------------------------------
# 4. Discrete contrast for factor variable
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("TEST 3 : discrete contrast for C(grp)  — AME for factor")
print("=" * 72)
grp_effects = ML.dydx("grp")
print(grp_effects)
# Sanity: statsmodels reports these too — we don't cross-check exact SEs here
# because statsmodels treats dummies differently, but the point estimate
# should match the average first-difference.

# Hand compute the b-vs-a contrast on the original factor
def avg_predict_at_grp(level):
    d = df.copy(); d["grp"] = level
    X = Margins(logit)._build_exog(d)
    return float(np.mean(logit.model.predict(logit.params, X)))
hand_b_vs_a = avg_predict_at_grp("b") - avg_predict_at_grp("a")
print(f"Hand b-vs-a avg discrete change: {hand_b_vs_a:.8f}")
assert np.isclose(grp_effects.estimate[0], hand_b_vs_a, atol=1e-8)

# ---------------------------------------------------------------------------
# 5. AAP and APM for a GLM Poisson — sanity check link handling
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("TEST 4 : Poisson AAP (exp link)")
print("=" * 72)
df["count"] = rng.poisson(np.exp(0.5 + 0.4 * df["x1"] - 0.2 * df["x2"]))
pois = smf.glm("count ~ x1 + x2", data=df,
               family=sm.families.Poisson()).fit()
MP = Margins(pois)
aap = MP.predict()
# Hand: mean(exp(Xb))
X = pois.model.exog
eta = X @ pois.params
hand_aap = float(np.mean(np.exp(eta)))
print(f"AAP (delta) : {aap.estimate[0]:.8f}  se {aap.se[0]:.8f}")
print(f"AAP (hand)  : {hand_aap:.8f}")
assert np.isclose(aap.estimate[0], hand_aap, atol=1e-10)
# Also: AAP should equal sample mean of count in Poisson with canonical link
print(f"sample mean : {df['count'].mean():.8f}  (Poisson+log AAP equals sample mean)")

# Hand SE for AAP: gradient = mean_i(mu_i * x_i),  Var = g' V g
mu = np.exp(eta)
g = (mu[:, None] * X).mean(axis=0)
V = np.asarray(pois.cov_params())
hand_aap_se = np.sqrt(g @ V @ g)
print(f"AAP SE (hand): {hand_aap_se:.8f}")
assert np.isclose(aap.se[0], hand_aap_se, rtol=1e-5)

print("\n" + "=" * 72)
print("ALL TESTS PASSED")
print("=" * 72)
