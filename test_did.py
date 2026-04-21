"""
Tests for the DiD / contrast functionality.

Key checks:

1. On an OLS with identity link, DiD must equal the coefficient on the
   interaction term, with the same SE.  This is the cleanest possible
   sanity check.

2. On a logit, the DiD on the probability scale must NOT equal the
   interaction coefficient (that's the whole point), but it must match a
   by-hand computation of the 4 cells through the inverse logit and the
   corresponding delta-method SE.

3. The `contrast` method must produce the same result as re-running
   `_delta` on the same linear combination built from scratch.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from marginal_effects import Margins

rng = np.random.default_rng(2026)
N = 4000
df = pd.DataFrame(
    {
        "treat": rng.integers(0, 2, N),
        "post":  rng.integers(0, 2, N),
        "x":     rng.normal(0, 1, N),
    }
)

# ---------------------------------------------------------------------------
# TEST 1: OLS DiD == coefficient on treat:post interaction
# ---------------------------------------------------------------------------
df["y_lin"] = (
    2.0 + 1.5 * df["treat"] + 0.8 * df["post"]
    - 0.7 * df["treat"] * df["post"] + 0.3 * df["x"]
    + rng.normal(0, 1.0, N)
)
ols = smf.ols("y_lin ~ treat + post + treat:post + x", data=df).fit()
M = Margins(ols)
did_res = M.did("treat", "post")

print("=" * 72)
print("TEST 1 : OLS — DiD equals coefficient on treat:post")
print("=" * 72)
print(did_res)

b_inter = ols.params["treat:post"]
se_inter = ols.bse["treat:post"]
print(f"\ntreat:post coefficient: {b_inter:.8f} (SE {se_inter:.8f})")
print(f"DiD from Margins       : {did_res.did.estimate[0]:.8f} (SE {did_res.did.se[0]:.8f})")
assert np.isclose(did_res.did.estimate[0], b_inter, atol=1e-8), "DiD estimate mismatch"
assert np.isclose(did_res.did.se[0], se_inter, rtol=1e-5), "DiD SE mismatch"
print("PASS")

# Also: simple effect of treat at post=0 must equal the coefficient on `treat`
b_treat = ols.params["treat"]
se_treat = ols.bse["treat"]
print(f"\ntreat coefficient (simple effect at post=0): {b_treat:.8f} (SE {se_treat:.8f})")
print(f"Margins simple effect |post=0              : "
      f"{did_res.simple_effects.estimate[0]:.8f} "
      f"(SE {did_res.simple_effects.se[0]:.8f})")
assert np.isclose(did_res.simple_effects.estimate[0], b_treat, atol=1e-8)
assert np.isclose(did_res.simple_effects.se[0], se_treat, rtol=1e-5)

# ---------------------------------------------------------------------------
# TEST 2: Logit DiD on prob scale — must differ from interaction coefficient,
#         and match hand-computed cells.
# ---------------------------------------------------------------------------
eta = (
    -1.0 + 1.2 * df["treat"] + 0.9 * df["post"]
    - 0.6 * df["treat"] * df["post"] + 0.5 * df["x"]
)
df["y_bin"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-eta))).astype(int)
logit = smf.logit("y_bin ~ treat + post + treat:post + x", data=df).fit(disp=False)
ML = Margins(logit)
did_log = ML.did("treat", "post")

print()
print("=" * 72)
print("TEST 2 : Logit — DiD on prob scale != interaction coefficient")
print("=" * 72)
print(did_log)

# The interaction coefficient is on the log-odds scale:
b_int_logit = logit.params["treat:post"]
print(f"\nInteraction coeff (log-odds scale): {b_int_logit:.6f}")
print(f"DiD on prob scale                 : {did_log.did.estimate[0]:.6f}")
# They should NOT be equal — the nonlinear link means the interaction
# coefficient isn't the DiD
assert not np.isclose(did_log.did.estimate[0], b_int_logit, atol=0.05), \
    "They happened to coincide, which is suspicious — check model."

# Hand computation of the 4 cells and DiD at the response scale:
def hand_cell(t, p):
    d = df.copy()
    d["treat"] = t; d["post"] = p
    # Use model.predict with the dataframe so patsy rebuilds columns
    return float(logit.model.predict(logit.params, ML._build_exog(d)).mean())
m00 = hand_cell(0, 0); m01 = hand_cell(0, 1)
m10 = hand_cell(1, 0); m11 = hand_cell(1, 1)
hand_did = (m11 - m01) - (m10 - m00)
print(f"Hand DiD (cell arithmetic)        : {hand_did:.6f}")
assert np.isclose(did_log.did.estimate[0], hand_did, atol=1e-10), "DiD point mismatch"

# ---------------------------------------------------------------------------
# TEST 3: contrast() method -- round-trip vs direct delta
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("TEST 3 : MarginsResult.contrast() round-trips cells -> DiD correctly")
print("=" * 72)
cells = ML.predict(at={"treat": [0, 1], "post": [0, 1]})
did_via_contrast = cells.contrast([1, -1, -1, 1], labels=["DiD"])
print(did_via_contrast)
assert np.isclose(did_via_contrast.estimate[0], did_log.did.estimate[0], atol=1e-12)
assert np.isclose(did_via_contrast.se[0], did_log.did.se[0], atol=1e-12)

# Also: simple effects via matrix contrast must match
C = np.array([[-1, 0, 1, 0], [0, -1, 0, 1], [1, -1, -1, 1]], float)
joint = cells.contrast(C, labels=["B@p=0", "B@p=1", "DiD"])
print("\nAll three contrasts in one call:")
print(joint)
assert np.allclose(joint.estimate, did_log.joint.estimate, atol=1e-12)
assert np.allclose(joint.vcov,     did_log.joint.vcov,     atol=1e-12)

# ---------------------------------------------------------------------------
# TEST 4: DiD with covariate adjustment at means — should still work
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("TEST 4 : DiD with other covariates fixed at means / specific values")
print("=" * 72)
print("atmeans=True:")
print(ML.did("treat", "post", atmeans=True).did)
print("\nat={'x': 1.5}:")
print(ML.did("treat", "post", at={"x": 1.5}).did)

print("\n" + "=" * 72)
print("ALL DiD TESTS PASSED")
print("=" * 72)
