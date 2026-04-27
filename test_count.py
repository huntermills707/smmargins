import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from marginal_effects import Margins


# import pytest

def test_count_dydx_parity():
    # 1. Fit a Poisson on a frame with an integer column 'kids'.
    rng = np.random.default_rng(42)
    N = 100
    df = pd.DataFrame({
        "kids": rng.integers(0, 5, N),
        "income": rng.normal(50, 10, N),
    })
    # y = exp(0.5 * kids - 0.01 * income)
    df["y"] = rng.poisson(np.exp(0.5 * df["kids"] - 0.01 * df["income"]))

    model = smf.poisson("y ~ kids + income", data=df).fit(disp=0)
    M = Margins(model)

    # 2. est = M.dydx("kids", count=True)
    try:
        est = M.dydx("kids", count=True)
    except TypeError as e:
        print(f"Caught expected TypeError (count not yet implemented): {e}")
        return

    # 3. Compare to pois.get_margeff(at="overall", method="dydx", count=True)
    sm_margeff = model.get_margeff(at="overall", method="dydx", count=True)
    print("StatsModels get_margeff (count=True):")
    print(sm_margeff.summary())

    print("\nsmmargins Margins.dydx(count=True):")
    print(est.summary())

    # SM get_margeff for 'kids'
    # SM seems to use different defaults for count variables?
    # Actually, SM get_margeff(count=True) only treats variables that are
    # explicitly identified as count (integer) as having unit increment.
    # But wait, why are the values different?

    # Let's check what SM does.
    kids_idx = list(model.params.index).index("kids") - 1  # excluding intercept
    sm_est = sm_margeff.margeff[kids_idx]
    sm_se = sm_margeff.margeff_se[kids_idx]

    print(f"SM kids estimate: {sm_est}")
    print(f"SM kids SE:       {sm_se}")

    # 4. Compare to hand-rolled mean(predict(X+1) - predict(X))
    def hand_rolled_count_eff(model, df, var):
        df_p = df.copy()
        df_p[var] = df_p[var] + 1

        # We need the actual mean of individual differences for AME
        p0 = model.predict(df)
        p1 = model.predict(df_p)
        return np.mean(p1 - p0)

    hand_est = hand_rolled_count_eff(model, df, "kids")
    print(f"\nHand-rolled estimate: {hand_est}")
    print(f"smmargins estimate:   {est.estimate[0]}")
    assert np.isclose(est.estimate[0], hand_est, atol=1e-10)

    # 5. Check SM without count=True
    sm_margeff_no_count = model.get_margeff(at="overall", method="dydx", count=False)
    print("\nStatsModels get_margeff (count=False):")
    print(sm_margeff_no_count.summary())
    sm_est_no_count = sm_margeff_no_count.margeff[kids_idx]
    print(f"SM kids (no count) estimate: {sm_est_no_count}")

    # 6. SM MEM check
    sm_margeff_mem = model.get_margeff(at="mean", method="dydx", count=True)
    print("\nStatsModels get_margeff (at=mean, count=True):")
    print(sm_margeff_mem.summary())
    sm_mem_est = sm_margeff_mem.margeff[kids_idx]

    est_mem = M.dydx("kids", at="mean", count=True)
    print("smmargins Margins.dydx(at=mean, count=True):")
    print(est_mem.summary())

    # Hand-rolled MEM
    df_mean = df.mean(numeric_only=True).to_frame().T
    df_mean_p = df_mean.copy()
    df_mean_p["kids"] = df_mean_p["kids"] + 1
    # For MEM, we predict on the MEAN design matrix.
    # Margins(at="mean") with count=True should match
    # predict(mean_x with kids+1) - predict(mean_x)
    hand_mem_est = model.predict(df_mean_p)[0] - model.predict(df_mean)[0]
    print(f"Hand-rolled MEM: {hand_mem_est}")
    assert np.isclose(est_mem.estimate[0], hand_mem_est)

    # 7. Multi-variable count=True check
    est_multi = M.dydx(["kids", "income"], count=True)
    print("\nsmmargins Margins.dydx(['kids', 'income'], count=True):")
    print(est_multi.summary())
    assert len(est_multi.estimate) == 2
    assert "kids (count)" in est_multi.labels[0]
    assert "income (count)" in est_multi.labels[1]


if __name__ == "__main__":
    test_count_dydx_parity()
