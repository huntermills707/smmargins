"""Generate shared data CSVs for R comparison tests.

Run this script before generate_r.R so both Python and R use identical data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset A: seed 42, n=2000 — used by logit_ame_hc3, poisson_irr,
# logit_cluster, logit_at
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 2000

df_a = pd.DataFrame({
    "x1": rng.normal(0, 1, n),
    "x2": rng.normal(0, 1, n),
    "grp": rng.choice([0, 1, 2], n),
})
z_logit = (
    -0.5 + 0.8 * df_a["x1"] - 0.6 * df_a["x2"]
    + 0.4 * (df_a["grp"] == 1).astype(int)
    - 0.3 * (df_a["grp"] == 2).astype(int)
)
df_a["y_logit"] = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-z_logit))).astype(int)
df_a["count"] = rng.poisson(np.exp(0.5 + 0.4 * df_a["x1"] - 0.2 * df_a["x2"]))
df_a["clust"] = rng.integers(0, 50, n)
df_a.to_csv(DATA_DIR / "data_seed42.csv", index=False)

# ---------------------------------------------------------------------------
# Dataset B: seed 12345, n=2000 — used by ols_hc1, ols_bootstrap
# ---------------------------------------------------------------------------
rng = np.random.default_rng(12345)
N = 2000

df_b = pd.DataFrame({
    "x1": rng.normal(0, 1, N),
    "x2": rng.normal(0, 1, N),
    "grp": rng.choice(["a", "b", "c"], N),
    "dummy": rng.integers(0, 2, N),
})
df_b["y_ols"] = (
    1.0 + 0.5 * df_b["x1"] - 0.25 * df_b["x1"] ** 2
    + 0.3 * df_b["x2"] + 0.4 * df_b["x1"] * df_b["x2"]
    + rng.normal(0, 0.3, N)
)
df_b.to_csv(DATA_DIR / "data_seed12345.csv", index=False)

print(f"Data files written to {DATA_DIR}")
