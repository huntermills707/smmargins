import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pytest
from smmargins import Margins

SEED = 12345
N = 2000


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def sim_frame(rng):
    """Standard simulated DataFrame (x1, x2, grp, dummy) used across tests."""
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, N),
        "x2": rng.normal(0, 1, N),
        "grp": rng.choice(["a", "b", "c"], N),
        "dummy": rng.integers(0, 2, N),
    })
    # Pre-generate outcomes for different models
    df["y_ols"] = (
        1.0 + 0.5 * df["x1"] - 0.25 * df["x1"] ** 2 + 0.3 * df["x2"]
        + 0.4 * df["x1"] * df["x2"] + rng.normal(0, 0.3, N)
    )
    z_logit = (
        -0.5 + 0.8 * df["x1"] - 0.6 * df["x2"]
        + 0.4 * (df["grp"] == "b").astype(int)
        - 0.3 * (df["grp"] == "c").astype(int)
    )
    df["y_logit"] = (rng.uniform(0, 1, N) < 1 / (1 + np.exp(-z_logit))).astype(int)
    df["count"] = rng.poisson(np.exp(0.5 + 0.4 * df["x1"] - 0.2 * df["x2"]))
    return df


@pytest.fixture(scope="session")
def ols_fit(sim_frame):
    return smf.ols("y_ols ~ x1 + I(x1**2) + x2 + x1:x2", data=sim_frame).fit()


@pytest.fixture(scope="session")
def logit_fit(sim_frame):
    return smf.logit("y_logit ~ x1 + x2 + C(grp)", data=sim_frame).fit(disp=False)


@pytest.fixture(scope="session")
def poisson_fit(sim_frame):
    return smf.glm("count ~ x1 + x2", data=sim_frame,
                   family=sm.families.Poisson()).fit()


@pytest.fixture(scope="session")
def mnlogit_frame(rng):
    """Data frame for a 3-class MNLogit."""
    n = 500
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
    })
    eta = np.column_stack([
        np.zeros(n),
        0.5 + df["x1"] - 0.3 * df["x2"],
        -0.2 + 0.4 * df["x1"] + 0.1 * df["x2"],
    ])
    probs = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    df["y_mnl"] = np.array([rng.choice([0, 1, 2], p=p) for p in probs])
    return df


@pytest.fixture(scope="session")
def mnlogit_fit(mnlogit_frame):
    return sm.MNLogit(mnlogit_frame["y_mnl"],
                      sm.add_constant(mnlogit_frame[["x1", "x2"]])).fit(disp=False)


@pytest.fixture(scope="session")
def ordered_frame(rng):
    """Data frame for a 4-category ordered logit."""
    n = 500
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
    })
    eta = 0.5 + 0.3 * df["x1"] - 0.2 * df["x2"]
    thresholds = [-0.5, 0.0, 0.5]
    probs = np.zeros((n, 4))
    probs[:, 0] = 1 / (1 + np.exp(-(thresholds[0] - eta)))
    for k in range(1, 3):
        probs[:, k] = (
            1 / (1 + np.exp(-(thresholds[k] - eta)))
            - 1 / (1 + np.exp(-(thresholds[k - 1] - eta)))
        )
    probs[:, 3] = 1 - probs[:, :3].sum(axis=1)
    df["y_ord"] = np.array([rng.choice([0, 1, 2, 3], p=p) for p in probs])
    return df


@pytest.fixture(scope="session")
def ordered_fit(ordered_frame):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    return OrderedModel(
        ordered_frame["y_ord"], ordered_frame[["x1", "x2"]], distr="logit"
    ).fit(disp=False)


@pytest.fixture(scope="function")
def did_frame(rng):
    """The (treat, post, x) frame used in DiD tests."""
    N_did = 4000
    df = pd.DataFrame({
        "treat": rng.integers(0, 2, N_did),
        "post": rng.integers(0, 2, N_did),
        "x": rng.normal(0, 1, N_did),
    })
    df["y_lin"] = (
        2.0 + 1.5 * df["treat"] + 0.8 * df["post"]
        - 0.7 * df["treat"] * df["post"] + 0.3 * df["x"]
        + rng.normal(0, 1.0, N_did)
    )
    eta = (
        -1.0 + 1.2 * df["treat"] + 0.9 * df["post"]
        - 0.6 * df["treat"] * df["post"] + 0.5 * df["x"]
    )
    df["y_bin"] = (rng.uniform(0, 1, N_did) < 1 / (1 + np.exp(-eta))).astype(int)
    return df
