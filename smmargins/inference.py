from __future__ import annotations

import inspect
import warnings
from typing import Callable, Optional

import numpy as np


def _summarize_draws(draws: np.ndarray, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical covariance and standard errors from draws.

    Parameters
    ----------
    draws : ndarray of shape (n_draws, m)
        Matrix of simulated or bootstrap margin values.
    ddof : int, default 1
        Delta degrees of freedom for variance/covariance.

    Returns
    -------
    tuple (vcov, se)
        vcov : ndarray of shape (m, m)
        se   : ndarray of shape (m,)
    """
    if draws.shape[1] == 1:
        vcov = np.array([[np.var(draws, ddof=ddof)]])
    else:
        vcov = np.cov(draws, rowvar=False, ddof=ddof)
    se = np.std(draws, axis=0, ddof=ddof)
    return vcov, se


def _simulate_vce(
    beta: np.ndarray,
    param_cov: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    n_sims: int = 2000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Krinsky–Robb simulation VCE.

    Draws parameter vectors from a multivariate normal and evaluates
    the margin statistic for each draw.

    Parameters
    ----------
    beta : ndarray of shape (p,)
        Point estimate of parameters.
    param_cov : ndarray of shape (p, p)
        Covariance matrix of the parameter estimates.
    statistic : callable
        Function mapping a parameter vector to a vector of margin values.
    n_sims : int, default 2000
        Number of simulation draws.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_sims, m)
        Matrix of simulated margin values.
    """
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(beta, param_cov, size=n_sims)
    results = []
    for b in draws:
        val = np.atleast_1d(np.asarray(statistic(b), dtype=float)).ravel()
        results.append(val)
    return np.vstack(results)


def _refit_model(results, idx):
    """Refit a statsmodels model on a bootstrap subsample.

    Parameters
    ----------
    results : statsmodels results object
        The original fitted results.
    idx : ndarray of int
        Bootstrap indices.

    Returns
    -------
    statsmodels results object
        Refitted results on the bootstrap sample.
    """
    model = results.model
    y = np.asarray(model.endog)
    X = np.asarray(model.exog)

    # Extract constructor kwargs generically, filtering to accepted params
    sig = inspect.signature(type(model).__init__)
    accepted = set(sig.parameters.keys())

    kwargs = {}
    for name in list(sig.parameters.keys()):
        if name in ("self", "endog", "exog", "data", "formula"):
            continue
        val = getattr(model, name, None)
        if val is not None:
            kwargs[name] = val

    # Also check common attributes that might not be in __init__ signature
    for attr in ("family", "distr", "offset", "exposure"):
        if attr not in kwargs and attr in accepted:
            val = getattr(model, attr, None)
            if val is not None:
                kwargs[attr] = val

    # Subset array-like kwargs that align with observations
    for k, v in list(kwargs.items()):
        if isinstance(v, np.ndarray) and v.shape[0] == y.shape[0]:
            kwargs[k] = v[idx]

    # Formula path: rebuild data frame and use from_formula
    formula_err = None
    if hasattr(model, "formula") and getattr(model, "formula", None) is not None:
        try:
            df = model.data.frame.iloc[idx].copy()
            # from_formula may have a different signature; be conservative
            model_b = type(model).from_formula(model.formula, data=df)
            return model_b.fit(disp=False)
        except Exception as exc:
            formula_err = exc

    # Raw matrix path
    y_b = y[idx]
    X_b = X[idx]
    # Only pass kwargs that the constructor accepts
    valid_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    try:
        model_b = type(model)(y_b, X_b, **valid_kwargs)
        return model_b.fit(disp=False)
    except Exception as exc:
        if formula_err is not None:
            warnings.warn(
                f"Bootstrap refit failed on both formula path ({type(formula_err).__name__}) "
                f"and raw path ({type(exc).__name__}).",
                RuntimeWarning,
            )
        raise


def _bootstrap_vce(
    results,
    margin_factory: Callable,
    n_boot: int = 1000,
    seed: Optional[int] = None,
    method: str = "pairs",
    cluster: Optional[np.ndarray] = None,
    block_size: Optional[int] = None,
    verbose: bool = False,
    n_jobs: int = 1,
) -> np.ndarray:
    """Bootstrap VCE for margins.

    Parameters
    ----------
    results : statsmodels results object
        The original fitted results.
    margin_factory : callable
        Function that takes a statsmodels results object and returns a
        ``MarginsResult`` (or any object with an ``.estimate`` attribute).
    n_boot : int, default 1000
        Number of bootstrap replications.
    seed : int, optional
        Random seed.
    method : {"pairs", "cluster", "block"}, default "pairs"
        Bootstrap resampling scheme.
    cluster : ndarray, optional
        Cluster IDs of length n, required for ``method="cluster"``.
    block_size : int, optional
        Block size, required for ``method="block"``.
    verbose : bool, default False
        Show progress bar.
    n_jobs : int, default 1
        Number of parallel jobs. Requires ``joblib``.

    Returns
    -------
    ndarray of shape (n_successful, m)
        Matrix of bootstrap margin estimates.
    """
    rng = np.random.default_rng(seed)
    n = len(results.model.endog)

    if method == "cluster":
        if cluster is None:
            raise ValueError("cluster is required for boot_method='cluster'")
        clusters = np.asarray(cluster)
    if method == "block":
        if block_size is None:
            raise ValueError("block_size is required for boot_method='block'")
        if block_size > n:
            raise ValueError("block_size cannot exceed the number of observations")

    indices = []
    for _ in range(n_boot):
        if method == "pairs":
            idx = rng.integers(0, n, size=n)
        elif method == "cluster":
            unique_cl = np.unique(clusters)
            n_cl = len(unique_cl)
            cl_draw = rng.choice(n_cl, size=n_cl, replace=True)
            drawn = unique_cl[cl_draw]
            idx = np.concatenate([np.where(clusters == c)[0] for c in drawn])
        elif method == "block":
            n_blocks = n // block_size
            blocks = rng.integers(0, n - block_size + 1, size=n_blocks)
            idx = np.concatenate([np.arange(b, b + block_size) for b in blocks])
        else:
            raise ValueError(
                f"boot_method must be 'pairs', 'cluster', or 'block', got {method!r}"
            )
        indices.append(idx)

    def _one(idx):
        try:
            results_b = _refit_model(results, idx)
            margin = margin_factory(results_b)
            return np.atleast_1d(margin.estimate).ravel()
        except Exception:
            return None

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed

            results_list = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(_one)(idx) for idx in indices
            )
        except Exception:
            results_list = [_one(idx) for idx in indices]
    else:
        results_list = []
        iterator = indices
        if verbose:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=n_boot, desc="Bootstrap")
            except ImportError:
                pass
        for idx in iterator:
            results_list.append(_one(idx))

    successful = [r for r in results_list if r is not None]
    n_failed = n_boot - len(successful)
    if n_failed > 0:
        pct = n_failed / n_boot
        msg = f"{n_failed}/{n_boot} bootstrap reps failed ({pct:.1%})"
        if pct > 0.05:
            warnings.warn(msg, RuntimeWarning)

    if len(successful) == 0:
        raise RuntimeError("All bootstrap reps failed")

    # Pad shorter results with NaN to uniform length if needed
    max_len = max(len(r) for r in successful)
    padded = []
    for r in successful:
        if len(r) < max_len:
            padded.append(
                np.concatenate([r, np.full(max_len - len(r), np.nan)])
            )
        else:
            padded.append(r)
    return np.vstack(padded)
