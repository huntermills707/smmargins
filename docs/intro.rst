Introduction
============

``smmargins`` is a small module that fills in the marginal-effects gaps
in `StatsModels <https://www.statsmodels.org/>`_: adjusted predictions
and marginal effects at user-specified covariate profiles, with
delta-method standard errors, for *any* fitted model that exposes
``params``, ``cov_params()``, and a ``predict(params, exog)`` method.

The design target is `Stata's
<https://www.stata.com/manuals/rmargins.pdf>`_ ``margins`` command:
the same statistics, the same parameter names where they translate, and
the same answers to the precision both tools agree on.

Why another margins module?
---------------------------

StatsModels ships ``Results.get_margeff``, but it is limited:

- only marginal effects, not adjusted predictions;
- ``atexog`` is keyed by *column index*, not variable name;
- no ``at(...)`` profiles, no representative-value contrasts;
- no joint covariance across statistics, so you cannot form contrasts
  like a difference-in-differences without re-deriving the delta
  method by hand;
- no support for difference-in-differences on the response scale (the
  Ai & Norton 2003 issue).

Installation
------------

.. code-block:: bash

    pip install smmargins

Requires Python ≥3.9. Dependencies (``numpy``, ``pandas``,
``statsmodels``, ``scipy``, ``patsy``) are installed automatically.

Quickstart
----------

.. code-block:: python

    import statsmodels.formula.api as smf
    from smmargins import Margins

    fit = smf.logit(
        "voted ~ age + income + C(educ) + female + age:female",
        data=df,
    ).fit()
    M = Margins(fit)

    # Adjusted predictions
    M.predict()                                    # AAP
    M.predict(at="mean")                           # APM (margins, atmeans)
    M.predict(atexog={"age": [25, 45, 65]})        # APR

    # Marginal effects on the response (probability) scale
    M.dydx("age")                                  # AME
    M.dydx("age", at="mean")                       # MEM
    M.dydx("age", atexog={"female": [0, 1]})       # MER, by sex
    M.dydx("educ", reference="college")            # discrete contrasts

    # Difference-in-differences on the response scale
    res = M.did("group", "preexist_Y",
                group_levels=["A", "B"], condition_levels=[0, 1])
    print(res)                                     # cells, simple effects, DiD

    # Alternative VCEs and robust covariance
    M.dydx("age", cov_type="HC3")                  # heteroskedastic-robust
    M.dydx("age", vce="simulation", n_sims=2000,
           sim_seed=0)                             # Krinsky–Robb
    M.dydx("age", vce="bootstrap", n_boot=1000,
           boot_seed=0)                            # pairs bootstrap

    # Simultaneous CIs across a family of three predictions
    M.predict(atexog={"age": [25, 45, 65]},
              vce="simulation", n_sims=4000,
              ci_method="sup-t")

Each call returns a :class:`~smmargins.MarginsResult` with ``.estimate``,
``.se``, ``.vcov``, ``.ci_lower``, ``.ci_upper``, ``.pvalue``, plus
``.summary()`` returning a :class:`pandas.DataFrame`. Pass ``use_t=True``
to the :class:`~smmargins.Margins` constructor for t-distribution
inference (uses ``results.df_resid``).

Observation weights
-------------------

Pass weights at construction. ``weight_type="sampling"`` (default) or
``"frequency"``. WLS-fitted results respect their fit-time weights when
``weights=None``; explicit ``weights=`` overrides and warns. Bootstrap
resampling uses weight-proportional draws under sampling weights.

.. code-block:: python

    M = Margins(fit, weights=w)
    M = Margins(fit, weights=counts, weight_type="frequency")

Where to next
-------------

**Tutorials — learn by doing.**

- :doc:`tutorials/getting_started` — install, fit, first AAP and AME.
- :doc:`tutorials/adjusted_predictions` — ``at=``, ``atexog=``, scales.
- :doc:`tutorials/marginal_effects` — continuous, discrete, elasticities.
- :doc:`tutorials/inference` — delta vs KR vs bootstrap, simultaneous CIs.
- :doc:`tutorials/did` — 2x2 difference-in-differences end-to-end.
- :doc:`tutorials/counterfactuals_and_plotting` — ``newdata=``, ``Expr``,
  the plotting API.

**How-to guides — task-focused recipes.**

- Robust covariance: :doc:`howto/robust_clustered_ses`.
- Alternative VCEs: :doc:`howto/kr_simulation`, :doc:`howto/bootstrap`,
  :doc:`howto/simultaneous_ci`.
- Multi-outcome models: :doc:`howto/multi_outcome_models`.
- Subgroup AMEs and joint tests: :doc:`howto/subgroup_analysis`,
  :doc:`howto/joint_tests_pairwise`.
- Custom scales: :doc:`howto/custom_transforms`,
  :doc:`howto/elasticities`.
- Counterfactuals and plotting: :doc:`howto/counterfactual_predictions`,
  :doc:`howto/expr_and_values`, :doc:`howto/plotting`.
- Validation and modes: :doc:`howto/verify_against_r`,
  :doc:`howto/formula_vs_raw`.

**Explanations — theory and design.**

- :doc:`math` — delta method, statistic schema, analytic vs FD Jacobian.
- :doc:`explanations/scales_link_functions`,
  :doc:`explanations/discrete_vs_continuous`,
  :doc:`explanations/outer_jacobian`.
- :doc:`explanations/patsy_design_rebuilding`,
  :doc:`explanations/formula_vs_raw_mode`,
  :doc:`explanations/expr_and_values`.
- :doc:`explanations/inference_comparison`,
  :doc:`explanations/multiple_comparisons`,
  :doc:`explanations/contrast_joint_covariance`.
- :doc:`explanations/ai_norton_did` — why DiD belongs on the response
  scale.

**Reference.**

- :doc:`api` — every public class and method.
- :doc:`demos` — full Williams-style and DiD walkthroughs from the
  repository root.
