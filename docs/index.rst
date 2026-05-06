smmargins
=========

Stata-style ``margins`` for `StatsModels <https://www.statsmodels.org/>`_:
adjusted predictions, marginal effects, elasticities, and
difference-in-differences — with delta-method, Krinsky–Robb simulation,
or bootstrap standard errors, robust covariance passthrough (HC0–HC3,
cluster, HAC), and simultaneous confidence intervals (Bonferroni, Šidák,
sup-t) — for any fitted model that exposes ``params``, ``cov_params()``,
and ``predict(params, exog)``.

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    intro

.. toctree::
    :maxdepth: 1
    :caption: Tutorials — learning by doing

    tutorials/getting_started
    tutorials/adjusted_predictions
    tutorials/marginal_effects
    tutorials/inference
    tutorials/did
    tutorials/counterfactuals_and_plotting

.. toctree::
    :maxdepth: 1
    :caption: How-to guides — task-focused recipes

    howto/robust_clustered_ses
    howto/kr_simulation
    howto/bootstrap
    howto/simultaneous_ci
    howto/custom_transforms
    howto/subgroup_analysis
    howto/joint_tests_pairwise
    howto/counterfactual_predictions
    howto/expr_and_values
    howto/elasticities
    howto/multi_outcome_models
    howto/plotting
    howto/verify_against_r
    howto/formula_vs_raw

.. toctree::
    :maxdepth: 2
    :caption: Reference

    api
    demos

.. toctree::
    :maxdepth: 1
    :caption: Explanations — theory and design

    math
    explanations/expr_and_values
    explanations/outer_jacobian
    explanations/patsy_design_rebuilding
    explanations/scales_link_functions
    explanations/ai_norton_did
    explanations/inference_comparison
    explanations/multiple_comparisons
    explanations/discrete_vs_continuous
    explanations/formula_vs_raw_mode
    explanations/contrast_joint_covariance

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
