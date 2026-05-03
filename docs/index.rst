smmargins
=========

Stata-style ``margins`` for `StatsModels
<https://www.statsmodels.org/>`_: adjusted predictions, marginal
effects, elasticities, and difference-in-differences — with
delta-method, Krinsky–Robb simulation, or bootstrap standard errors,
robust covariance passthrough (HC0–HC3, cluster, HAC), and
simultaneous confidence intervals (Bonferroni, Šidák, sup-t) — for any
fitted model that exposes ``params``, ``cov_params()``, and
``predict(params, exog)``.

.. toctree::
    :maxdepth: 2
    :caption: Getting started

    intro

.. toctree::
    :maxdepth: 1
    :caption: Tutorials (executable notebooks)

    tutorials/plotting

.. toctree::
    :maxdepth: 1
    :caption: How-to recipes

    cookbook_values
    cookbook_transforms
    cookbook_newdata
    cookbook_contrast
    cookbook_plotting
    demos

.. toctree::
    :maxdepth: 2
    :caption: Reference

    api

.. toctree::
    :maxdepth: 1
    :caption: Explanations (theory & design)

    math
    explanations/contrast_joint_covariance

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
