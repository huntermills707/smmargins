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

``smmargins`` provides:

- :meth:`~smmargins.Margins.predict` — adjusted predictions
  (AAP / APM / APR), with ``at=`` and name-keyed ``atexog=``;
- :meth:`~smmargins.Margins.dydx` — marginal effects (AME / MEM / MER),
  continuous and discrete, including elasticities (``eyex`` / ``dyex``
  / ``eydx``);
- :meth:`~smmargins.Margins.did` — 2x2 difference-in-differences on the
  response scale, with the joint covariance baked in;
- :meth:`~smmargins.MarginsResult.contrast` — exact linear combinations
  of any result, reusing the joint covariance;
- **Multi-outcome support** for ``MNLogit`` and ``OrderedModel``.

Multi-outcome models
--------------------

``smmargins`` supports ``statsmodels.MNLogit`` (multinomial logit) and
``statsmodels.miscmodels.ordinal_model.OrderedModel`` (ordered logit/probit).
For these models every statistic returns one value per outcome class — ``K``
values in place of the usual scalar — with full joint covariance across
both rows and classes.

.. code-block:: python

    ame = M.dydx("x1")          # AME of x1 on each class probability; K rows
    ame.summary()               # long-format DataFrame with `outcome` column

    # Subset to specific outcomes
    M.predict(outcome=1)               # only class 1
    M.predict(outcome="versicolor")    # by label, if labeled

Difference-in-differences
-------------------------

Two small additions turn the module into a full DiD estimator:

- :meth:`~smmargins.MarginsResult.contrast` forms any linear combination
  of the estimates directly on the already-computed joint covariance.
- :meth:`~smmargins.Margins.did` sets up the 2x2 grid and returns a
  :class:`~smmargins.DiDResult` bundling the four cell predictions, the
  two simple effects, and the DiD — all sharing the same joint covariance.

For **multi-outcome models**, ``did()`` returns a ``DiDResult`` where
every field carries the K-outcome axis. The DiD contains K estimates
whose sum is exactly zero.

Installation
------------

.. code-block:: bash

    pip install smmargins

Requires Python ≥3.9. Dependencies (``numpy``, ``pandas``, ``statsmodels``,
``scipy``, ``patsy``) are installed automatically.

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

Each call returns a :class:`~smmargins.MarginsResult` with ``.estimate``,
``.se``, ``.vcov``, ``.ci_lower``, ``.ci_upper``, ``.pvalue``, plus
``.summary()`` returning a :class:`pandas.DataFrame`. Pass ``use_t=True``
to the :class:`~smmargins.Margins` constructor for t-distribution
inference (uses ``results.df_resid``).

Why patsy
---------

When the formula is ``y ~ x1 + I(x1**2) + x1:x2 + C(group)`` and we
want the marginal effect of ``x1``, we cannot just nudge one column of
the design matrix — ``x1`` enters three columns. What we *can* nudge is
the ``x1`` column of the **original data frame**, then ask patsy to
rebuild the design matrix using the stored ``DesignInfo``:

.. code-block:: python

    patsy.dmatrix(design_info, perturbed_frame, return_type="matrix")

That preserves polynomial terms, interactions, splines (``bs(x, df=4)``),
and categorical contrasts automatically. It is also the right
abstraction for "hold ``age=45``" or "set ``group='b'``" — you mutate
the data frame, not the design matrix.

Formula vs. raw exog mode
-------------------------

:class:`~smmargins.Margins` supports models fit without formulas
(``sm.OLS(y, X).fit()``). In this *raw mode*, variable names are taken
from ``model.exog_names``.

.. warning::

    In raw mode, ``Margins`` cannot know about relationships between
    columns of the design matrix. If you manually included an
    interaction column (e.g. ``X["x1_x2"] = X["x1"] * X["x2"]``),
    perturbing ``x1`` for a marginal effect will **not** automatically
    update ``x1_x2``, and the marginal effect will be wrong.

If your model has interactions or transformations, fit it with a
formula so ``Margins`` can rebuild the design matrix correctly.

Where to next
-------------

- :doc:`math` — delta method, statistic schema, analytic vs FD
  Jacobian.
- :doc:`demos` — full Williams-style and DiD walkthroughs.
- :doc:`api` — reference documentation for every public class and
  method.
