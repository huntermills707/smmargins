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
- **Multi-outcome support** for ``MNLogit`` and ``OrderedModel``;
- **Robust covariance passthrough** via ``cov_type=`` (``HC0``–``HC3``,
  ``cluster``, ``HAC``) and ``vcov=`` (user-supplied matrix) on every
  call;
- **Krinsky–Robb simulation** (``vce="simulation"``) and **bootstrap**
  (``vce="bootstrap"``, with pairs / cluster / moving-block resampling)
  alternatives to the delta method;
- **Simultaneous confidence intervals** for a family of margins via
  ``ci_method="bonferroni" | "sidak" | "sup-t"``.
- **Prediction scales** (``scale=``) for ``predict()`` / ``dydx()``:
  ``"response"`` (default), ``"linear"`` (Stata's ``xb``), ``"pr"``,
  ``"ir"``, ``"or"``, ``"exp"``, ``"log"``, plus user-defined
  :class:`~smmargins.Transform` instances with analytic
  :math:`\lambda'` and :math:`\lambda''`.
- **Observation weights** (``weights=``, ``weight_type=``) threaded
  through AME averaging and bootstrap resampling.
- **Subgroup AMEs** via ``over=``, with the full joint covariance
  preserved across subgroups (not block-diagonal).
- **Joint Wald tests and pairwise comparisons** via
  :meth:`~smmargins.MarginsResult.wald` and
  :meth:`~smmargins.MarginsResult.pairwise`, returning a
  :class:`~smmargins.WaldResult` or a pairwise
  :class:`~smmargins.MarginsResult` whose ``ci_method=`` composes
  with the simultaneous-CI machinery.

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

Inference options
-----------------

Every call to :meth:`~smmargins.Margins.predict`,
:meth:`~smmargins.Margins.dydx`, and :meth:`~smmargins.Margins.did`
accepts the same block of inference kwargs.

**Robust covariance.** Pass ``cov_type=`` and (where relevant)
``cov_kwds=`` to recompute the parameter covariance under a sandwich
estimator before delta-method propagation. Pass ``vcov=`` to inject
your own :math:`(k, k)` matrix instead. The two are mutually
exclusive; if neither is set, ``results.cov_params()`` is used as-is
(so a robust ``cov_type`` baked into the original ``fit`` is
preserved).

.. code-block:: python

    M = Margins(fit, cov_type="HC3")
    M.dydx("x1", cov_type="cluster",
           cov_kwds={"groups": df["firm_id"]})
    M.predict(at="mean", vcov=my_custom_matrix)

**Krinsky–Robb simulation** (``vce="simulation"``). Draws
:math:`\beta_s \sim N(\hat\beta, \hat V)` and evaluates the margin
function for each draw. The reported point estimate stays the
analytic :math:`g(\hat\beta)`; draws contribute only to standard
errors and percentile intervals. Composes with ``cov_type=``.

.. code-block:: python

    M.dydx("x1", vce="simulation", n_sims=2000, sim_seed=42)
    M.dydx("x1", vce="simulation", cov_type="HC1")

**Bootstrap** (``vce="bootstrap"``). Refits the model on each
bootstrap sample. Pairs is the default; ``boot_method="cluster"``
takes a ``cluster=`` ID array, and ``boot_method="block"`` does a
moving-block resample with ``block_size=``. Optional ``verbose=True``
shows a progress bar; ``n_jobs`` parallelises refits via ``joblib``.

.. code-block:: python

    M.dydx("x1", vce="bootstrap", n_boot=1000, boot_seed=42)
    M.dydx("x1", vce="bootstrap", boot_method="cluster",
           cluster=df["firm_id"], n_boot=1000)

**Simultaneous confidence intervals** (``ci_method=``). Bonferroni
and Šidák are pure critical-value adjustments and work with any VCE.
``"sup-t"`` consumes the simulation/bootstrap draw matrix to compute
the simultaneous critical value as the upper-tail quantile of the
maximum standardised absolute deviation across the family — typically
narrower than Bonferroni when the margins in the family are
correlated.

.. code-block:: python

    M.dydx(["x1", "x2", "x3"], ci_method="bonferroni")
    M.predict(atexog={"age": [25, 45, 65]},
              vce="simulation", n_sims=4000,
              ci_method="sup-t")

Scales, weights, subgroups, joint tests
---------------------------------------

Six features added in 0.4. Each composes with the inference options
above (``vce=``, ``cov_type=``, ``ci_method=``).

**Prediction scale (``scale=``).** Choose what ``predict()`` and
``dydx()`` operate on. The default ``"response"`` keeps current
behaviour; ``"linear"`` is the linear predictor :math:`\eta = X\beta`
(Stata's ``predict(xb)``). Other built-ins: ``"pr"``, ``"ir"``,
``"or"``, ``"exp"``, ``"log"``. Invalid combinations error clearly
(``scale="or"`` on Poisson, ``scale="log"`` when any prediction is
non-positive).

.. code-block:: python

    M.predict(scale="linear")
    M.dydx("x1", scale="or")          # AME on the odds-ratio scale
    M.dydx("x1", method="eyex", scale="linear")  # elasticities × scales compose

**Custom transforms (``Transform``).** Pass a
:class:`~smmargins.Transform` for a user-defined :math:`\lambda`.
``dydx`` requires an analytic second derivative ``hess=``; ``predict``
only needs ``grad=``. No autodiff — analytic derivatives are a
deliberate package contract.

.. code-block:: python

    from smmargins import Transform
    import numpy as np

    square = Transform(value=lambda e: e**2,
                       grad=lambda e: 2*e,
                       hess=lambda e: np.full_like(e, 2.0),
                       name="square")
    M.dydx("x1", scale=square)

Built-ins (``Identity``, ``Linear``, ``Exp``, ``Log``, ``Logit``,
``Probit``) live in :mod:`smmargins.transforms`.

**Observation weights (``weights=``).** Pass weights at construction.
``weight_type="sampling"`` (default) or ``"frequency"``. WLS-fitted
results respect their fit-time weights when ``weights=None``;
explicit ``weights=`` overrides and warns. Bootstrap resampling uses
weight-proportional draws under sampling weights.

.. code-block:: python

    M = Margins(fit, weights=w)
    M = Margins(fit, weights=counts, weight_type="frequency")

**Subgroup AMEs (``over=``).** Partition the sample by one or more
columns and average within each subgroup. The full joint covariance
is preserved (not block-diagonal), so cross-subgroup contrasts and
Wald tests stay valid.

.. code-block:: python

    M.dydx("x1", over="region")
    M.dydx("x1", over=["region", "sex"])

**Joint tests and pairwise comparisons.**
:meth:`~smmargins.MarginsResult.wald` returns a
:class:`~smmargins.WaldResult`; :meth:`~smmargins.MarginsResult.pairwise`
returns a pairwise :class:`~smmargins.MarginsResult` whose
``ci_method=`` flows into the simultaneous-CI machinery.

.. code-block:: python

    res = M.dydx(["x1", "x2", "x3"])
    res.wald()                       # joint test that all margins == 0
    res.wald(C=[[1, -1, 0]])         # H0: AME(x1) == AME(x2)

    res = M.dydx("group")
    res.pairwise(by="group")                          # raw
    res.pairwise(by="group", ci_method="bonferroni")  # adjusted
    res.pairwise(by="group", ci_method="sup-t")       # needs vce='simulation'/'bootstrap'

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

    # Alternative VCEs and robust covariance
    M.dydx("age", cov_type="HC3")                  # heteroskedastic-robust
    M.dydx("age", vce="simulation", n_sims=2000,
           sim_seed=0)                              # Krinsky–Robb
    M.dydx("age", vce="bootstrap", n_boot=1000,
           boot_seed=0)                             # pairs bootstrap

    # Simultaneous CIs across a family of three predictions
    M.predict(atexog={"age": [25, 45, 65]},
              vce="simulation", n_sims=4000,
              ci_method="sup-t")

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
