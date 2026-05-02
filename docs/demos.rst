Demos
=====

Two end-to-end walkthroughs ship in the repository root. Each one is a
single self-contained script you can run with
``python demo_<name>.py`` after installing ``smmargins``.

.. contents::
    :local:
    :depth: 2

.. _demo-margins:

Williams-style logit walkthrough
--------------------------------

``demo_margins.py`` reproduces, on a simulated voting dataset, every
core statistic in Richard Williams' *Margins01* notes and then
exercises the 0.3 inference surface end-to-end:

1.  Adjusted predictions at specific values (APR / ``margins, at(...)``)
2.  APM vs AAP (``margins, atmeans`` vs ``margins``)
3.  MER vs MEM vs AME for a continuous covariate
4.  Discrete contrasts for a multi-level categorical variable
5.  Discrete change for a 0/1 dummy
6.  Williams' classic interaction example: AME of ``age`` by sex
7.  Predicted probability over age, by sex (table for plotting)
8.  Analytic vs FD parity check
9.  Robust covariance via ``cov_type="HC3"``
10. Krinsky–Robb simulation VCE
11. Pairs bootstrap VCE
12. Simultaneous CIs via sup-t
13. Cluster-robust SEs (``cov_type="cluster"`` with ``cov_kwds=``)
14. Multiple-comparison adjustments side-by-side
    (Bonferroni / Šidák / sup-t)
15. User-supplied parameter covariance (``vcov=``)

Highlights from the script
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit a logit with an interaction:

.. code-block:: python

    fit = smf.logit(
        "voted ~ age + income + C(educ) + female + age:female",
        data=df,
    ).fit()
    M = Margins(fit)

APR — predictions at policy-relevant ages, averaging everything else
over the sample:

.. code-block:: python

    M.predict(atexog={"age": [25, 45, 65]})

MER, MEM, and AME for ``age`` — these can differ meaningfully in
nonlinear models with interactions:

.. code-block:: python

    M.dydx("age", atexog={"age": [25, 45, 65]})   # MER
    M.dydx("age", at="mean")                      # MEM
    M.dydx("age")                                 # AME

Discrete AME for a multi-level factor with an explicit reference
level:

.. code-block:: python

    M.dydx("educ", reference="college")

Williams' interaction lesson — same model, AME of ``age`` for each
sex:

.. code-block:: python

    M.dydx("age", atexog={"female": [0, 1]})

Robust SEs and alternative VCEs (sections 9–11):

.. code-block:: python

    Margins(fit, cov_type="HC3").dydx("age")          # heteroskedastic-robust
    M.dydx("age", vce="simulation",
           n_sims=2000, sim_seed=42)                   # Krinsky–Robb
    M.dydx("age", vce="bootstrap",
           n_boot=500, boot_seed=42)                   # pairs bootstrap

Cluster-robust SEs through ``cov_type="cluster"`` with cluster IDs
passed in ``cov_kwds`` (section 13):

.. code-block:: python

    Margins(fit, cov_type="cluster",
            cov_kwds={"groups": df["household"]}).dydx("age")

Family-wise CI methods side-by-side at five ages (section 14) — for
a correlated family of predictions, sup-t is typically narrower than
Bonferroni / Šidák:

.. code-block:: python

    common = dict(atexog={"age": [25, 35, 45, 55, 65]},
                  vce="simulation", n_sims=4000, sim_seed=123)
    M.predict(**common, ci_method="pointwise")
    M.predict(**common, ci_method="bonferroni")
    M.predict(**common, ci_method="sidak")
    M.predict(**common, ci_method="sup-t")

User-supplied parameter covariance (section 15) — drop in any
:math:`(k, k)` matrix and ``smmargins`` sandwiches it through the
Jacobian:

.. code-block:: python

    Margins(fit, vcov=my_vcov_matrix).dydx("age")

Full source
~~~~~~~~~~~

.. literalinclude:: ../demo_margins.py
    :language: python
    :linenos:

.. _demo-did:

Healthcare-style 2x2 difference-in-differences
----------------------------------------------

``demo_did.py`` answers a clinical question:

    *Is there a rate difference of condition* :math:`X` *between groups
    A and B, with or without a preexisting condition* :math:`Y`\ *?*

The script fits a logit on simulated patient data and reports, on the
**probability scale**:

- 4 cell predictions :math:`P(X \mid \text{group}, Y)`
- 2 simple effects :math:`P(X \mid B, Y) - P(X \mid A, Y)` at each
  :math:`Y`
- 1 difference-in-differences (whether the A-vs-B gap depends on
  :math:`Y`)

All with delta-method standard errors and confidence intervals. The
DiD here is **not** the coefficient on the ``group:Y`` interaction —
that coefficient is on the log-odds scale, while the clinical question
is about probabilities. This is Ai & Norton (2003) in practice; see
:doc:`math` for the derivation.

Highlights from the script
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit and call ``did()``:

.. code-block:: python

    fit = smf.logit(
        "condition_X ~ C(group) + preexist_Y + C(group):preexist_Y "
        "+ age + female",
        data=df,
    ).fit()
    M = Margins(fit)

    did = M.did("group", "preexist_Y",
                group_levels=["A", "B"],
                condition_levels=[0, 1])
    print(did)              # cells + simple effects + DiD

Same DiD at one specific patient profile (60-year-old male):

.. code-block:: python

    M.did("group", "preexist_Y",
          group_levels=["A", "B"], condition_levels=[0, 1],
          atexog={"age": 60, "female": 0})

Plot-ready cell table:

.. code-block:: python

    tbl = did.cells.summary()        # estimate / SE / CI per cell

Full source
~~~~~~~~~~~

.. literalinclude:: ../demo_did.py
    :language: python
    :linenos:
