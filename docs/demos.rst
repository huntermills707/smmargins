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
core statistic in Richard Williams' *Margins01* notes:

1. Adjusted predictions at specific values (APR / ``margins, at(...)``)
2. Adjusted predictions at means (APM / ``margins, atmeans``)
3. Average adjusted predictions (AAP / ``margins``)
4. Marginal effects at representative values
   (MER / ``margins, dydx(..) at(..)``)
5. Marginal effects at means (MEM / ``margins, dydx(..) atmeans``)
6. Average marginal effects (AME / ``margins, dydx(..)``)
7. Discrete contrasts for a categorical variable
8. Williams' classic interaction example: AME of ``age`` separately by
   sex

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
