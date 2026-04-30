Mathematical motivation
=======================

This page derives, in one place, every statistic that
:class:`~smmargins.Margins` computes and the standard error attached to
each one. The motivation is identical to Stata's `margins-delta-method
FAQ <https://www.stata.com/support/faqs/statistics/compute-standard-errors-with-margins/>`_
and the cases enumerated in Richard Williams' `Margins01 notes
<https://academicweb.nd.edu/~rwilliam/stats/Margins01.pdf>`_; the goal
here is to write them down using a single primitive so the
implementation reduces to one Jacobian.

The delta method
----------------

For a fitted parameter vector :math:`\hat\beta` with estimated
covariance :math:`\widehat V`, and a (possibly vector-valued) statistic
:math:`g(\beta)`, a first-order Taylor expansion gives

.. math::

    g(\hat\beta) \;\approx\; g(\beta_0)
        + G\,(\hat\beta - \beta_0),
    \qquad
    G = \left.\tfrac{\partial g}{\partial \beta}\right|_{\beta_0}.

Taking variances on both sides yields

.. math::

    \widehat{\operatorname{Var}}\bigl[g(\hat\beta)\bigr]
        \;\approx\; G\,\widehat V\,G^\top .

Three things to notice:

1. The approximation depends only on **first** derivatives of :math:`g`.
2. Once :math:`G\widehat V G^\top` is in hand, *any* linear combination
   :math:`C\,g(\hat\beta)` has covariance :math:`C(G\widehat V G^\top)C^\top`
   — no further differentiation is needed. That is why
   :meth:`~smmargins.MarginsResult.contrast` is exact under the same
   approximation.
3. The standard error of :math:`g(\hat\beta)` is just the square-root
   of the diagonal of :math:`G\widehat V G^\top`; we report both.

A single statistic schema
-------------------------

Let :math:`f(\eta) = E[Y \mid \eta]` be the response-scale map (the
identity for OLS, the inverse link for a GLM, etc.). Williams
enumerates seven statistics; every one can be written as a linear
combination of mean predictions on different design matrices:

.. list-table::
    :header-rows: 1
    :widths: 15 60

    * - Statistic
      - :math:`g(\beta)`
    * - AAP
      - :math:`\frac{1}{n}\sum_i f(x_i^\top\beta)`
    * - APM
      - :math:`f(\bar x^\top\beta)`
    * - APR
      - :math:`\frac{1}{n}\sum_i f(x_i^\top\beta)` with some
        :math:`x` fixed
    * - AME
      - :math:`\frac{1}{n}\sum_i \partial f(x_i^\top\beta)/\partial x_{ij}`
    * - MEM
      - :math:`\partial f(\bar x^\top\beta)/\partial x_j`
    * - MER
      - AME with some :math:`x` held at representative values
    * - Contrast
      - :math:`E[f(\cdot)\mid x_j=\text{level}]
        - E[f(\cdot)\mid x_j=\text{ref}]`

Every entry is a linear combination of mean-predictions
:math:`\frac{1}{n}\sum_i f(x_i^\top\beta)` on different design
matrices. The single Jacobian primitive is therefore

.. math::

    \frac{\partial}{\partial\beta}\,
    \frac{1}{n}\sum_i f(x_i^\top\beta)
    \;=\; \frac{1}{n}\sum_i f'(x_i^\top\beta)\,x_i .

Substituting back into the delta method gives a closed form for the
covariance of every statistic in the table.

Marginal effects of a single variable
-------------------------------------

For a continuous covariate :math:`x_{ij}`,

.. math::

    \frac{\partial f(x_i^\top\beta)}{\partial x_{ij}}
    \;=\; f'(x_i^\top\beta)\,\frac{\partial (x_i^\top\beta)}{\partial x_{ij}} .

If the design matrix is just :math:`X = [\,\dots,\, x_j,\,\dots]`, the
inner derivative is :math:`\beta_j` and the AME collapses to the
familiar :math:`\overline{f'(x^\top\beta)}\,\beta_j`. With a formula
involving ``I(x**2)``, ``x1:x2``, ``bs(x, df=4)``, or a
:class:`~patsy.Categorical` contrast, the inner derivative is whatever
patsy produces when the *data column* is perturbed and the design
matrix is rebuilt — which is exactly what
:class:`~smmargins.Margins` does.

Discrete contrasts (factors, booleans, low-cardinality numerics) are
handled by computing :math:`f` at the level and at the reference, and
differencing — the delta-method covariance follows from stacking the
two predictions and applying the contrast matrix
:math:`C = [-1, +1]`.

Elasticities
------------

The four ``method=`` choices on :meth:`~smmargins.Margins.dydx` apply a
per-observation transform to the raw derivative
:math:`d_i = \partial f(x_i^\top\beta)/\partial x_{ij}` before
averaging:

.. list-table::
    :header-rows: 1
    :widths: 10 30 60

    * - Method
      - Per-obs transform
      - Interpretation
    * - ``dydx``
      - :math:`d_i`
      - Level change (default).
    * - ``dyex``
      - :math:`d_i\,x_i`
      - Semi-elasticity, :math:`dy/d(\ln x)`.
    * - ``eyex``
      - :math:`d_i\,x_i / y_i`
      - Full elasticity.
    * - ``eydx``
      - :math:`d_i / y_i`
      - Semi-elasticity, :math:`d(\ln y)/dx`.

Elasticities are only defined for continuous variables; the
implementation raises if you ask for an elasticity on a discrete one.

Outer Jacobian: analytic vs. finite differences
-----------------------------------------------

Stata's ``margins`` computes the entire :math:`G` numerically. We
compute it analytically when possible:

- For any GLM, :math:`f'` is read off
  ``family.link.inverse_deriv``. This covers Logit, Probit, Poisson,
  NegBin, Gamma, and Gaussian out of the box.
- For ``OLS`` / ``WLS`` / ``GLS``, :math:`f' \equiv 1` (identity link).

Whenever an analytic :math:`f'` is available **and** the linear
predictor is :math:`\eta = X\beta` (no offset/exposure), the chain rule
gives :math:`\partial g/\partial\beta` in closed form, saving
:math:`p` forward ``predict`` calls per statistic.

The fallback is central differences,

.. math::

    \frac{\partial f}{\partial x_j}
    \approx \frac{f(x + h e_j) - f(x - h e_j)}{2 h_j},
    \qquad
    h_j = \epsilon^{1/3}\cdot\max(|x_j|, 1),

with :math:`h_j` chosen to balance truncation error
(:math:`O(h^2)`) against round-off (:math:`O(\epsilon/h)`); the
optimum scales as :math:`h \sim \epsilon^{1/3}`. See
:func:`~smmargins._central_jacobian` and Nocedal & Wright,
*Numerical Optimization* (2006), Chapter 8.

The two paths agree to FD tolerance — there is a parity test that
runs every public API call on three different model classes with
``analytic=True`` and ``analytic=False`` and checks the estimates and
SEs match.

Why the response scale matters for DiD (Ai & Norton 2003)
---------------------------------------------------------

In a logit, the coefficient on a ``group:condition`` interaction is on
the *log-odds* scale. On the *probability* scale, the
difference-in-differences

.. math::

    \mathrm{DiD}(x_*)
        = \bigl[f(\eta_{1,1}) - f(\eta_{1,0})\bigr]
        - \bigl[f(\eta_{0,1}) - f(\eta_{0,0})\bigr]

is a nonlinear function of *every* parameter and *every* covariate
profile :math:`x_*`. You cannot read it off the interaction
coefficient. :meth:`~smmargins.Margins.did` evaluates the four cells
on the response scale, with their joint delta-method covariance, and
forms the DiD as a contrast — Ai & Norton's recommendation in code.

References
----------

- Stata FAQ, *How are the standard errors computed with margins?*
  https://www.stata.com/support/faqs/statistics/compute-standard-errors-with-margins/
- Williams, R. (2012). *Using the margins command to estimate and
  interpret adjusted predictions and marginal effects.*
  Stata Journal, 12(2), 308–331.
  https://www3.nd.edu/~rwilliam/stats/Margins01.pdf
- Ai, C., & Norton, E. C. (2003). *Interaction terms in logit and
  probit models.* Economics Letters, 80(1), 123–129.
  https://doi.org/10.1016/S0165-1765(03)00032-6
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*,
  2nd ed., Springer. Chapter 8.
