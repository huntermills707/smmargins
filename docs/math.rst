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
- For ``MNLogit``, we use an analytic softmax-derivative gradient.

Whenever an analytic :math:`f'` or Jacobian is available **and** the linear
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
:func:`~smmargins.utils._central_jacobian` and Nocedal & Wright,
*Numerical Optimization* (2006), Chapter 8.

- **Analytic-vs-FD parity matrix.** Every public API call (AAP, APM,
  APR, AME, MEM, MER — continuous and discrete) is run twice on each
  of OLS+poly, Logit+`C(grp)`, Poisson, and MNLogit, with `analytic=True`
  and `analytic=False`, and the two paths must agree on both estimate and
  SE. This guards against the analytic chain-rule formula drifting
  from the FD answer that the other tests pin to Stata /
  `get_margeff`.

Beyond delta: simulation and bootstrap VCEs
-------------------------------------------

The delta method is a first-order approximation: :math:`G\widehat V
G^\top` is exact only to the extent that :math:`g` is locally linear
in :math:`\beta` near :math:`\hat\beta`. For statistics that are
sharply nonlinear (a probability near 0 or 1, an elasticity at a
near-zero prediction) this can produce symmetric Wald intervals that
extend beyond the natural support, or under-cover when the curvature
is non-negligible.

``smmargins`` exposes two alternatives behind the same ``vce=``
keyword.

**Krinsky–Robb simulation** (``vce="simulation"``). Draw
:math:`S` parameter vectors :math:`\beta_s\sim N(\hat\beta,\widehat
V)`, evaluate :math:`g(\beta_s)` for each, and read inference off
the empirical distribution of the draws:

.. math::

    \widehat{\operatorname{Var}}_{\mathrm{KR}}\bigl[g(\hat\beta)\bigr]
    = \frac{1}{S-1}\sum_{s=1}^S
        \bigl(g(\beta_s) - \bar g\bigr)\!\bigl(g(\beta_s) - \bar g\bigr)^{\!\top}.

The reported point estimate stays the *analytic* :math:`g(\hat\beta)`,
not :math:`\bar g`; this matches Stata's ``vce(simulation)``
convention and avoids spurious bias from finite :math:`S`. Pointwise
intervals default to empirical quantiles of the draws and so are not
forced to be symmetric — an asymmetric CI for an extreme probability
is the practical reason to reach for KR.

**Bootstrap** (``vce="bootstrap"``). Refit the model on
:math:`B` resamples of the rows and read inference off the bootstrap
distribution. Three resampling schemes are built in (``boot_method=``):

- ``"pairs"`` — :math:`(y_i, x_i)` resampled IID (default);
- ``"cluster"`` — whole clusters resampled with replacement,
  required for robustness to within-cluster correlation;
- ``"block"`` — moving-block resampling for time series, with
  ``block_size=`` chosen to span the dependence horizon.

Failed refits are caught and counted; a ``RuntimeWarning`` fires when
the failure rate exceeds 5%. The point estimate is again the analytic
:math:`g(\hat\beta)`, not the bootstrap mean.

Simultaneous confidence intervals
---------------------------------

For a family of :math:`m` margins returned in a single call,
``ci_method=`` controls the critical value. With pointwise critical
value :math:`z_{1-\alpha/2}`:

- ``"bonferroni"`` uses :math:`z_{1-\alpha/(2m)}` — works with any
  VCE;
- ``"sidak"`` uses
  :math:`z_{(1+(1-\alpha)^{1/m})/2}`, slightly narrower than
  Bonferroni — also works with any VCE;
- ``"sup-t"`` uses the :math:`(1-\alpha)` quantile of the maximum
  standardised absolute deviation across the family, computed from
  the simulation/bootstrap draw matrix:

  .. math::

      c_{\sup\text{-}t}^{\,(1-\alpha)}
      = Q_{1-\alpha}\!\left(
          \max_{j=1,\dots,m}
          \left|\frac{g_j(\beta_s) - g_j(\hat\beta)}{\widehat\sigma_j}\right|
        \right).

Sup-t requires draws (so ``vce`` must be ``"simulation"`` or
``"bootstrap"``) and is typically narrower than Bonferroni / Šidák
when the margins in the family are correlated — exactly the case for
margins evaluated at neighbouring covariate profiles.

Prediction scales and the chain rule
------------------------------------

The ``scale=`` kwarg on :meth:`~smmargins.Margins.predict` and
:meth:`~smmargins.Margins.dydx` swaps the response-scale map
:math:`f(\eta)` for an arbitrary :math:`g(\eta)`. Built-in scales
(``"linear"``, ``"or"``, ``"exp"``, ``"log"``, …) and any user-defined
:class:`~smmargins.Transform` provide :math:`g`, :math:`g'`, and (for
``dydx``) :math:`g''` analytically.

**Predicted values.** For a single design :math:`X`, the gradient of
:math:`g(X\beta)` w.r.t. :math:`\beta` is

.. math::

    \frac{\partial g(\eta)}{\partial \beta} = g'(\eta) \cdot X.

Only :math:`g'` is needed; this is why ``predict(scale=t)`` accepts
``Transform`` instances with ``hess=None``.

**AME on the g-scale.** For a continuous covariate :math:`x_k`, the
AME on the g-scale is

.. math::

    \mathrm{AME}_k^{(g)}
        = \frac{1}{n}\sum_i g'(\eta_i) \cdot \beta_k.

Differentiating w.r.t. :math:`\beta`:

.. math::

    \frac{\partial \mathrm{AME}_k^{(g)}}{\partial \beta_j}
        = \frac{1}{n}\sum_i \bigl[\,g''(\eta_i) \cdot x_{ij} \cdot \beta_k
            + g'(\eta_i) \cdot \delta_{jk}\,\bigr].

Both terms appear: the first comes from differentiating
:math:`g'(\eta_i)` through :math:`\eta_i = x_i^\top\beta`, the second
from differentiating the explicit :math:`\beta_k`. ``smmargins``
computes this Jacobian analytically when ``Transform.hess`` is
provided (every built-in does). Without ``hess``, ``dydx`` raises —
this is the package's strict-derivative contract. A module-level
``_DYDX_FD_ONLY`` flag (in :mod:`smmargins._derivs`) forces a
finite-difference fallback for testing parity.

**Discrete contrasts** under a g-scale are simpler: the contrast is
:math:`g(\eta_a) - g(\eta_b)` averaged over the sample, and its
gradient is :math:`g'(\eta_a)\,x_a - g'(\eta_b)\,x_b`. Only
:math:`g'` enters; :math:`g''` is not used on the discrete path.

**Composition with elasticities.** Each ``method=`` (``"eyex"``,
``"dyex"``, ``"eydx"``) wraps the g-scale formulas above with the
appropriate :math:`x` or :math:`g(\eta)` factor, e.g.

.. math::

    \mathrm{eyex}_k^{(g)} = \mathrm{AME}_k^{(g)} \cdot
        \frac{\overline{x_k}}{\overline{g(\eta)}}.

The Jacobian inherits an extra factor or quotient but no new
analytic term, so the same :math:`g'` / :math:`g''` machinery powers
every elasticity × scale combination.

Subgroup AMEs (``over=``)
-------------------------

``over=`` partitions the sample into :math:`G` subgroups and computes
the AME within each. Because every subgroup AME is a function of the
*same* :math:`\hat\beta`, the joint covariance of the :math:`G`
estimates is **not block-diagonal**:

.. math::

    \widehat{\mathrm{Var}}\bigl(\hat\theta_{1\ldots G}\bigr)
        = J\,\widehat V\,J^\top,
    \quad
    J = \begin{bmatrix}
        \partial \theta_1 / \partial \beta \\
        \vdots \\
        \partial \theta_G / \partial \beta
    \end{bmatrix}.

Computing each subgroup independently and concatenating would discard
the off-diagonal blocks and break Wald tests across subgroups, so
``smmargins`` always builds the stacked Jacobian.

Joint Wald and pairwise contrasts
---------------------------------

For a result with point estimate :math:`\hat\theta` and covariance
:math:`V`, :meth:`~smmargins.MarginsResult.wald` computes

.. math::

    W = (C\hat\theta - c_0)^\top (C V C^\top)^{-1} (C\hat\theta - c_0),

distributed as :math:`\chi^2_{\mathrm{rank}(CVC^\top)}` under
:math:`H_0`. ``pairwise(by=…)`` builds :math:`C` for all
level-vs-level comparisons of a factor and returns a
:class:`~smmargins.MarginsResult` whose ``ci_method=`` flows through
the simultaneous-CI machinery — including ``"sup-t"`` when the
underlying result carries draws.

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
