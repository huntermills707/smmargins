Cookbook: Plotting
==================

``smmargins.plot`` provides Matplotlib-based plotting for predictions,
slopes, and comparisons.

Prediction curve
----------------

::

    from smmargins import plot_predictions
    fig, ax = plot_predictions(M, "age")

Marginal effect curve
---------------------

::

    from smmargins import plot_slopes
    fig, ax = plot_slopes(M, "age", condition="income")

Comparison curve
----------------

::

    from smmargins import plot_comparisons
    fig, ax = plot_comparisons(M, "treat", condition="age")

Faceting with ``by=``
---------------------

::

    fig, ax = plot_predictions(M, "age", by="sex")
