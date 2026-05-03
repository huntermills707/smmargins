Cookbook: Per-variable DSL
==========================

The ``values=`` kwarg lets you fix variables at specific values or computed
statistics without restating every column.

Fixed value
-----------

Fix ``x1`` at 1 and leave everything else as observed::

    M.predict(values={"x1": 1})

Grid axis
---------

Create a grid over ``x1``::

    M.predict(values={"x1": [0, 1, 2]})

Reducers
--------

Reduce unspecified columns to their mean::

    M.predict(values={"x1": 1}, default_values="mean")

Use a percentile::

    M.predict(values={"income": "p25"})
