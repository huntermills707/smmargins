Cookbook: Generate-style transforms
===================================

You can pass callables or ``Expr`` wrappers inside ``values=`` to transform
variables row-by-row.

Callable
--------

Scale income by 10%::

    M.predict(values={"income": lambda df: df["income"] * 1.10})

Expr wrapper
------------

Same with a pandas-eval expression::

    from smmargins import Expr
    M.predict(values={"income": Expr("income * 1.10")})

Chaining with reducers
----------------------

Fix ``x1`` at its mean, then evaluate ``x2 + x1`` per row::

    M.predict(values={"x1": "mean", "x2": Expr("x2 + x1")})
