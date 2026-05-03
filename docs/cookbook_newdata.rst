Cookbook: newdata escape hatch
==============================

Pass an arbitrary DataFrame for out-of-sample evaluation.

Predictions on hypothetical profiles
------------------------------------

::

    import pandas as pd

    hypo = pd.DataFrame({
        "age": [25, 45, 65],
        "income": [30_000, 50_000, 80_000],
    })
    M.predict(newdata=hypo)

Marginal effects on new data
----------------------------

::

    M.dydx("age", newdata=hypo)
