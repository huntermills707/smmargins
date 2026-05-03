Cookbook: Joint contrasts
=========================

``Margins.contrast`` computes two arms and their difference with the full
joint covariance.

PATE-style contrast
-------------------

::

    joint = M.contrast(a={"treat": 1}, b={"treat": 0})
    print(joint)

Mixed arms (DSL + newdata)
--------------------------

::

    joint = M.contrast(
        a={"treat": 1},
        b_newdata=control_df,
    )
