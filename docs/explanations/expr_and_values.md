# Why `Expr` and the `values=` DSL

## The problem

Counterfactual profiling requires three distinct operations:

1. **Reduce** a column to a summary (mean, median, 25th percentile).
2. **Transform** every observation by a formula ("income + 10%").
3. **Replace** the data with an arbitrary frame.

A single API surface has to serve all three without ambiguity. In particular, bare strings are already overloaded: `"mean"` is a reducer, `"college"` is a categorical level, and `"income * 1.1"` is a formula. `smmargins` resolves this with a small, explicit type system.

## The design

### Reducers vs. scalars

Strings in `values=` are inspected at runtime:

- If the string matches a built-in reducer (`"mean"`, `"median"`, `"p25"`, …) or a percentile pattern (`p0`–`p100`), it is interpreted as a reducer.
- Otherwise it is treated as a scalar categorical level.

This is why `"p50"` reduces income to its median, while `"college"` sets education to that level.

### `Expr` as an explicit opt-in

Formula strings are **not** parsed automatically, because a formula like `"age + 5"` could be mistaken for a categorical level or a mistyped reducer. :class:`~smmargins.Expr` is an explicit wrapper that says "evaluate this via `df.eval`":

```python
M.predict(values={"income": Expr("income * 1.10")})
```

This avoids the ambiguity that would arise if bare strings were passed to `pandas.eval`, and it makes the user's intent visible at the call site.

### Callables for programmatic transforms

For transforms that are easier to write as Python functions than as strings, `values=` also accepts callables. The callable receives the original data frame and must return a Series or array of the same length. This is useful when the transform reuses intermediate computations or depends on external parameters:

```python
def add_tax(d):
    return d["income"] * 1.10 + 500

M.predict(values={"income": add_tax})
```

### `newdata=` as the escape hatch

The `values=` DSL always starts from the original fitting data. If you need out-of-sample profiles or a completely different population, `newdata=` replaces the entire frame. It is mutually exclusive with `values=` because the two concepts do not compose: there is no sensible way to apply a per-variable DSL on top of an arbitrary foreign frame.

## Tradeoffs

- **Explicit over implicit.** `Expr` requires an import and a wrapper. The payoff is that strings never silently change meaning when a new reducer is added.
- **Observed-values default.** `default_values="asobserved"` means that forgetting a variable leaves the full distribution intact. This is conservative but can produce wide confidence intervals when many variables vary. Switch to `default_values="mean"` for a tighter, MEM-style profile.
- **Cartesian product only.** Sequences in `values=` are expanded as a full cartesian product. There is no "meshgrid-only" mode; if you need a custom grid shape, build the frame yourself and use `newdata=`.

## What the implementation does

The resolution pipeline lives in `smmargins/_design.py`:

1. `_classify_value` inspects each `values=` entry and returns a `ValueSpec` (`scalar`, `sequence`, `reducer`, `callable`, or `expr`).
2. `expand_values` applies `default_values` to unspecified columns, then applies non-sequence specs, then takes the cartesian product over sequences.
3. The resulting frames are passed to `_Profile.materialize`, which rebuilds the design matrix via patsy (or raw exog lookup) for each profile.

This separation means the DSL is pure data manipulation: it knows nothing about predictions, derivatives, or scales.
