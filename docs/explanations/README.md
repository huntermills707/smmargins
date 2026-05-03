# Explanations (theory)

Explanations are **MyST Markdown documents** that answer *why does this exist?* and *how does it work under the hood?* — derivations, design decisions, statistical theory, and tradeoffs.

## When to write an explanation

Write one when the answer to *"why?"* is not obvious from the code or the API docstring. Concretely:

- A statistical claim that needs a derivation (joint covariance, simultaneous CI calibration, KR vs delta).
- A design decision that has visible consequences for users (why `Expr` instead of bare strings; why `newdata=` is mutually exclusive with `at=`).
- A subtle invariant that future maintainers will need to preserve (why `_Profile` is ignorant of how its frame was built).

If the answer is *"read the code"* or *"see the docstring"*, you don't need an explanation page.

## Conventions

- Format: MyST Markdown (`.md`), not RST. Math via `$...$` (inline) and `$$...$$` (display).
- One topic per file. Filename: `<topic>.md` (e.g. `contrast_joint_covariance.md`, `delta_method.md`).
- Cross-link to the matching tutorial (`docs/tutorials/`) for *how to use* and to the API reference for *what kwargs exist* — explanations should not duplicate either.
- LaTeX-heavy is fine; this is the right place for it.

## Structure

A typical explanation has four sections:

1. **The problem** — what fails or is wrong without this feature/decision.
2. **The math / mechanism** — derivation, with explicit assumptions.
3. **What the implementation does** — pointer to the code path (file:line) so readers can verify.
4. **Tradeoffs and pitfalls** — when does the assumption break, what's the workaround.

## Adding an explanation

1. Create `docs/explanations/<topic>.md`.
2. Add it to the `explanations` toctree in `docs/index.rst`.
3. No execution needed — explanations are pure prose + math.
