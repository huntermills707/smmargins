# Tutorials (notebooks)

Tutorials are **executable Jupyter notebooks**. They answer *how do I use this feature?* and *when does it pay off?* — combining a worked example with rendered output (plots, tables, numbers).

## Conventions

- One notebook per feature area. Filename: `<feature>.ipynb` (e.g. `plotting.ipynb`, `values_dsl.ipynb`).
- **Do not commit pre-computed outputs.** Sphinx (`myst-nb`, mode `cache`) executes every notebook on docs build; outputs are cached out-of-tree. Manually edited outputs will silently rot when the API changes.
- Each notebook should be self-contained: simulate or load its own data, fit its own model, and run end-to-end without external state.
- Keep cells short. A reader should be able to scan the notebook top-to-bottom in a few minutes and get the point of the feature.
- Cross-link to the corresponding explanation page (`docs/explanations/`) for theory and to the API reference for full kwargs.

## Structure

Every tutorial follows roughly:

1. **Setup** — imports, simulate data, fit a model.
2. **Motivating example** — the simplest call that does something useful.
3. **Variations** — composition with other features (`vce=`, `over=`, `scale=`).
4. **Pitfalls** — what *not* to do, and what error messages mean.
5. **See also** — links to explanation + API.

## Adding a tutorial

1. Create `docs/tutorials/<feature>.ipynb`.
2. Add it to the `tutorials` toctree in `docs/index.rst`.
3. Build locally: `sphinx-build -b html docs docs/_build/html`. The first build executes the notebook; subsequent builds reuse the cache unless cells change.
