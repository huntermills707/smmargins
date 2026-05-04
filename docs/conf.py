import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'smmargins'
copyright = '2026'
author = ''
release = '0.5.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'myst_nb',
    'sphinx_design',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
    '.ipynb': 'myst-nb',
}

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autosummary_generate = True
autodoc_member_order = 'bysource'

# MyST + notebook execution
myst_enable_extensions = [
    'dollarmath',   # $...$ and $$...$$ math
    'amsmath',      # \begin{align}...\end{align}
    'colon_fence',  # ::: directives
    'deflist',
]
myst_heading_anchors = 3

# Notebooks: execute on build with caching, fail on errors. Stale outputs
# are the most common docs bug — never accept manually-edited notebook
# outputs in CI.
nb_execution_mode = 'cache'
nb_execution_timeout = 120
nb_execution_raise_on_error = True

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Conventions notes for contributors are not user-facing docs.
# Build artefacts (_build, jupyter_execute) must also be excluded so myst-nb
# doesn't double-count its own intermediate notebooks.
exclude_patterns = [
    '_build',
    'jupyter_execute',
    '**/jupyter_execute',
    'tutorials/README.md',
    'explanations/README.md',
    '**/README.md',
]

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = ['custom.css']
