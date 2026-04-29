import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'marginal_effects'
copyright = '2026'
author = ''
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

autosummary_generate = True
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
}

html_theme = 'alabaster'
html_static_path = []
