import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Trimodal-DBS'
copyright = '2024'
author = 'Project Authors'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax'
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
