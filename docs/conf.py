"""Sphinx configuration of the STRIDER documentation."""

import datetime

import strider

project = "STRIDER"
author = "Roman Supernova PIT"
copyright = f"{datetime.date.today().year}, {author}"
release = strider.__version__
version = release.split("-", 1)[0]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"
default_role = "py:obj"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

html_theme = "alabaster"
html_static_path = ["_static"]
html_sidebars = {"**": ["about.html", "navigation.html", "relations.html", "searchbox.html"]}
html_theme_options = {
    "fixed_sidebar": True,
    "logo": "logo_black_filled_sm.png",
    "logo_text_align": "left",
    "description": "Transient classification and redshift inference from Roman prism spectra",
    "sidebar_width": "250px",
    "page_width": "75%",
    "body_max_width": "120ex",
    "show_relbars": True,
}

autoclass_content = "both"
autosummary_generate = True
automodapi_toctreedirnm = "api"
