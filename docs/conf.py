# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime

import enjoyn

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

year = datetime.datetime.utcnow().year
project = "enjoyn"
copyright = f"2022 to {year}, ahlive" if year != 2022 else "2022"
release = enjoyn.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "enjoyn.svg"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/ahuang11/enjoyn",
    "path_to_docs": "https://enjoyn.readthedocs.io/en/latest/",
    "repository_branch": "main",
    "logo_only": True,
    "use_issues_button": True,
    "use_download_button": False,
    "use_repository_button": True,
}

nbsphinx_allow_errors = False
nbsphinx_kernel_name = "python3"
