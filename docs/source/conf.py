# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from evalwrf import __version__

sys.path.insert(0, os.path.abspath("../../src"))


project = "evalwrf"
copyright = "2025, Florian Ennemoser"
author = "Florian Ennemoser"
version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "inherited-members": None,
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "none"
autosummary_generate = True

toc_object_entries_show_parents = 'hide'
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["_build"]
source_suffix = {
    ".rst": "restructuredtext",
}
master_doc = "index"



# COLORS
dark_red = "#9f2127"
red = "#ee2832"
dark_grey = "#696e6f"
grey = "#aaadac"
light_grey = "#e2e3e2"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# html_logo = "_static/MAIN_LOGO.svg"
html_sidebars = {"index": ["search-button-field"], "**": ["search-button-field", "sidebar-nav-bs"]}
html_theme_options = {
    "external_links": [],
    # "logo": {"image_dark": "_static/WHITE_IMAGE.svg"},
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "search_bar_text": "Search the docs...",
    "show_version_warning_banner": False,
    "github_url": "https://github.com/FlorianEnnemoser/evalwrf",
    "show_nav_level": 0,  # collapse navigation items
    "secondary_sidebar_items": [],
    # "switcher": {
    #     "version_match": version,
    #     "json_url": "_static/switcher.json",
    # },
    # # VERSION SWITCHING: https://pydata-sphinx-theme.readthedocs.io/en/v0.8.1/user_guide/configuring.html#add-a-json-file-to-define-your-switcher-s-versions

}


# CREATE DOC: uv run sphinx-build -M html docs/source docs/build