"""Build the same Markdown guides and API locally and on Read the Docs."""

from importlib.metadata import version as package_version

project = "LLGM"
author = "Jae Sim"
copyright = "2026, Jae Sim"
release = package_version("llgm")
version = release
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinxcontrib.mermaid"]
source_suffix = {".md": "markdown"}
html_theme = "sphinx_rtd_theme"
html_title = f"LLGM {release}"
html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": True,
    "includehidden": True,
    "logo_only": True,
}
html_logo = "_static/brand/llgm-logo-dark.svg"
html_favicon = "_static/brand/llgm-favicon.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "development", "_static/brand/explorations"]
myst_heading_anchors = 3
autodoc_member_order = "bysource"
mermaid_version = "11.12.1"
mermaid_light_theme = "neutral"
mermaid_dark_theme = "neutral"
mermaid_height = "auto"
