"""EOTLayer documentation: build without importing numerical dependencies."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
project = "EOTLayer"
author = "EOTLayer contributors"
copyright = "2026, EOTLayer contributors"
release = metadata["project"]["version"]
version = release
language = "en"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "myst_parser",
    "sphinx_copybutton",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store",
    "api-contract.md", "api-audit.md", "backend-adapter-contract.md",
    "backend-migration-audit.md", "effective-resistance-gpu-audit.md",
    "effective-resistance-paper-ledger.md",
]
html_theme = "sphinx_book_theme"
html_title = "EOTLayer Documentation"
html_short_title = "EOTLayer"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "repository_url": "https://github.com/Sparky2076/eotlayer",
    "repository_branch": os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "codex/documentation"),
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "logo": {"text": "EOTLayer"},
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "use_download_button": False,
    "use_fullscreen_button": True,
}
html_context = {"default_mode": "light"}
html_show_sourcelink = False
html_copy_source = False
# Local previews do not claim to be a deployed Read the Docs version.
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_last_updated_fmt = None
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
myst_heading_anchors = 3
mathjax3_config = {"tex": {"tags": "ams"}}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
pygments_style = "friendly"
pygments_dark_style = "monokai"


def public_signature() -> str:
    """Read the real constructor with AST; no torch/SciPy/GPU import needed."""
    tree = ast.parse((ROOT / "src/eotlayer/layer.py").read_text(encoding="utf-8"))
    layer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EOTLayer")
    constructor = next(n for n in layer.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    arguments = copy.deepcopy(constructor.args)
    arguments.args = arguments.args[1:]
    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
        argument.annotation = None
    signature = f"EOTLayer({ast.unparse(arguments)})"
    contract = json.loads((ROOT / "project-contract.json").read_text(encoding="utf-8"))
    if signature != contract["product"]["signature"]:
        raise RuntimeError("EOTLayer constructor and project-contract.json disagree")
    return signature


def insert_signature(app, docname, source):
    if docname == "api/layer":
        app.env.note_dependency(str(ROOT / "src/eotlayer/layer.py"))
        app.env.note_dependency(str(ROOT / "project-contract.json"))
        source[0] = source[0].replace("EOTLAYER_CONSTRUCTOR_FROM_SOURCE", public_signature())


def setup(app):
    public_signature()
    app.connect("source-read", insert_signature)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
