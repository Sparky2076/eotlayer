# Building and maintaining the docs

## Documentation stack

The site uses Sphinx, MyST Markdown, Sphinx Book Theme, MathJax, and copyable code
blocks. The layout follows the same theme family as the RLAOpt reference site:
navigation on the left, article content in the center, and page contents on the
right. Narrow screens use collapsible navigation.

## Build locally

From the repository root, using Python 3.11 or newer:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m http.server 8768 --bind 127.0.0.1 --directory docs/_build/html
```

The build treats warnings as errors. API signatures are parsed from source and
checked against the contract without loading the numerical package.

## Check examples

With EOTLayer and its runtime dependencies installed:

```bash
python scripts/check_docs.py --run-examples
```

Each example page includes its actual script with Sphinx `literalinclude`. Edit
the script rather than pasting a second copy into the page. The examples use
deterministic data and verify numerical behavior; they are not performance tests.

## Page structure

```text
docs/
├── index.md                 # Landing page and quick start
├── installation.md          # Source installation and local build
├── key_concepts/            # EOT, derivatives, solvers, accuracy
├── demos/                   # Runnable examples
├── api/                     # Public class, diagnostics, backend contract
├── development/             # Status, troubleshooting, maintenance
├── _static/                 # Theme adjustments and favicon
└── conf.py                  # Sphinx configuration and signature validation
```

## Read the Docs configuration

The root `.readthedocs.yaml` points to `docs/conf.py`, uses Python 3.12, and installs
only the documentation requirements. The source repository is
[Sparky2076/eotlayer](https://github.com/Sparky2076/eotlayer).

The intended address is `https://eotlayer.readthedocs.io/en/stable/`. Read the Docs
assigns the project slug and manages the `stable` version; the URL cannot be
created by changing Sphinx settings alone. Import the repository, use project
slug `eotlayer`, then activate the `stable` branch in the project's Versions tab.
The `stable` branch names the documentation version; the numerical package
remains pre-alpha. The repository provides source, edit, and issue links.

For future setup, consult the official
[configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html)
and [version documentation](https://docs.readthedocs.com/platform/stable/versions.html).
