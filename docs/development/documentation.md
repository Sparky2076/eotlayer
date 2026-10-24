# Building and maintaining the docs

## Documentation stack

The site uses Sphinx, MyST Markdown, Read the Docs Sphinx Theme, MathJax, and
copyable code blocks. The layout follows
[TorchOpt's documentation](https://torchopt.readthedocs.io/): a searchable left
sidebar with grouped sections and a single article column. Narrow screens use
collapsible navigation. The logo, wording, examples, and technical contracts
belong to EOTLayer.

## Build locally

From the repository root, using Python 3.11 or newer:

```bash
python -m pip install -r docs/requirements.txt
python scripts/check_docs.py
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m http.server 8768 --bind 127.0.0.1 --directory docs/_build/html
```

The build treats warnings as errors. API signatures are parsed from source and
checked against the contract without loading the numerical package.

The numerical implementation is the `0.1.0.dev0` source snapshot preserved from
commit `8b1b42f25f779b0ece522c525e3cd7820a82e3ec`. A visual refresh must keep its
constructor, mode semantics, tensor contracts, and canonical backend names tied
to that source. Documentation of a later EOTLayer interface requires a separate
source and API review. No PyPI release is implied by this website.

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
├── _static/                 # Theme adjustments, branding, and favicon
└── conf.py                  # Sphinx configuration and signature validation
```

## Read the Docs configuration

The root `.readthedocs.yaml` points to `docs/conf.py`, uses Python 3.12, and installs
only the documentation requirements. The source repository is
[Sparky2076/eotlayer](https://github.com/Sparky2076/eotlayer).

The project `eotlayer` is already deployed at
[https://eotlayer.readthedocs.io/en/stable/](https://eotlayer.readthedocs.io/en/stable/).
That version tracks this source repository's Git branch `stable`. The default
documentation version is `latest`, which follows the default source branch
`codex/documentation`. The two Git branches are independent: update `stable` to
the reviewed documentation commit to change the requested stable website.

The existing webhook can trigger a hosted build after the branch update. Check
the [Read the Docs builds](https://app.readthedocs.org/projects/eotlayer/builds/)
for the intended commit and version; the project owner can trigger a build if
the webhook does not start one. A successful GitHub Actions artifact alone does
not update Read the Docs. The `stable` documentation version does not change
the numerical package's pre-alpha status or version `0.1.0.dev0`.

The documentation copy in `Percy-Deng/diff-sinkhorn` is separate from the existing
hosted project's source. Its changes do not update this website unless the
Read the Docs owner explicitly changes that source configuration.

For future setup, consult the official
[configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html)
and [version documentation](https://docs.readthedocs.com/platform/stable/versions.html).
