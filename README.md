# EOTLayer

An English Sphinx documentation site for the supplied EOTLayer `0.1.0.dev0`
source snapshot. It uses **Read the Docs Sphinx Theme**, following the documentation
layout of [TorchOpt](https://torchopt.readthedocs.io/): a searchable left navigation,
grouped documentation sections, and a clear single-column article. EOTLayer-specific
concepts, runnable examples, and the source-checked public API are retained.

Source repository: [Sparky2076/eotlayer](https://github.com/Sparky2076/eotlayer).
The [EOTLayer stable website](https://eotlayer.readthedocs.io/en/stable/) is already
hosted by the Read the Docs project `eotlayer` and builds from this repository's
`stable` branch. The project's default version is `latest`, which follows the
default source branch `codex/documentation`. Updating the requested `/en/stable/`
site therefore requires updating `stable`, followed by a successful hosted build.
See [Publishing](PUBLISHING.md) for the existing deployment settings.

## Browse the built site

Open `docs/_build/html/index.html`, or serve it locally:

```bash
python -m http.server 8768 --bind 127.0.0.1 --directory docs/_build/html
```

Then open <http://127.0.0.1:8768>.

## Rebuild

Use Python 3.11 or newer for the documentation tools:

```bash
python -m pip install -r docs/requirements.txt
python scripts/check_docs.py
python -m sphinx -b html -W --keep-going docs docs/_build/html
```

No PyTorch or CUDA installation is needed to build the HTML. The real public
constructor is parsed from `src/eotlayer/layer.py` and checked against the supplied
`project-contract.json`.

The theme refresh preserves the runtime source snapshot from commit
`8b1b42f25f779b0ece522c525e3cd7820a82e3ec`. It does not introduce another
EOTLayer API, add solvers, or declare a PyPI release.

## Run documented examples

Install the included numerical package first:

```bash
python -m pip install -e .
python scripts/check_docs.py --run-examples
```

The four complete demos check a plan VJP, two implicit solvers against each other,
the envelope gradient identity, and batch shape/marginal behavior. Their source
is included directly in the rendered pages.

## Contents

- Installation and setup
- Optimal transport, differentiation modes, backward solvers, workflow, accuracy
- Four runnable demos
- `EOTLayer` reference, diagnostics, and backend contract
- Project status, troubleshooting, and documentation maintenance

The numerical `src/` is copied unchanged from the supplied archive. This repository
does not include the archived Git history, DDN reference checkout, paper outputs,
or original repository-wide completion gate. The documentation describes the
current implementation explicitly; no GPU speedup or production-readiness claim
is made by publishing the documentation.
