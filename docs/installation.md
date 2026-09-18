# Installation

## Requirements

EOTLayer requires Python 3.10 or newer, PyTorch, NumPy, and SciPy. The public layer
accepts `torch.float32` and `torch.float64` tensors. Start with float64 for small
numerical checks; low-precision dtypes such as float16 and bfloat16 are not accepted.

## Install from source

Open a terminal in the directory containing `pyproject.toml` and `src/eotlayer/`.
Create and activate an isolated environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package, then run the first example:

```bash
python -m pip install -e .
python examples/docs_quickstart.py
```

The current package version is `0.1.0.dev0`. Installation from PyPI is not part of
this source snapshot's documented release process.

## Optional CUDA support

Install a CUDA-enabled PyTorch build suitable for your system using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).
Dense and Jacobi-W use PyTorch operations on the input device. The public
`sparse_w` backend additionally requires the cuDSS integration:

```bash
python -m pip install -e ".[cuda]"
```

This extra declares `nvmath-python>=0.6`. A compatible NVIDIA CUDA runtime and
cuDSS environment are also necessary. Installing the extra alone does not prove
the backend is usable on a particular machine. `ic0` currently accepts CPU input
only. See [Backward solvers](key_concepts/solvers.md).

## Check your installation

```python
import torch
from eotlayer import EOTLayer

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(EOTLayer)
```

## Build the documentation

The documentation toolchain uses Python 3.11 or newer. It reads the public API
signature directly from source without importing PyTorch, SciPy, or a GPU library.

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m http.server 8768 --bind 127.0.0.1 --directory docs/_build/html
```

Open [http://127.0.0.1:8768](http://127.0.0.1:8768) to browse the local site.
You can also open `docs/_build/html/index.html` directly; use the local server
when checking search and other browser features.
