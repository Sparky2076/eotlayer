# EOTLayer

EOTLayer is a PyTorch library for differentiating through balanced entropic
optimal transport. It computes transport plans and exposes the numerical controls
needed to use them inside a learning objective.

- **Differentiation modes.** Differentiate finite Sinkhorn iterations, apply
  implicit plan differentiation, or compute the envelope gradient of the
  regularized transport value.
- **Backward solvers.** Select Dense, Jacobi-W, CPU IC(0), or CUDA Sparse-W
  through one `EOTLayer` interface.
- **Numerical diagnostics.** Inspect marginal errors, backward residuals,
  iteration counts, and effective tolerances alongside the result.

```{raw} html
<p class="eot-home-links">
  <a href="https://github.com/Sparky2076/eotlayer">GitHub</a>
  <a href="api/layer.html">API reference</a>
  <a href="development/status.html">Project status</a>
</p>
```

## Installation

Install PyTorch for your CPU or CUDA environment first. From the root of the
EOTLayer source checkout, run:

```bash
python -m pip install -e .
```

These docs describe the included `0.1.0.dev0` source snapshot.
See [Installation](installation.md) for environment setup and optional CUDA
support.

## Quick start

Create a cost matrix and two positive, equal-mass marginals. The output is a
transport plan with the same shape as the cost matrix.

```{literalinclude} ../examples/docs_quickstart.py
:language: python
:start-after: docs-start
:end-before: docs-end
```

`plan.sum(1)` approximates `source`, and `plan.sum(0)` approximates `target`.
Calling `loss.backward()` computes the gradient with respect to `cost`.
The example uses float64 so that the selected tolerances are meaningful.
See [Differentiation modes](key_concepts/differentiation.md) for the distinction
between finite unrolling and implicit differentiation.

```{toctree}
:maxdepth: 2
:caption: Documentation

installation
key_concepts/index
```

```{toctree}
:maxdepth: 2
:caption: Examples

demos/index
```

```{toctree}
:maxdepth: 2
:caption: Developer Documentation

development/index
```

```{toctree}
:maxdepth: 2
:caption: API Documentation

api/index
```

## Support

Report issues on [GitHub](https://github.com/Sparky2076/eotlayer/issues).
For installation and numerical errors, see [Troubleshooting](development/troubleshooting.md).

## Indices and search

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
