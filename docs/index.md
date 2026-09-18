# Welcome to EOTLayer

```{raw} html
<div class="eot-eyebrow">Differentiable optimal transport · PyTorch</div>
<p class="eot-intro">Entropic optimal transport, as a PyTorch layer. Compute transport plans, differentiate through them, and choose the backward solver that fits your problem.</p>
<div class="eot-meta"><span>0.1.0.dev0</span><span>Python ≥ 3.10</span><span>PyTorch</span><span>Pre-alpha</span></div>
```

EOTLayer solves balanced entropic optimal transport (EOT) with log-domain
Sinkhorn iterations. A single interface exposes three differentiation modes:
finite **unrolling**, **implicit** plan differentiation, and the **envelope**
gradient of the regularized transport value.

```{note}
EOTLayer is research software. The API is documented against the included source
snapshot. GPU backends and performance claims require workload-specific validation;
see [Project status](development/status.md).
```

## Quick start

### Installation

From the root of the EOTLayer source checkout:

```bash
python -m pip install -e .
```

See [Installation](installation.md) for environment setup, optional CUDA support,
and building these docs. These instructions use the source package; a PyPI
release is not assumed.

### Basic usage

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

## Choose a differentiation mode

| Mode | Returns | Differentiates |
| --- | --- | --- |
| `unrolled` | Transport plan | Exactly `iter_num` Sinkhorn iterations |
| `implicit` | Transport plan | The fixed-point equations at an accepted approximate plan |
| `envelope` | Regularized EOT value | The value using the plan and dual potentials |

Read [Differentiation modes](key_concepts/differentiation.md) before comparing
gradients: a finite unrolled derivative and an approximate fixed-point derivative
need not agree.

## Key features

- **One PyTorch interface.** Use `EOTLayer` with tensors, losses, and autograd.
- **Stable forward updates.** Log-domain Sinkhorn with optional epsilon continuation.
- **Interchangeable backward solvers.** Dense, Jacobi-W, CPU IC(0), and CUDA Sparse-W.
- **Explicit numerical diagnostics.** Inspect marginal errors, exact-system residuals,
  iteration counts, and effective tolerances.
- **Shared and batched marginals.** Solve one problem or a batch using the same call
  signature. The current implementation processes batch items sequentially.

## Explore the documentation

```{raw} html
<div class="eot-grid">
  <a class="eot-card" href="installation.html"><strong>Installation →</strong><span>Set up EOTLayer and run your first transport problem.</span></a>
  <a class="eot-card" href="key_concepts/index.html"><strong>Key concepts →</strong><span>Understand transport, differentiation, and solver choices.</span></a>
  <a class="eot-card" href="demos/index.html"><strong>Demos and examples →</strong><span>Run complete examples for plans, values, and batches.</span></a>
  <a class="eot-card" href="api/index.html"><strong>API reference →</strong><span>Find parameters, tensor shapes, and diagnostic fields.</span></a>
</div>
```

```{toctree}
:maxdepth: 2
:caption: Documentation
:hidden:

installation
key_concepts/index
demos/index
api/index
development/index
```

## Indices and search

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
