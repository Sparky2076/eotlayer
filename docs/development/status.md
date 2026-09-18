# Project status

## Current scope

EOTLayer is a pre-alpha PyTorch research package for differentiable balanced
entropic optimal transport. The included implementation exposes the unrolled,
implicit, and envelope contracts behind one public layer.

The project is intended as software, systems, and benchmarking work. It does not
claim the first implementation of implicit EOT differentiation or matrix-free EOT
backward.

## Supported interface

- Dense float32/float64 cost tensors, unbatched or batched.
- Strictly positive, balanced source and target marginals.
- Log-domain Sinkhorn and optional continuation in converged modes.
- Four canonical implicit backend names with a common residual target.
- Per-item forward and backward diagnostic snapshots.

## Research and validation boundaries

The source archive contains CPU regression coverage for Dense, Jacobi-W, and
IC(0), plus internal sparse tests with injected factorizers. Such tests do not
establish that the public cuDSS path runs correctly or quickly on a GPU.

The experimental `auto` selector, effective-resistance support construction,
internal slack-matching formulation, and Flash-Sinkhorn adapters are outside the
top-level interface documented for normal use. Native AMGX is not one of the
canonical backends in this snapshot.

No historical timing, speedup, or memory result is presented here as a fresh
benchmark. Comparing backends requires the same forward state, upstream gradient,
right-hand side, dtype/device, and acceptance residual. Complete backward time
includes setup, solve, checks, and gradient reconstruction.

## Documentation snapshot

These pages describe version `0.1.0.dev0` from the supplied EOTLayer source archive.
The [source repository](https://github.com/Sparky2076/eotlayer) includes the runtime
source needed to reproduce these examples. The `stable` branch is a documentation
deployment target; it does not change the package's pre-alpha maturity or its
`0.1.0.dev0` version. Read the Docs builds are configured in `.readthedocs.yaml`.
