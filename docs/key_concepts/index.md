# Key concepts

The core workflow is to construct a cost matrix, choose what you want to
differentiate, solve the transport problem, and inspect both forward and backward
accuracy.

```{toctree}
:maxdepth: 1

optimal_transport
differentiation
solvers
workflow
numerical_accuracy
```

Start with [Entropic optimal transport](optimal_transport.md) for the mathematical
problem. [Differentiation modes](differentiation.md) explains the different
derivative contracts. [Backward solvers](solvers.md) describes which computations
can be exchanged without changing the adjoint target.
