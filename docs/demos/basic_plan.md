# Basic plan and gradient

This example matches three source masses to three target masses, then
differentiates a weighted plan loss with respect to the costs. The weights are
chosen so the loss is not constant on the transport polytope.

```{literalinclude} ../../examples/docs_quickstart.py
:language: python
:start-after: docs-start
:end-before: docs-end
:linenos:
```

## What to check

The row sums approach `[0.20, 0.30, 0.50]`; the column sums approach
`[0.40, 0.35, 0.25]`. `cost.grad` has shape `(3, 3)` and finite entries.
The diagnostic dictionary is read after backward so that it contains the
linear solve's final status and residual.

The runnable file includes checks for these properties. Its exact floating-point
output may vary across PyTorch versions and hardware.

## Run the example

```bash
python examples/docs_quickstart.py
```

For a different backward method, change `backward_solver` while keeping the cost,
marginals, epsilon, and upstream weight matrix fixed. See
[Backward solvers](../key_concepts/solvers.md).
