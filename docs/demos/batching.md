# Batched transport

For a cost tensor shaped `(B, m, n)`, marginals can be shared vectors or matching
batched matrices. Each marginal independently supports these two forms.

```{literalinclude} ../../examples/docs_batching.py
:language: python
:start-after: docs-start
:end-before: docs-end
```

## Shapes and gradients

The first call uses shared `source` and `target` vectors; the second uses a
separate source vector per item and a shared target vector. Both return `(2, 3, 3)`.
Shared marginal gradients accumulate contributions from the batch through normal
PyTorch autograd.

For `envelope`, a batched call returns shape `(B,)`, so reduce the values before
calling `.backward()` or supply an explicit upstream vector.

The current implementation evaluates each batch item sequentially. This example
demonstrates shape semantics, not batched-kernel throughput.

```bash
python examples/docs_batching.py
```
