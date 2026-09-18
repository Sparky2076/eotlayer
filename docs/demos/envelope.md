# Regularized transport value

Use `envelope` when the outer objective is the regularized EOT value itself.
The value includes the entropy term and is not just `sum(cost * plan)`.

```{literalinclude} ../../examples/docs_envelope.py
:language: python
:start-after: docs-start
:end-before: docs-end
```

## Check the envelope identity

For scalar upstream derivative one, `cost.grad` equals the accepted plan. This
example computes a separate reference plan using identical forward settings and
checks the equality. Source and target gradients are grounded dual potentials;
the final target component is zero by the selected gauge.

No adjoint linear solver runs in envelope mode. The backward diagnostic status is
`completed`, while linear residual and solver fields remain `None`.

```bash
python examples/docs_envelope.py
```
