# Compare differentiation modes

Compare a finite unrolled plan with two implicit solves on the same deterministic
problem. Dense and Jacobi-W target the same fixed-point equations; the finite
unrolled mode differentiates the executed iteration sequence.

```{literalinclude} ../../examples/docs_compare_modes.py
:language: python
:start-after: docs-start
:end-before: docs-end
```

## Interpret the comparison

`gradient difference` reports a relative difference against the Dense implicit
cost gradient. The example checks that the two implicit methods agree on this
small float64 problem. It does not require the 8-step unrolled gradient to match
the fixed-point gradient.

This is a numerical demonstration, not a performance benchmark: there are no
timing, warmup, memory, or statistical controls in this script.

```bash
python examples/docs_compare_modes.py
```
