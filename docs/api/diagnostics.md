# Diagnostics

`layer.diagnostics` is a tuple containing one record for each batch item, or a
single record for an unbatched call. Every access returns copied scalar data;
it does not retain an extra computation graph.

```python
plan = layer(cost, source, target)
before_backward = layer.diagnostics
loss = (plan * weights).sum()
loss.backward()
after_backward = layer.diagnostics
```

This fragment assumes the tensors, `weights`, and `layer` have been created.
`before_backward` remains unchanged. Use `after_backward` for solve results.

## Top-level fields

| Field | Meaning |
| --- | --- |
| `mode` | `unrolled`, `implicit`, or `envelope` |
| `batch_index` | Zero-based index within the latest call |
| `forward` | Forward measurements and status |
| `backward` | Backward measurements and status |

## Forward fields

| Field | Meaning |
| --- | --- |
| `status` | `pending`, `completed`, `converged`, `not_converged`, or `error` |
| `iterations` | Number of completed Sinkhorn updates |
| `absolute_marginal_error` | Maximum of the row and column L1 marginal errors |
| `normalized_marginal_error` | Maximum L1 error divided by the respective total mass |
| `tolerance` | Configured forward tolerance, or `None` in unrolled mode |
| `error` | Error message, or `None` |

Unrolled completion reports `completed` regardless of marginal convergence.
Accepted implicit and envelope forwards report `converged`.

## Backward fields

| Field | Meaning |
| --- | --- |
| `status` | `pending`, `not_applicable`, `completed`, `converged`, `not_converged`, or `error` |
| `solver` | Requested solver name, or `None` |
| `selected_solver` | Solver reported after a successful implicit attempt |
| `iterations` | Reported Krylov iterations; Dense reports zero |
| `initial_relative_residual` | Exact-system relative residual of the initial guess |
| `relative_residual` | Exact full-plan scaled source Schur residual |
| `requested_tolerance` | User-selected implicit backward tolerance |
| `effective_tolerance` | Acceptance threshold, including the Dense dtype floor |
| `initial_guess_used` | Whether the solver used a supplied initial guess |
| `error` | Error message, or `None` |

Some fields stay `None` when no solve occurred or an exception happened before a
measurement was produced. Unrolled backward is `not_applicable` to these custom
diagnostics because ordinary PyTorch autograd performs it. Successful envelope
backward is `completed` and has no linear solve residual.

## Repeated calls and failures

The property describes the most recent forward attempt. Each outstanding graph
updates its own diagnostic record, so backward on an older graph does not relabel
the latest forward's records. Invalid input metadata clears the records. When
a solver fails after a record exists, inspect its `status` and `error` fields.

Residual definitions and their limits are described in
[Numerical accuracy](../key_concepts/numerical_accuracy.md).
