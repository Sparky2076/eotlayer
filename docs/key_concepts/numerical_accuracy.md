# Numerical accuracy

## Forward stopping

The forward normalized marginal error is

```{math}
e_{\mathrm{forward}}=\max\left(
\frac{\|P\mathbf{1}-a\|_1}{\sum_i a_i},
\frac{\|P^\top\mathbf{1}-b\|_1}{\sum_j b_j}
\right).
```

`implicit` and `envelope` require this value to be at most `forward_tol` at the
target epsilon. Reaching `forward_iter_max` without acceptance raises an error.
The internal forward configuration checks every ten iterations by default and
at the final iteration. `unrolled` reports the error but does not use it to stop.

## Backward stopping

For the scaled source Schur system $Sx=r$, the residual is

```{math}
e_{\mathrm{backward}}=
\frac{\|Sx-r\|_2}{\max(\|r\|_2,\operatorname{tiny}(\mathrm{dtype}))}.
```

The backend adapter independently recomputes this residual before gradient
reconstruction. Sparse supports are not substituted into the acceptance operator.
Dense has a dtype-dependent effective tolerance; other canonical backends use the
requested tolerance. Check the reported value, especially in float32.

## Residuals and gradient errors

A small linear residual concerns the solve at the supplied state. It does not
alone certify an accurate outer gradient: forward error, conditioning, gauge
handling, and numerical underflow also matter. Compare gradients against an
appropriate reference when accuracy is important.

## Precision and small epsilon

The default `forward_tol=1e-10` can be too strict for float32. Use float64 for
reference work and choose an attainable tolerance for the input precision. Small
epsilon can produce concentrated plans, slow convergence, and numerical loss of
connectivity. Epsilon continuation can help the forward path but does not remove
these limitations.

## Memory and batching

The public layer accepts a dense cost and returns a dense plan. Matrix-free
backward avoids an explicit Schur matrix, not the $m\times n$ plan. Batch inputs
are processed in a Python loop; batch support does not imply fused GPU kernels.

## Autograd boundary

The custom implicit and envelope paths support first derivatives only. Avoid
in-place modification of tensors needed by an outstanding backward graph. Use
new diagnostics snapshots after backward and do not close a layer while it still
has pending backward work.
