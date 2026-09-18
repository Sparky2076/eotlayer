# Differentiation modes

## Unrolled: differentiate the executed algorithm

`EOTLayer(epsilon, "unrolled", iter_num=K)` returns the plan after exactly $K$
Sinkhorn updates. PyTorch retains the iteration graph and differentiates that
finite computation. No marginal-convergence criterion is applied.

This mode is useful when the finite iteration count is part of the model or when
checking small problems. Retaining the iteration graph makes memory use grow
with the number of updates. Only `iter_num` is accepted as a mode-specific
configuration argument.

## Implicit: differentiate the fixed-point equations

`implicit` first accepts a plan meeting `forward_tol`. Its custom backward solves
the fixed-point adjoint equations at that approximate plan, using the prescribed
marginals as the diagonal blocks. It does not differentiate the stopping decision
or retain the full forward iteration graph.

Changing `backward_solver` selects how the same adjoint system is solved. The
resulting derivative approaches the converged fixed-point derivative as the
forward and backward errors become small, subject to conditioning and precision.
The custom backward supports first derivatives only.

## Envelope: differentiate the regularized value

`envelope` returns a scalar EOT value, or one value per batch item. For upstream
scalar one, its cost gradient is the accepted plan:

```{math}
\nabla_C V_\varepsilon=P.
```

The marginal gradients use dual potentials with the final target potential fixed
to zero. This removes the additive gauge freedom. On the equal-mass domain,
gradients with respect to marginals are interpreted along feasible perturbations.

This mode does not solve a backward linear system. Supplying `backward_solver`,
`backward_tol`, or `backward_iter_max` is therefore an error. Its custom backward
also supports first derivatives only.

## Which mode should I use?

| Your objective | Mode |
| --- | --- |
| A loss depending on a fixed number of Sinkhorn updates | `unrolled` |
| A loss depending on an approximately converged transport plan | `implicit` |
| The regularized EOT optimum itself | `envelope` |

For a loss that depends on the transport **plan**, an envelope gradient is not a
replacement for a plan VJP. Conversely, computing a plan VJP is unnecessary when
you only need the gradient of the regularized optimum value.

## Finite-plan analytical derivatives

The internal research API also contains `transport_backward_from_plan`. It uses
the actual row and column sums of a supplied finite plan in its adjoint system.
That analytical derivative is distinct from both finite unrolling and the public
`implicit` mode. It is not exported by the supported top-level package interface.
