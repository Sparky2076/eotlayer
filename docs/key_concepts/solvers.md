# Backward solvers

## A shared adjoint target

The `implicit` backward reduces the adjoint equations to a scaled source Schur
system. After fixing the final target potential to zero, define

```{math}
R=D_a^{-1/2}\bar{P}D_{\bar{b}}^{-1/2},\qquad S=I-RR^\top.
```

Here $\bar{P}$ and $\bar{b}$ omit the grounded target column. Each backend solves
$Sx=r$ for the same plan and right-hand side. At a strictly positive balanced
solution, grounding removes the gauge nullspace. At finite precision or with an
inexact forward state, conditioning and positive definiteness still need care.

## Available backends

| Name | Implementation | Device boundary | Typical role |
| --- | --- | --- | --- |
| `dense` | Explicit Schur matrix and dense solves, with residual-based refinement | PyTorch operations on the input device | Small-problem reference |
| `jacobi_w` | PCG on the full-plan scaled Schur operator | PyTorch operations on the input device | Matrix-free baseline |
| `ic0` | PCG with a block IC(0) preconditioner | CPU only | Sparse preconditioning experiments |
| `sparse_w` | PCG with a sparse proxy factored through cuDSS | CUDA and optional cuDSS integration | GPU sparse preconditioning experiments |

CPU regression coverage does not certify GPU behavior. No speed ranking is
implied by this table. `auto` also exists in the source but remains experimental;
the four names above are the canonical backends.

## Dense

Dense explicitly constructs $S$ and solves it using `torch.linalg.solve`.
It can retry with numerical stabilization and refinement, but acceptance is
always measured against the original full-plan operator. Its effective tolerance
is

```{math}
\max\bigl(\texttt{backward\_tol},100\,\epsilon_{\mathrm{machine}}\bigr).
```

It reports zero Krylov iterations. This does not mean the backward did no work.
The public `backward_iter_max` setting does not cap its direct factorization trials.

## Jacobi-W

Jacobi-W applies $S$ through products with $P$ and $P^\top$. It avoids materializing
the $m\times m$ Schur matrix, but still stores the dense transport plan. Each
matrix-vector product uses the full current plan.

PCG confirms apparent convergence by recomputing the true residual. If the
recursively updated residual has drifted, it restarts from the true residual.
The requested `backward_tol` is used without the Dense tolerance floor.

## IC(0) and Sparse-W

These methods sparsify an auxiliary preconditioner. They keep the full-plan
operator and its residual as the acceptance target. The public layer currently
uses `dropped_mass=0.005` and `diagonal_shift=1e-10` internally; these are not
constructor parameters on `EOTLayer`.

Sparse-W's mass-based support is separate from the experimental effective-resistance
support implementation. Effective-resistance sampling has additional setup and
accuracy costs, and is not a finite-run spectral certificate.

## Selecting a backend

Use `dense` for a small reference computation. Compare `jacobi_w` against the same
plan and upstream gradient before exploring IC(0) or Sparse-W. Include support
construction, factorization, solve, residual checks, and gradient reconstruction
when measuring backward time.

See [Numerical accuracy](numerical_accuracy.md) for what a passing residual means.
