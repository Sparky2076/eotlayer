# Backend contract

## Public selection

Pass one canonical name to `EOTLayer(..., mode="implicit", backward_solver=...)`:
`dense`, `jacobi_w`, `ic0`, or `sparse_w`.
There is no public constructor argument for passing an arbitrary backend object.
Custom adapters use the internal research API.

## What an adapter must declare

The internal `BackendAdapterContract` records the solver name, accepted devices
and dtypes, state scope, initial-guess policy, requested tolerance, tolerance
floor, iteration cap, declared fallbacks, and residual operator identifier.

`solve_with_backend_contract` checks the returned solution's shape, dtype, device,
finiteness, independent storage, and iteration count. It verifies that inputs
were not modified in place and that reported residuals and convergence agree with
independent full-plan measurements.

## Exact operator, approximate preconditioner

The residual operator is `full_plan_scaled_source_schur`. A sparse or low-rank
preconditioner may accelerate the solve, but it must not redefine the accepted
system. A regularized factorization used internally is still judged against the
original system.

## Result metadata

The internal `LinearSolveResult` carries `solution`, `iterations`,
`initial_relative_residual`, `relative_residual`, `converged`, and
`initial_guess_used`. A declared fallback must be visible in the corresponding
diagnostics and remain within the adapter's total iteration contract.

## Stability boundary

This page explains the architecture for contributors. The stable public-facing
entry point in this source snapshot remains `EOTLayer`. Internal adapter classes,
stateful sessions, finite-plan VJPs, and effective-resistance constructors may
change. See [Project status](../development/status.md).
