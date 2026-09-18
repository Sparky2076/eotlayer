"""Plan-through implicit VJP for converged EOT fixed points."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable

import torch

from .backends import (
    AdjointBackend,
    DenseBackend,
    JacobiWBackend,
    LinearSolveResult,
    solve_with_backend_contract,
)
from .forward import TransportState
from .backends.sparse import IC0Backend, SparseWBackend
from ._core.systems import (
    AdjointSystem,
    CostVJP,
    MarginalLinearization,
    PlanCotangent,
    build_adjoint_system,
    build_finite_plan_adjoint_system,
    validate_marginal_linearization,
)


@dataclass(frozen=True)
class BackwardConfig:
    """Canonical method and stopping policy for a stateless adjoint call."""

    method: str = "dense"
    tolerance: float = 1e-8
    maximum_iterations: int = 1_000
    residual_check_interval: int = 1
    validate_spd: bool = True
    marginal_linearization: MarginalLinearization = "target"
    dropped_mass: float = 0.005
    diagonal_shift: float = 1e-10
    rhs_block_rows: int = 128
    materialize_grad_cost: bool = True

    def __post_init__(self) -> None:
        if self.method not in {"dense", "jacobi_w", "ic0", "sparse_w"}:
            raise ValueError("method must be dense, jacobi_w, ic0, or sparse_w")
        if (
            not math.isfinite(float(self.tolerance))
            or self.tolerance <= 0
            or self.maximum_iterations <= 0
            or self.residual_check_interval <= 0
        ):
            raise ValueError("invalid backward stopping configuration")
        validate_marginal_linearization(self.marginal_linearization)
        if not 0.0 <= self.dropped_mass < 1.0:
            raise ValueError("dropped_mass must lie in [0, 1)")
        if not math.isfinite(self.diagonal_shift) or self.diagonal_shift < 0:
            raise ValueError("diagonal_shift must be finite and nonnegative")
        if self.rhs_block_rows <= 0:
            raise ValueError("rhs_block_rows must be positive")


@dataclass(frozen=True)
class TransportBackwardResult:
    grad_cost: torch.Tensor | None
    cost_vjp: CostVJP
    grad_source: torch.Tensor
    grad_target: torch.Tensor
    source_adjoint: torch.Tensor
    target_adjoint: torch.Tensor
    scaled_source_solution: torch.Tensor
    iterations: int
    initial_relative_residual: float
    relative_residual: float
    converged: bool
    linear_result: LinearSolveResult


def result_from_linear_solve(
    system: AdjointSystem,
    result: LinearSolveResult,
    *,
    materialize_grad_cost: bool = True,
) -> TransportBackwardResult:
    cost_vjp, source_adjoint, target_adjoint = system.reconstruct_cost_vjp(
        result.solution
    )
    grad_cost = cost_vjp.materialize() if materialize_grad_cost else None
    return TransportBackwardResult(
        grad_cost=grad_cost,
        cost_vjp=cost_vjp,
        grad_source=source_adjoint,
        grad_target=target_adjoint,
        source_adjoint=source_adjoint,
        target_adjoint=target_adjoint,
        scaled_source_solution=result.solution.detach(),
        iterations=result.iterations,
        initial_relative_residual=result.initial_relative_residual,
        relative_residual=result.relative_residual,
        converged=result.converged,
        linear_result=result,
    )


def _backend_from_config(config: BackwardConfig) -> AdjointBackend:
    common = dict(
        tolerance=config.tolerance,
        maximum_iterations=config.maximum_iterations,
        residual_check_interval=config.residual_check_interval,
        validate_spd=config.validate_spd,
    )
    if config.method == "dense":
        return DenseBackend(tolerance=config.tolerance)
    if config.method == "jacobi_w":
        return JacobiWBackend(**common)
    if config.method == "ic0":
        return IC0Backend(
            dropped_mass=config.dropped_mass,
            diagonal_shift=config.diagonal_shift,
            **common,
        )
    return SparseWBackend(
        dropped_mass=config.dropped_mass,
        diagonal_shift=config.diagonal_shift,
        **common,
    )


def transport_backward(
    state: TransportState,
    grad_plan: PlanCotangent,
    *,
    config: BackwardConfig | None = None,
    initial_guess: torch.Tensor | None = None,
    backend: AdjointBackend | None = None,
    slot_key: Hashable | None = None,
    marginal_linearization: MarginalLinearization | None = None,
    rhs_block_rows: int | None = None,
    materialize_grad_cost: bool | None = None,
) -> TransportBackwardResult:
    """Apply an exact fixed-point VJP using one interchangeable backend."""

    if backend is not None and config is not None:
        raise ValueError("an explicit backend owns solver configuration")
    resolved_config = config or BackwardConfig()
    selected_linearization = (
        resolved_config.marginal_linearization
        if marginal_linearization is None
        else validate_marginal_linearization(marginal_linearization)
    )
    if (
        config is not None
        and marginal_linearization is not None
        and selected_linearization != config.marginal_linearization
    ):
        raise ValueError("marginal_linearization conflicts with BackwardConfig")
    system = build_adjoint_system(
        state,
        grad_plan,
        marginal_linearization=selected_linearization,
        rhs_block_rows=(
            resolved_config.rhs_block_rows
            if rhs_block_rows is None
            else rhs_block_rows
        ),
    )
    owns_backend = backend is None
    selected_backend = backend or _backend_from_config(resolved_config)
    try:
        linear_result = solve_with_backend_contract(
            selected_backend,
            system,
            slot_key=slot_key,
            initial_guess=initial_guess,
        )
        return result_from_linear_solve(
            system,
            linear_result,
            materialize_grad_cost=(
                resolved_config.materialize_grad_cost
                if materialize_grad_cost is None
                else bool(materialize_grad_cost)
            ),
        )
    finally:
        if owns_backend:
            selected_backend.close()


def transport_backward_from_plan(
    state: TransportState,
    grad_plan: PlanCotangent,
    *,
    config: BackwardConfig | None = None,
    initial_guess: torch.Tensor | None = None,
    backend: AdjointBackend | None = None,
    slot_key: Hashable | None = None,
    rhs_block_rows: int | None = None,
    materialize_grad_cost: bool | None = None,
) -> TransportBackwardResult:
    """Apply Eisenberger's analytical VJP to a finite-iteration plan.

    This explicit entry point does not claim to differentiate the finite
    Sinkhorn algorithm.  It evaluates the fixed-point analytical formula at
    the supplied approximate plan, using that plan's actual row and column
    sums.  The normal :func:`transport_backward` convergence guard remains
    unchanged.
    """

    if backend is not None and config is not None:
        raise ValueError("an explicit backend owns solver configuration")
    resolved_config = config or BackwardConfig(marginal_linearization="plan")
    if resolved_config.marginal_linearization != "plan":
        raise ValueError(
            "transport_backward_from_plan requires marginal_linearization='plan'"
        )
    system = build_finite_plan_adjoint_system(
        state,
        grad_plan,
        rhs_block_rows=(
            resolved_config.rhs_block_rows
            if rhs_block_rows is None
            else rhs_block_rows
        ),
    )
    owns_backend = backend is None
    selected_backend = backend or _backend_from_config(resolved_config)
    try:
        linear_result = solve_with_backend_contract(
            selected_backend,
            system,
            slot_key=slot_key,
            initial_guess=initial_guess,
        )
        return result_from_linear_solve(
            system,
            linear_result,
            materialize_grad_cost=(
                resolved_config.materialize_grad_cost
                if materialize_grad_cost is None
                else bool(materialize_grad_cost)
            ),
        )
    finally:
        if owns_backend:
            selected_backend.close()
