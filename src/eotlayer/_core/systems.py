"""Exact Jacobi-scaled Schur systems for EOT plan-through adjoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import torch

from .constraints import BalancedConstraints, SlackMatchingConstraints
from ..forward import TransportState
from .layout import SchurLayout


MarginalLinearization = Literal["target", "plan"]


@runtime_checkable
class RowBlockTensor(Protocol):
    """Matrix-like tensor provider with bounded row materialization."""

    @property
    def shape(self) -> torch.Size:
        ...

    @property
    def dtype(self) -> torch.dtype:
        ...

    @property
    def device(self) -> torch.device:
        ...

    def row_block(self, start: int, stop: int) -> torch.Tensor:
        ...


@runtime_checkable
class PlanWeightedRowBlockTensor(Protocol):
    """Cotangent that can form ``P * G`` in a stable operation order."""

    @property
    def shape(self) -> torch.Size:
        ...

    @property
    def dtype(self) -> torch.dtype:
        ...

    @property
    def device(self) -> torch.device:
        ...

    def weighted_plan_row_block(
        self,
        plan_block: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        ...


PlanCotangent: TypeAlias = torch.Tensor | RowBlockTensor


def validate_marginal_linearization(value: str) -> MarginalLinearization:
    if value not in {"target", "plan"}:
        raise ValueError("marginal_linearization must be 'target' or 'plan'")
    return value


def _validate_grad_plan_metadata(
    state: TransportState, grad_plan: PlanCotangent
) -> None:
    if not isinstance(grad_plan, (torch.Tensor, RowBlockTensor)):
        raise TypeError("grad_plan must be a Tensor or RowBlockTensor")
    if grad_plan.shape != state.plan.shape:
        raise ValueError("grad_plan must have the same shape as the visible plan")
    if grad_plan.dtype != state.plan.dtype or grad_plan.device != state.plan.device:
        raise ValueError("grad_plan must match the plan dtype and device")


def _grad_plan_row_block(
    plan: torch.Tensor,
    grad_plan: PlanCotangent,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Materialize and validate exactly one cotangent row block."""

    if not 0 <= start < stop <= plan.shape[0]:
        raise ValueError("grad_plan row block is outside the visible plan")
    if isinstance(grad_plan, torch.Tensor):
        block = grad_plan[start:stop]
    else:
        block = grad_plan.row_block(start, stop)
    expected_shape = torch.Size((stop - start, plan.shape[1]))
    if not isinstance(block, torch.Tensor):
        raise TypeError("grad_plan row_block must return a Tensor")
    if block.shape != expected_shape:
        raise ValueError("grad_plan row block has the wrong shape")
    if block.dtype != plan.dtype or block.device != plan.device:
        raise ValueError("grad_plan row block must match the plan dtype and device")
    if not torch.all(torch.isfinite(block)):
        raise ValueError("grad_plan row block must be finite")
    return block


def _weighted_marginal_block(
    plan_block: torch.Tensor, gradient_block: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row and column sums of one bounded ``P * G`` block."""

    weighted = plan_block * gradient_block
    return weighted.sum(1), weighted.sum(0)


def _specialized_weighted_plan_row_block(
    plan: torch.Tensor,
    grad_plan: PlanWeightedRowBlockTensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Materialize and validate one stable, already weighted ``P * G`` block."""

    weighted = grad_plan.weighted_plan_row_block(plan[start:stop], start, stop)
    expected_shape = torch.Size((stop - start, plan.shape[1]))
    if not isinstance(weighted, torch.Tensor):
        raise TypeError("weighted plan row block must return a Tensor")
    if weighted.shape != expected_shape:
        raise ValueError("weighted plan row block has the wrong shape")
    if weighted.dtype != plan.dtype or weighted.device != plan.device:
        raise ValueError("weighted plan row block must match plan dtype and device")
    if not torch.all(torch.isfinite(weighted)):
        raise ValueError("weighted plan row block must be finite")
    return weighted


@dataclass(frozen=True)
class CostVJP:
    """Lazy exact cost cotangent with bounded row-block materialization."""

    plan: torch.Tensor
    grad_plan: PlanCotangent
    epsilon: float
    source_adjoint: torch.Tensor
    target_adjoint: torch.Tensor

    def row_block(self, start: int, stop: int) -> torch.Tensor:
        if not 0 <= start < stop <= self.plan.shape[0]:
            raise ValueError("row block is outside the cost matrix")
        grad_plan_block = _grad_plan_row_block(
            self.plan, self.grad_plan, start, stop
        )
        correction = (
            self.source_adjoint[start:stop, None]
            + self.target_adjoint[None, :]
        )
        return (
            -self.plan[start:stop]
            * (grad_plan_block - correction)
            / float(self.epsilon)
        )

    def iter_row_blocks(self, *, block_rows: int):
        if block_rows <= 0:
            raise ValueError("block_rows must be positive")
        for start in range(0, self.plan.shape[0], block_rows):
            stop = min(start + block_rows, self.plan.shape[0])
            yield start, stop, self.row_block(start, stop)

    def materialize(self) -> torch.Tensor:
        return self.row_block(0, self.plan.shape[0])


@dataclass(frozen=True)
class AdjointSystem:
    """Exact scaled source Schur system and VJP reconstruction data.

    ``plan_block`` is always selected from the complete current plan. Sparse
    supports are absent by construction and may only appear in a backend's
    preconditioner.
    """

    state: TransportState
    grad_plan: PlanCotangent
    marginal_linearization: MarginalLinearization
    layout: SchurLayout
    source_total: torch.Tensor
    target_total: torch.Tensor
    active_target_indices: torch.Tensor
    plan_block: torch.Tensor
    rhs_source: torch.Tensor
    rhs_target: torch.Tensor
    inverse_sqrt_source: torch.Tensor
    inverse_target: torch.Tensor
    scaled_rhs: torch.Tensor

    @property
    def grounded(self) -> bool:
        return self.layout.grounded_target_index is not None

    @property
    def source_dimension(self) -> int:
        return int(self.source_total.numel())

    @property
    def target_dimension(self) -> int:
        return int(self.active_target_indices.numel())

    @property
    def structural_signature(self) -> tuple[object, ...]:
        device = self.state.plan.device
        return (
            type(self.state.constraints).__qualname__,
            tuple(self.state.plan.shape),
            self.state.plan.dtype,
            device.type,
            device.index,
            self.layout.active_target_indices,
            self.layout.grounded_target_index,
            self.marginal_linearization,
        )

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply ``I - R R^T`` with the complete current plan."""

        if (
            vector.ndim != 1
            or vector.numel() != self.source_dimension
            or vector.dtype != self.scaled_rhs.dtype
            or vector.device != self.scaled_rhs.device
        ):
            raise ValueError("vector is incompatible with the adjoint system")
        inner = self.plan_block.T @ (self.inverse_sqrt_source * vector)
        correction = self.inverse_sqrt_source * (
            self.plan_block @ (self.inverse_target * inner)
        )
        return vector - correction

    def dense_matrix(self) -> torch.Tensor:
        """Materialize the same exact Schur operator for the dense baseline."""

        normalized = (
            self.inverse_sqrt_source[:, None]
            * self.plan_block
            * torch.sqrt(self.inverse_target)[None, :]
        )
        identity = torch.eye(
            self.source_dimension,
            dtype=self.state.plan.dtype,
            device=self.state.plan.device,
        )
        return identity - normalized @ normalized.T

    def reconstruct_vjp(
        self, scaled_solution: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cost_vjp, source_adjoint, target_adjoint = self.reconstruct_cost_vjp(
            scaled_solution
        )
        return cost_vjp.materialize(), source_adjoint, target_adjoint

    def reconstruct_cost_vjp(
        self, scaled_solution: torch.Tensor
    ) -> tuple[CostVJP, torch.Tensor, torch.Tensor]:
        """Recover lazy cost and explicit marginal cotangents."""

        if (
            scaled_solution.ndim != 1
            or scaled_solution.numel() != self.source_dimension
            or scaled_solution.dtype != self.scaled_rhs.dtype
            or scaled_solution.device != self.scaled_rhs.device
        ):
            raise ValueError("scaled solution is incompatible with the system")
        source_adjoint = self.inverse_sqrt_source * scaled_solution
        active_target_adjoint = self.inverse_target * (
            self.rhs_target - self.plan_block.T @ source_adjoint
        )
        target_adjoint = torch.zeros_like(self.target_total)
        target_adjoint.index_copy_(
            0, self.active_target_indices, active_target_adjoint
        )
        cost_vjp = CostVJP(
            plan=self.state.plan,
            grad_plan=self.grad_plan,
            epsilon=float(self.state.epsilon),
            source_adjoint=source_adjoint,
            target_adjoint=target_adjoint,
        )
        return cost_vjp, source_adjoint, target_adjoint


def _build_adjoint_system(
    state: TransportState,
    grad_plan: PlanCotangent,
    *,
    marginal_linearization: MarginalLinearization,
    rhs_block_rows: int,
    require_converged: bool,
) -> AdjointSystem:
    """Assemble one grounded-balanced or ungrounded-slack Schur system."""

    if require_converged and not state.converged:
        raise RuntimeError("implicit adjoints require a converged forward fixed point")
    _validate_grad_plan_metadata(state, grad_plan)
    marginal_linearization = validate_marginal_linearization(
        marginal_linearization
    )
    if rhs_block_rows <= 0:
        raise ValueError("rhs_block_rows must be positive")
    constraints = state.constraints
    if isinstance(constraints, BalancedConstraints):
        if marginal_linearization == "plan":
            source_total = state.plan.sum(1)
            target_total = state.plan.sum(0)
        else:
            source_total = constraints.source
            target_total = constraints.target
        layout = SchurLayout.balanced(target_dimension=target_total.numel())
    elif isinstance(constraints, SlackMatchingConstraints):
        if marginal_linearization == "plan":
            if state.source_slack is None or state.target_slack is None:
                raise ValueError("slack plan linearization requires stored slack values")
            source_total = state.plan.sum(1) + state.source_slack
            target_total = state.plan.sum(0) + state.target_slack
        else:
            source_total = constraints.source_total
            target_total = constraints.target_total
        layout = SchurLayout.slack(target_dimension=target_total.numel())
    else:  # pragma: no cover - guarded by EOTProblem
        raise TypeError(f"unsupported constraint type: {type(constraints)!r}")

    active_target_indices = torch.tensor(
        layout.active_target_indices, dtype=torch.long, device=state.plan.device
    )
    active_tuple = layout.active_target_indices
    if active_tuple == tuple(range(len(active_tuple))):
        plan_block = state.plan[:, : len(active_tuple)]
    elif active_tuple == tuple(range(state.plan.shape[1])):
        plan_block = state.plan
    else:
        plan_block = state.plan.index_select(1, active_target_indices)
    target_block = target_total.index_select(0, active_target_indices)
    rhs_source = torch.empty(
        state.plan.shape[0], dtype=state.plan.dtype, device=state.plan.device
    )
    rhs_target_full = torch.zeros_like(target_total)
    for start in range(0, state.plan.shape[0], rhs_block_rows):
        stop = min(start + rhs_block_rows, state.plan.shape[0])
        if isinstance(grad_plan, PlanWeightedRowBlockTensor):
            weighted = _specialized_weighted_plan_row_block(
                state.plan, grad_plan, start, stop
            )
            source_block = weighted.sum(1)
            target_block_contribution = weighted.sum(0)
        else:
            gradient_block = _grad_plan_row_block(
                state.plan, grad_plan, start, stop
            )
            source_block, target_block_contribution = _weighted_marginal_block(
                state.plan[start:stop], gradient_block
            )
        rhs_source[start:stop] = source_block
        rhs_target_full.add_(target_block_contribution)
    rhs_target = rhs_target_full.index_select(0, active_target_indices)
    inverse_sqrt_source = torch.rsqrt(source_total)
    inverse_target = torch.reciprocal(target_block)
    scaled_rhs = inverse_sqrt_source * (
        rhs_source - plan_block @ (inverse_target * rhs_target)
    )
    return AdjointSystem(
        state=state,
        grad_plan=grad_plan,
        marginal_linearization=marginal_linearization,
        layout=layout,
        source_total=source_total,
        target_total=target_total,
        active_target_indices=active_target_indices,
        plan_block=plan_block,
        rhs_source=rhs_source,
        rhs_target=rhs_target,
        inverse_sqrt_source=inverse_sqrt_source,
        inverse_target=inverse_target,
        scaled_rhs=scaled_rhs,
    )


def build_adjoint_system(
    state: TransportState,
    grad_plan: PlanCotangent,
    *,
    marginal_linearization: MarginalLinearization = "target",
    rhs_block_rows: int = 128,
) -> AdjointSystem:
    """Build the exact adjoint system of a converged EOT fixed point."""

    return _build_adjoint_system(
        state,
        grad_plan,
        marginal_linearization=marginal_linearization,
        rhs_block_rows=rhs_block_rows,
        require_converged=True,
    )


def build_finite_plan_adjoint_system(
    state: TransportState,
    grad_plan: PlanCotangent,
    *,
    rhs_block_rows: int = 128,
) -> AdjointSystem:
    """Build Eisenberger's analytical system at a finite-iteration plan.

    The diagonal blocks use the actual row and column sums of ``state.plan``.
    This is the analytical fixed-point formula evaluated at an approximate
    plan; it is intentionally distinct from differentiating the finite
    Sinkhorn iteration map and from the converged fixed-point contract exposed
    by :func:`build_adjoint_system`.
    """

    if not isinstance(state.constraints, BalancedConstraints):
        raise TypeError("finite-plan adjoints currently require balanced constraints")
    return _build_adjoint_system(
        state,
        grad_plan,
        marginal_linearization="plan",
        rhs_block_rows=rhs_block_rows,
        require_converged=False,
    )
