"""Explicit ownership of forward/backward warm starts and backend state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Hashable

import torch

from ..backends import AdjointBackend, DenseBackend, solve_with_backend_contract
from ..backward import TransportBackwardResult, result_from_linear_solve
from .constraints import (
    BalancedConstraints,
    SlackMatchingConstraints,
    constraint_vectors,
)
from ..forward import (
    ForwardConfig,
    TransportState,
    _validate_initial_scalings,
    transport_forward,
)
from .problem import EOTProblem
from .systems import (
    MarginalLinearization,
    PlanCotangent,
    build_adjoint_system,
    validate_marginal_linearization,
)


@dataclass(frozen=True)
class _ForwardWarmStart:
    signature: tuple[object, ...]
    source: torch.Tensor
    target: torch.Tensor
    log_left: torch.Tensor
    log_right: torch.Tensor


@dataclass(frozen=True)
class _BackwardWarmStart:
    signature: tuple[object, ...]
    declared_source: torch.Tensor
    declared_target: torch.Tensor
    solution: torch.Tensor


def _forward_signature(problem: EOTProblem) -> tuple[object, ...]:
    constraints = problem.constraints
    if isinstance(constraints, BalancedConstraints):
        kind, slack_logit = "balanced", None
    elif isinstance(constraints, SlackMatchingConstraints):
        kind, slack_logit = "slack", float(constraints.slack_logit)
    else:  # pragma: no cover
        raise TypeError(f"unsupported constraint type: {type(constraints)!r}")
    device = problem.cost.device
    return (
        kind,
        tuple(problem.cost.shape),
        problem.cost.dtype,
        device.type,
        device.index,
        float(problem.epsilon),
        slack_logit,
    )


def _validate_backward_initial_guess(
    right_hand_side: torch.Tensor,
    initial_guess: torch.Tensor | None,
) -> None:
    if initial_guess is None:
        return
    if not isinstance(initial_guess, torch.Tensor):
        raise ValueError("initial_guess must be a torch.Tensor")
    if (
        initial_guess.shape != right_hand_side.shape
        or initial_guess.dtype != right_hand_side.dtype
        or initial_guess.device != right_hand_side.device
    ):
        raise ValueError(
            "initial_guess must match the adjoint right-hand side shape, "
            "dtype, and device"
        )
    if not bool(torch.all(torch.isfinite(initial_guess))):
        raise ValueError("initial_guess must contain only finite values")


class EOTSolverSession:
    """Own all warm starts for a sequence of logically keyed EOT solves."""

    def __init__(
        self,
        *,
        adjoint_backend: AdjointBackend | None = None,
        marginal_linearization: MarginalLinearization = "target",
        adjoint_rhs_block_rows: int = 128,
        materialize_grad_cost: bool = True,
        solve_observer: Callable[
            [Hashable | None, TransportState, TransportBackwardResult], None
        ]
        | None = None,
    ) -> None:
        self.adjoint_backend = adjoint_backend or DenseBackend()
        self.marginal_linearization = validate_marginal_linearization(
            marginal_linearization
        )
        if adjoint_rhs_block_rows <= 0:
            raise ValueError("adjoint_rhs_block_rows must be positive")
        self.adjoint_rhs_block_rows = int(adjoint_rhs_block_rows)
        self.materialize_grad_cost = bool(materialize_grad_cost)
        self.solve_observer = solve_observer
        self._forward_warm_starts: dict[Hashable, _ForwardWarmStart] = {}
        self._backward_warm_starts: dict[Hashable, _BackwardWarmStart] = {}
        self._closed = False
        self._lock = RLock()

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("EOT solver session is closed")

    def _invalidate(self, slot_key: Hashable) -> None:
        self._forward_warm_starts.pop(slot_key, None)
        self._backward_warm_starts.pop(slot_key, None)
        try:
            self.adjoint_backend.reset(slot_key)
        except Exception:
            pass

    def forward(
        self,
        problem: EOTProblem,
        *,
        config: ForwardConfig | None = None,
        slot_key: Hashable | None = None,
        initial_log_left: torch.Tensor | None = None,
        initial_log_right: torch.Tensor | None = None,
    ) -> TransportState:
        with self._lock:
            self._require_open()
            signature = _forward_signature(problem)
            source, target = constraint_vectors(problem.constraints)
            _validate_initial_scalings(
                source,
                target,
                initial_log_left,
                initial_log_right,
            )
            selected_left = initial_log_left
            selected_right = initial_log_right
            if slot_key is not None:
                cached = self._forward_warm_starts.get(slot_key)
                if (
                    cached is not None
                    and cached.signature == signature
                    and torch.equal(cached.source, source)
                    and torch.equal(cached.target, target)
                ):
                    selected_left, selected_right = (
                        cached.log_left,
                        cached.log_right,
                    )
            try:
                state = transport_forward(
                    problem,
                    config=config,
                    initial_log_left=selected_left,
                    initial_log_right=selected_right,
                )
            except Exception:
                if slot_key is not None:
                    self._invalidate(slot_key)
                raise
            if slot_key is not None and state.converged:
                self._forward_warm_starts[slot_key] = _ForwardWarmStart(
                    signature,
                    source.detach().clone(),
                    target.detach().clone(),
                    state.log_left.detach().clone(),
                    state.log_right.detach().clone(),
                )
            elif slot_key is not None:
                self._invalidate(slot_key)
            return state

    def backward(
        self,
        state: TransportState,
        grad_plan: PlanCotangent,
        *,
        slot_key: Hashable | None = None,
        initial_guess: torch.Tensor | None = None,
        materialize_grad_cost: bool | None = None,
    ) -> TransportBackwardResult:
        with self._lock:
            self._require_open()
            system = build_adjoint_system(
                state,
                grad_plan,
                marginal_linearization=self.marginal_linearization,
                rhs_block_rows=self.adjoint_rhs_block_rows,
            )
            declared_source, declared_target = constraint_vectors(state.constraints)
            _validate_backward_initial_guess(system.scaled_rhs, initial_guess)
            selected_initial_guess = initial_guess
            if slot_key is not None:
                cached = self._backward_warm_starts.get(slot_key)
                if (
                    cached is not None
                    and cached.signature == system.structural_signature
                    and torch.equal(cached.declared_source, declared_source)
                    and torch.equal(cached.declared_target, declared_target)
                ):
                    selected_initial_guess = cached.solution
                elif cached is not None:
                    self._invalidate(slot_key)
            try:
                linear_result = solve_with_backend_contract(
                    self.adjoint_backend,
                    system,
                    slot_key=slot_key,
                    initial_guess=selected_initial_guess,
                )
            except Exception:
                if slot_key is not None:
                    self._invalidate(slot_key)
                raise
            result = result_from_linear_solve(
                system,
                linear_result,
                materialize_grad_cost=(
                    self.materialize_grad_cost
                    if materialize_grad_cost is None
                    else bool(materialize_grad_cost)
                ),
            )
            if slot_key is not None and result.converged:
                self._backward_warm_starts[slot_key] = _BackwardWarmStart(
                    system.structural_signature,
                    declared_source.detach().clone(),
                    declared_target.detach().clone(),
                    result.scaled_source_solution.detach().clone(),
                )
            elif slot_key is not None:
                self._invalidate(slot_key)
            if self.solve_observer is not None:
                try:
                    self.solve_observer(slot_key, state, result)
                except Exception:
                    if slot_key is not None:
                        self._invalidate(slot_key)
                    raise
            return result

    def reset(self, slot_key: Hashable | None = None) -> None:
        with self._lock:
            self._require_open()
            if slot_key is None:
                self._forward_warm_starts.clear()
                self._backward_warm_starts.clear()
            else:
                self._forward_warm_starts.pop(slot_key, None)
                self._backward_warm_starts.pop(slot_key, None)
            self.adjoint_backend.reset(slot_key)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._forward_warm_starts.clear()
                self._backward_warm_starts.clear()
                self._closed = True
                self.adjoint_backend.close()

    def __enter__(self) -> "EOTSolverSession":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
