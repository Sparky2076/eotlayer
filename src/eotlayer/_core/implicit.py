"""Converged balanced EOT plan with a custom implicit backward."""

from __future__ import annotations

from typing import Hashable

import torch
from torch.autograd.function import once_differentiable

from .constraints import BalancedConstraints
from .diagnostics import SolveDiagnostics
from ..forward import ForwardConfig
from .problem import EOTProblem
from .session import EOTSolverSession


class _ImplicitPlanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cost: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        forward_config: ForwardConfig,
        session: EOTSolverSession,
        slot_key: Hashable | None,
        diagnostics: SolveDiagnostics | None,
    ) -> torch.Tensor:
        problem = EOTProblem(
            cost,
            BalancedConstraints(source, target),
            float(epsilon),
        )
        state = session.forward(
            problem, config=forward_config, slot_key=slot_key
        )
        if diagnostics is not None:
            diagnostics.record_forward(state)
        if not state.converged:
            raise RuntimeError(
                "implicit Sinkhorn forward did not converge within "
                f"{forward_config.maximum_iterations} iterations "
                f"(normalized marginal error={state.normalized_marginal_error:.3e})"
            )
        ctx.transport_state = state
        ctx.session = session
        ctx.slot_key = slot_key
        ctx.diagnostics = diagnostics
        # The detached problem shares input storage and the returned plan is
        # also the plan used by backward. Register every aliased tensor with
        # autograd so a later in-place update cannot silently alter the VJP.
        ctx.save_for_backward(cost, source, target, state.plan)
        return state.plan

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_plan: torch.Tensor):
        # Accessing saved_tensors checks their version counters before the
        # state is consumed. The implicit VJP supports first derivatives only.
        if ctx.diagnostics is not None:
            ctx.diagnostics.begin_backward()
        try:
            _ = ctx.saved_tensors
            result = ctx.session.backward(
                ctx.transport_state, grad_plan, slot_key=ctx.slot_key
            )
            if ctx.diagnostics is not None:
                ctx.diagnostics.record_backward(result)
            if not result.converged:
                raise RuntimeError("implicit adjoint solve did not converge")
            if result.grad_cost is None:
                raise RuntimeError("autograd requires a materialized cost gradient")
            if not all(bool(torch.isfinite(value).all()) for value in (
                result.grad_cost, result.grad_source, result.grad_target,
            )):
                raise RuntimeError("implicit backward produced nonfinite gradients")
        except Exception as error:
            if ctx.diagnostics is not None:
                ctx.diagnostics.fail("backward", error)
            raise
        return (
            result.grad_cost,
            result.grad_source,
            result.grad_target,
            None,
            None,
            None,
            None,
            None,
        )


def implicit_transport_plan(
    cost: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    forward_config: ForwardConfig,
    session: EOTSolverSession,
    slot_key: Hashable | None = None,
    diagnostics: SolveDiagnostics | None = None,
) -> torch.Tensor:
    """Solve one 2-D balanced problem and attach its implicit VJP."""

    return _ImplicitPlanFunction.apply(
        cost,
        source,
        target,
        float(epsilon),
        forward_config,
        session,
        slot_key,
        diagnostics,
    )
