"""Converged balanced EOT scalar value with an envelope backward."""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

from .constraints import BalancedConstraints
from .diagnostics import SolveDiagnostics
from ..forward import ForwardConfig, transport_forward
from .problem import EOTProblem


class _EnvelopeValueFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        cost: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        forward_config: ForwardConfig,
        diagnostics: SolveDiagnostics | None,
    ) -> torch.Tensor:
        problem = EOTProblem(
            cost,
            BalancedConstraints(source, target),
            float(epsilon),
        )
        state = transport_forward(problem, config=forward_config)
        if diagnostics is not None:
            diagnostics.record_forward(state)
        if not state.converged:
            raise RuntimeError(
                "envelope Sinkhorn forward did not converge within "
                f"{forward_config.maximum_iterations} iterations "
                f"(normalized marginal error={state.normalized_marginal_error:.3e})"
            )

        log_plan = (
            state.log_left[:, None]
            - state.cost / float(epsilon)
            + state.log_right[None, :]
        )
        value = (state.plan * state.cost).sum() + float(epsilon) * (
            state.plan * (log_plan - 1.0)
        ).sum()

        # The dual potentials have one additive gauge freedom. Ground the
        # final target potential while preserving every pairwise sum.
        gauge = state.log_right[-1]
        source_potential = float(epsilon) * (state.log_left + gauge)
        target_potential = float(epsilon) * (state.log_right - gauge)
        if not all(bool(torch.isfinite(item).all()) for item in (
            value, source_potential, target_potential,
        )):
            raise RuntimeError("envelope forward produced a nonfinite value or potential")
        ctx.diagnostics = diagnostics
        ctx.save_for_backward(state.plan, source_potential, target_potential)
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_value: torch.Tensor):
        # These forward snapshots contain no differentiation graph for the
        # plan or potentials, so exposing a higher derivative would be partial.
        if ctx.diagnostics is not None:
            ctx.diagnostics.begin_backward()
        try:
            plan, source_potential, target_potential = ctx.saved_tensors
            gradients = (
                grad_value * plan,
                grad_value * source_potential,
                grad_value * target_potential,
            )
            if not all(bool(torch.isfinite(value).all()) for value in gradients):
                raise RuntimeError("envelope backward produced nonfinite gradients")
            if ctx.diagnostics is not None:
                ctx.diagnostics.record_envelope_backward()
            return (*gradients, None, None, None)
        except Exception as error:
            if ctx.diagnostics is not None:
                ctx.diagnostics.fail("backward", error)
            raise


def envelope_transport_value(
    cost: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    forward_config: ForwardConfig,
    diagnostics: SolveDiagnostics | None = None,
) -> torch.Tensor:
    """Return the scalar regularized EOT optimum for one 2-D problem."""

    return _EnvelopeValueFunction.apply(
        cost,
        source,
        target,
        float(epsilon),
        forward_config,
        diagnostics,
    )
