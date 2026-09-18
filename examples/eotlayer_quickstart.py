#!/usr/bin/env python3
"""Minimal EOTLayer quickstart with explicit, inspectable tensors."""

from __future__ import annotations

import torch

from eotlayer import EOTLayer


DTYPE = torch.float64
EPSILON = 0.3


def explicit_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one small balanced EOT problem whose entries are visible."""

    cost = torch.tensor(
        (
            (0.10, 0.80, 0.55),
            (0.65, 0.20, 0.45),
            (0.70, 0.50, 0.15),
        ),
        dtype=DTYPE,
    )
    source = torch.tensor((0.20, 0.30, 0.50), dtype=DTYPE)
    target = torch.tensor((0.40, 0.35, 0.25), dtype=DTYPE)
    return cost, source, target


def differentiable_copy(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(value.clone().requires_grad_(True) for value in inputs)


def main() -> None:
    inputs = explicit_problem()
    weight = torch.tensor(
        (
            (0.2, -0.4, 0.7),
            (0.5, 0.1, -0.3),
            (-0.6, 0.8, 0.4),
        ),
        dtype=DTYPE,
    )

    unrolled_inputs = differentiable_copy(inputs)
    unrolled_plan = EOTLayer(
        EPSILON,
        mode="unrolled",
        iter_num=8,
    )(*unrolled_inputs)
    torch.sum(unrolled_plan * weight).backward()

    implicit_inputs = differentiable_copy(inputs)
    implicit_layer = EOTLayer(
        EPSILON,
        mode="implicit",
        forward_tol=1e-12,
        forward_iter_max=20_000,
        backward_solver="dense",
        backward_tol=1e-10,
    )
    implicit_plan = implicit_layer(*implicit_inputs)
    torch.sum(implicit_plan * weight).backward()
    implicit_diagnostics = implicit_layer.diagnostics[0]
    assert implicit_diagnostics["forward"]["status"] == "converged"
    assert implicit_diagnostics["backward"]["status"] == "converged"
    implicit_layer.close()

    envelope_inputs = differentiable_copy(inputs)
    envelope_value = EOTLayer(
        EPSILON,
        mode="envelope",
        forward_tol=1e-12,
        forward_iter_max=20_000,
    )(*envelope_inputs)
    envelope_value.backward()

    cost, source, target = inputs
    detached_plan = implicit_plan.detach()
    row_error = float((detached_plan.sum(dim=1) - source).abs().max())
    target_error = float((detached_plan.sum(dim=0) - target).abs().max())
    print("EXAMPLE INPUT")
    print(f"cost_shape={tuple(cost.shape)}")
    print(f"source={tuple(float(value) for value in source)}")
    print(f"target={tuple(float(value) for value in target)}")
    print("EXAMPLE OUTPUT")
    print(f"unrolled_plan_shape={tuple(unrolled_plan.shape)}")
    print(f"implicit_plan_shape={tuple(implicit_plan.shape)}")
    print(f"envelope_value_shape={tuple(envelope_value.shape)}")
    print(f"implicit_row_marginal_error={row_error:.3e}")
    print(f"implicit_target_marginal_error={target_error:.3e}")
    print(f"implicit_normalized_marginal_error={implicit_diagnostics['forward']['normalized_marginal_error']:.3e}")
    print(f"implicit_backward_relative_residual={implicit_diagnostics['backward']['relative_residual']:.3e}")
    print(f"implicit_backward_effective_tolerance={implicit_diagnostics['backward']['effective_tolerance']:.3e}")
    print(f"unrolled_cost_grad_norm={float(unrolled_inputs[0].grad.norm()):.6e}")
    print(f"implicit_cost_grad_norm={float(implicit_inputs[0].grad.norm()):.6e}")
    print(f"envelope_cost_grad_norm={float(envelope_inputs[0].grad.norm()):.6e}")


if __name__ == "__main__":
    main()
