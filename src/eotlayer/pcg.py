"""Preconditioned conjugate gradient with exact residual confirmation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable

import torch


@dataclass(frozen=True)
class PCGResult:
    solution: torch.Tensor
    iterations: int
    initial_relative_residual: float
    relative_residual: float
    converged: bool


def pcg(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    *,
    apply_preconditioner: Callable[[torch.Tensor], torch.Tensor] | None = None,
    tolerance: float = 1e-8,
    maximum_iterations: int = 1_000,
    initial_guess: torch.Tensor | None = None,
    residual_check_interval: int = 1,
    validate_spd: bool = True,
) -> PCGResult:
    if right_hand_side.ndim != 1 or not right_hand_side.is_floating_point():
        raise ValueError("right_hand_side must be a floating-point vector")
    if not torch.all(torch.isfinite(right_hand_side)):
        raise ValueError("right_hand_side must be finite")
    if (
        not math.isfinite(float(tolerance))
        or tolerance <= 0
        or maximum_iterations <= 0
        or residual_check_interval <= 0
    ):
        raise ValueError("invalid PCG stopping configuration")
    if initial_guess is not None and (
        initial_guess.shape != right_hand_side.shape
        or initial_guess.dtype != right_hand_side.dtype
        or initial_guess.device != right_hand_side.device
        or not torch.all(torch.isfinite(initial_guess))
    ):
        raise ValueError("initial_guess is incompatible with right_hand_side")

    apply = apply_preconditioner or (lambda vector: vector)
    solution = torch.zeros_like(right_hand_side) if initial_guess is None else initial_guess.clone()
    residual = right_hand_side - matvec(solution)
    rhs_norm = float(torch.linalg.vector_norm(right_hand_side))
    normalizer = max(rhs_norm, torch.finfo(right_hand_side.dtype).tiny)
    relative = float(torch.linalg.vector_norm(residual)) / normalizer
    initial_relative = relative
    if relative <= tolerance:
        return PCGResult(solution, 0, initial_relative, relative, True)

    preconditioned = apply(residual)
    residual_product = torch.dot(residual, preconditioned)
    if validate_spd and not float(residual_product) > 0.0:
        raise RuntimeError("preconditioner is not positive definite")
    direction = preconditioned.clone()

    for iteration in range(1, maximum_iterations + 1):
        image = matvec(direction)
        curvature = torch.dot(direction, image)
        if validate_spd and not float(curvature) > 0.0:
            raise RuntimeError("operator is not positive definite")
        step = residual_product / curvature
        solution = solution + step * direction
        residual = residual - step * image
        if (
            iteration % residual_check_interval == 0
            or iteration == maximum_iterations
            # Exact recursive convergence can precede a scheduled check.
            # Confirm it before a zero search direction looks like an SPD
            # breakdown; residual drift still goes through the restart below.
            or not bool(torch.count_nonzero(residual))
        ):
            relative = float(torch.linalg.vector_norm(residual)) / normalizer
            if relative <= tolerance:
                exact_residual = right_hand_side - matvec(solution)
                relative = float(torch.linalg.vector_norm(exact_residual)) / normalizer
                if relative <= tolerance:
                    return PCGResult(
                        solution, iteration, initial_relative, relative, True
                    )
                # Restart after recursive residual drift.
                residual = exact_residual
                preconditioned = apply(residual)
                residual_product = torch.dot(residual, preconditioned)
                if validate_spd and not float(residual_product) > 0.0:
                    raise RuntimeError("preconditioner is not positive definite")
                direction = preconditioned.clone()
                continue
        next_preconditioned = apply(residual)
        next_product = torch.dot(residual, next_preconditioned)
        if validate_spd and not float(next_product) > 0.0:
            raise RuntimeError("preconditioner lost positive definiteness")
        coefficient = next_product / residual_product
        direction = next_preconditioned + coefficient * direction
        residual_product = next_product

    exact_residual = right_hand_side - matvec(solution)
    relative = float(torch.linalg.vector_norm(exact_residual)) / normalizer
    return PCGResult(
        solution, maximum_iterations, initial_relative, relative, relative <= tolerance
    )
