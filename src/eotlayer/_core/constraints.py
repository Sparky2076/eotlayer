"""Affine constraint families supported by the reusable EOT core."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _validate_positive_vector(values: torch.Tensor, name: str) -> None:
    if not isinstance(values, torch.Tensor) or values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if not values.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if values.numel() == 0 or not torch.all(torch.isfinite(values)):
        raise ValueError(f"{name} must be finite and nonempty")
    if torch.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True)
class BalancedConstraints:
    """Prescribed source and target marginals with equal total mass."""

    source: torch.Tensor
    target: torch.Tensor

    def __post_init__(self) -> None:
        _validate_positive_vector(self.source, "source")
        _validate_positive_vector(self.target, "target")
        if self.source.dtype != self.target.dtype:
            raise TypeError("balanced marginals must use the same dtype")
        if self.source.device != self.target.device:
            raise ValueError("balanced marginals must use the same device")
        source_mass = float(self.source.detach().sum())
        target_mass = float(self.target.detach().sum())
        if not math.isfinite(source_mass) or not math.isfinite(target_mass):
            raise ValueError("balanced marginal total mass must be finite")
        tolerance = 100.0 * torch.finfo(self.source.dtype).eps * max(
            source_mass, target_mass
        )
        if abs(source_mass - target_mass) > tolerance:
            raise ValueError("balanced marginals must have equal total mass")


@dataclass(frozen=True)
class SlackMatchingConstraints:
    """RPM-style real totals with one unconstrained slack node on each side."""

    source_total: torch.Tensor
    target_total: torch.Tensor
    slack_logit: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive_vector(self.source_total, "source_total")
        _validate_positive_vector(self.target_total, "target_total")
        if self.source_total.dtype != self.target_total.dtype:
            raise TypeError("slack totals must use the same dtype")
        if self.source_total.device != self.target_total.device:
            raise ValueError("slack totals must use the same device")
        if not math.isfinite(float(self.slack_logit)):
            raise ValueError("slack_logit must be finite")


TransportConstraints = BalancedConstraints | SlackMatchingConstraints


def constraint_vectors(
    constraints: TransportConstraints,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(constraints, BalancedConstraints):
        return constraints.source, constraints.target
    if isinstance(constraints, SlackMatchingConstraints):
        return constraints.source_total, constraints.target_total
    raise TypeError(f"unsupported constraint type: {type(constraints)!r}")
