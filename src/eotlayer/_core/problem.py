"""Validated physical EOT problem specification."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .constraints import TransportConstraints, constraint_vectors


@dataclass(frozen=True)
class EOTProblem:
    """One EOT problem with an explicit physical entropic coefficient.

    Normalized regularization is a case-level protocol. The core receives the
    resolved physical ``epsilon`` and never silently rescales ``cost``.
    """

    cost: torch.Tensor
    constraints: TransportConstraints
    epsilon: float
    validate: bool = True

    def __post_init__(self) -> None:
        if not self.validate:
            return
        if not isinstance(self.cost, torch.Tensor) or self.cost.ndim != 2:
            raise ValueError("cost must be a matrix")
        if not self.cost.is_floating_point():
            raise TypeError("cost must use a floating-point dtype")
        if not torch.all(torch.isfinite(self.cost)):
            raise ValueError("cost must be finite")
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0:
            raise ValueError("epsilon must be finite and positive")
        source, target = constraint_vectors(self.constraints)
        if self.cost.shape != (source.numel(), target.numel()):
            raise ValueError("cost shape does not match the constraint dimensions")
        if self.cost.dtype != source.dtype or self.cost.device != source.device:
            raise ValueError("cost and constraints must use the same dtype and device")
