"""Matrix-free transport-plan operators used by structured backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class PlanOperator(Protocol):
    """The small linear-operator contract needed by EOT adjoint systems."""

    @property
    def shape(self) -> torch.Size: ...

    @property
    def dtype(self) -> torch.dtype: ...

    @property
    def device(self) -> torch.device: ...

    def matvec(self, vector: torch.Tensor) -> torch.Tensor: ...

    def rmatvec(self, vector: torch.Tensor) -> torch.Tensor: ...

    def matmat(self, matrix: torch.Tensor) -> torch.Tensor: ...

    def rmatmat(self, matrix: torch.Tensor) -> torch.Tensor: ...

    def materialize(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class DensePlanOperator:
    """Wrap a materialized transport plan behind the operator contract."""

    plan: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.plan, torch.Tensor) or self.plan.ndim != 2:
            raise ValueError("plan must be a matrix")
        if not self.plan.is_floating_point():
            raise TypeError("plan must use a floating-point dtype")
        if not torch.all(torch.isfinite(self.plan)):
            raise ValueError("plan must be finite")

    @property
    def shape(self) -> torch.Size:
        return self.plan.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.plan.dtype

    @property
    def device(self) -> torch.device:
        return self.plan.device

    def _validate_vector(
        self, vector: torch.Tensor, expected: int, operation: str
    ) -> None:
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
            raise ValueError(f"{operation} operand must be a vector")
        if vector.shape[0] != expected:
            raise ValueError(f"{operation} operand must have length {expected}")
        if vector.dtype != self.dtype or vector.device != self.device:
            raise ValueError(f"{operation} operand must match plan dtype and device")

    def _validate_matrix(
        self, matrix: torch.Tensor, expected: int, operation: str
    ) -> None:
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            raise ValueError(f"{operation} operand must be a matrix")
        if matrix.shape[0] != expected:
            raise ValueError(
                f"{operation} operand must have leading dimension {expected}"
            )
        if matrix.dtype != self.dtype or matrix.device != self.device:
            raise ValueError(f"{operation} operand must match plan dtype and device")

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector, self.shape[1], "matvec")
        return self.plan @ vector

    def rmatvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector, self.shape[0], "rmatvec")
        return self.plan.transpose(0, 1) @ vector

    def matmat(self, matrix: torch.Tensor) -> torch.Tensor:
        self._validate_matrix(matrix, self.shape[1], "matmat")
        return self.plan @ matrix

    def rmatmat(self, matrix: torch.Tensor) -> torch.Tensor:
        self._validate_matrix(matrix, self.shape[0], "rmatmat")
        return self.plan.transpose(0, 1) @ matrix

    def materialize(self) -> torch.Tensor:
        return self.plan
