"""Gauge layout for exact EOT adjoint Schur systems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchurLayout:
    """Active target duals after optional balanced grounding."""

    target_dimension: int
    active_target_indices: tuple[int, ...]
    grounded_target_index: int | None

    def __post_init__(self) -> None:
        if self.target_dimension <= 0:
            raise ValueError("target_dimension must be positive")
        active = tuple(int(index) for index in self.active_target_indices)
        object.__setattr__(self, "active_target_indices", active)
        if active != tuple(sorted(set(active))):
            raise ValueError("active target indices must be sorted and unique")
        if any(index < 0 or index >= self.target_dimension for index in active):
            raise ValueError("active target index is out of range")
        all_indices = set(range(self.target_dimension))
        if self.grounded_target_index is None:
            if set(active) != all_indices:
                raise ValueError("ungrounded layouts must activate every target")
        else:
            grounded = int(self.grounded_target_index)
            if grounded not in all_indices:
                raise ValueError("grounded target index is out of range")
            all_indices.remove(grounded)
            if set(active) != all_indices:
                raise ValueError("grounded layouts activate every other target")

    @classmethod
    def balanced(
        cls, *, target_dimension: int, grounded_target_index: int | None = None
    ) -> "SchurLayout":
        grounded = target_dimension - 1 if grounded_target_index is None else int(grounded_target_index)
        return cls(
            target_dimension,
            tuple(index for index in range(target_dimension) if index != grounded),
            grounded,
        )

    @classmethod
    def slack(cls, *, target_dimension: int) -> "SchurLayout":
        return cls(target_dimension, tuple(range(target_dimension)), None)
