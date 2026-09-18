"""CPU block-IC(0) preconditioning for the exact adjoint Schur operator."""

from __future__ import annotations

import math
from typing import Hashable

import torch

from .. import BackendAdapterContract, LinearSolveResult
from ...pcg import pcg
from .support import MassThresholdSupport, build_mass_threshold_support
from ..._core.systems import AdjointSystem


class BlockIC0Preconditioner:
    """Source-block inverse action of source-first no-fill block IC(0)."""

    def __init__(
        self,
        system: AdjointSystem,
        support: MassThresholdSupport,
        *,
        diagonal_shift: float = 1e-10,
    ) -> None:
        if (
            not math.isfinite(float(diagonal_shift))
            or float(diagonal_shift) < 0.0
        ):
            raise ValueError("diagonal_shift must be finite and nonnegative")
        if support.shape != tuple(system.state.plan.shape):
            raise ValueError("support shape must match the current plan")
        for indices in (support.edge_rows, support.edge_columns):
            if (
                indices.dtype != torch.long
                or indices.device != system.state.plan.device
                or indices.ndim != 1
            ):
                raise ValueError("support indices are incompatible with the plan")
        if (
            support.edge_rows.shape != support.edge_columns.shape
            or support.edge_weights.shape != support.edge_rows.shape
            or support.edge_weights.dtype != system.state.plan.dtype
            or support.edge_weights.device != system.state.plan.device
            or not bool(torch.all(torch.isfinite(support.edge_weights)))
            or bool(torch.any(support.edge_weights < 0))
        ):
            raise ValueError("support weights are incompatible with the plan")

        self._dimension = system.source_dimension
        self._dtype = system.state.plan.dtype
        self._device = system.state.plan.device
        self._closed = False
        self._c = 1.0 + float(diagonal_shift)

        active_indices = system.active_target_indices
        active_count = int(active_indices.numel())
        compact_columns = torch.full(
            (system.target_total.numel(),),
            -1,
            dtype=torch.long,
            device=self._device,
        )
        compact_columns[active_indices] = torch.arange(
            active_count, dtype=torch.long, device=self._device
        )
        selected_columns = compact_columns[support.edge_columns]
        active_edges = selected_columns >= 0
        rows = support.edge_rows[active_edges]
        columns = selected_columns[active_edges]
        original_columns = support.edge_columns[active_edges]
        values = support.edge_weights[active_edges] / torch.sqrt(
            system.source_total[rows] * system.target_total[original_columns]
        )
        if (
            not bool(torch.all(torch.isfinite(values)))
            or bool(torch.any(values < 0))
        ):
            raise ValueError("normalized IC(0) weights must be finite and nonnegative")

        indices = torch.stack((rows, columns))
        with torch.sparse.check_sparse_tensor_invariants():
            self._normalized_support = torch.sparse_coo_tensor(
                indices,
                values,
                (self._dimension, active_count),
                dtype=self._dtype,
                device=self._device,
            ).coalesce()
        squared_columns = torch.zeros(
            active_count, dtype=self._dtype, device=self._device
        )
        if self._normalized_support._nnz():
            squared_columns.scatter_add_(
                0,
                self._normalized_support.indices()[1],
                self._normalized_support.values().square(),
            )
        self._target_pivots = self._c - squared_columns / self._c
        if not bool(
            torch.all(torch.isfinite(self._target_pivots))
            and torch.all(self._target_pivots > 0)
        ):
            raise ValueError("IC(0) produced a nonpositive or nonfinite pivot")

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("preconditioner is closed")
        if (
            residual.shape != (self._dimension,)
            or residual.dtype != self._dtype
            or residual.device != self._device
            or not bool(torch.all(torch.isfinite(residual)))
        ):
            raise ValueError("residual is incompatible with the IC(0) preconditioner")
        if not self._target_pivots.numel():
            return residual / self._c
        projected = torch.sparse.mm(
            self._normalized_support.transpose(0, 1), residual[:, None]
        ).squeeze(1)
        correction = torch.sparse.mm(
            self._normalized_support,
            (projected / self._target_pivots)[:, None],
        ).squeeze(1)
        output = residual / self._c + correction / (self._c * self._c)
        if not bool(torch.all(torch.isfinite(output))):
            raise RuntimeError("IC(0) application produced nonfinite values")
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._normalized_support = None
        self._target_pivots = None


class IC0Backend:
    """Stateless CPU PCG with a current-system block-IC(0) preconditioner."""

    solver_name = "ic0"

    def __init__(
        self,
        *,
        dropped_mass: float = 0.005,
        diagonal_shift: float = 1e-10,
        tolerance: float = 1e-8,
        maximum_iterations: int = 1_000,
        residual_check_interval: int = 1,
        validate_spd: bool = True,
    ) -> None:
        if not 0.0 <= float(dropped_mass) < 1.0:
            raise ValueError("dropped_mass must lie in [0, 1)")
        if not math.isfinite(float(diagonal_shift)) or diagonal_shift < 0.0:
            raise ValueError("diagonal_shift must be finite and nonnegative")
        if (
            not math.isfinite(float(tolerance))
            or tolerance <= 0.0
            or maximum_iterations <= 0
            or residual_check_interval <= 0
        ):
            raise ValueError("invalid IC(0) PCG stopping configuration")
        self.dropped_mass = float(dropped_mass)
        self.diagonal_shift = float(diagonal_shift)
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.residual_check_interval = int(residual_check_interval)
        self.validate_spd = bool(validate_spd)
        self._adapter_contract = BackendAdapterContract(
            solver_name=self.solver_name,
            accepted_device_types=frozenset({"cpu"}),
            accepted_dtypes=frozenset({torch.float32, torch.float64}),
            state_scope="stateless",
            initial_guess_policy="accepted",
            requested_tolerance=self.tolerance,
            tolerance_floor_factor=0.0,
            maximum_iterations=self.maximum_iterations,
        )
        self._closed = False

    @property
    def adapter_contract(self) -> BackendAdapterContract:
        return self._adapter_contract

    def solve(
        self,
        system: AdjointSystem,
        *,
        slot_key: Hashable | None,
        initial_guess: torch.Tensor | None = None,
    ) -> LinearSolveResult:
        del slot_key
        if self._closed:
            raise RuntimeError("backend is closed")
        if system.state.plan.device.type != "cpu":
            raise RuntimeError("IC(0) is currently certified only on CPU")
        support = build_mass_threshold_support(
            system.state.plan, dropped_mass=self.dropped_mass
        )
        preconditioner = BlockIC0Preconditioner(
            system, support, diagonal_shift=self.diagonal_shift
        )
        try:
            result = pcg(
                system.matvec,
                system.scaled_rhs,
                apply_preconditioner=preconditioner.apply,
                tolerance=self.tolerance,
                maximum_iterations=self.maximum_iterations,
                initial_guess=initial_guess,
                residual_check_interval=self.residual_check_interval,
                validate_spd=self.validate_spd,
            )
        finally:
            preconditioner.close()
        return LinearSolveResult(
            solution=result.solution,
            iterations=result.iterations,
            initial_relative_residual=result.initial_relative_residual,
            relative_residual=result.relative_residual,
            converged=result.converged,
            initial_guess_used=initial_guess is not None,
        )

    def reset(self, slot_key: Hashable | None = None) -> None:
        del slot_key
        if self._closed:
            raise RuntimeError("backend is closed")

    def close(self) -> None:
        self._closed = True
