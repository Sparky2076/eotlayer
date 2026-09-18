"""Progressive, forward-screened selection for implicit EOT adjoints.

The selector is deliberately conservative.  It never computes a Dense
reference answer, never runs a second Sinkhorn forward, and never constructs a
sparse support merely to inspect it.  A short Jacobi-W run is either the final
solve or the initial guess for the selected continuation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from threading import RLock
from typing import Callable, Hashable

import torch

from . import (
    AdjointBackend,
    BackendAdapterContract,
    DenseBackend,
    LinearSolveResult,
)
from ..forward import PlanScreeningFeatures, collect_plan_screening_features
from ..pcg import PCGResult, pcg
from .sparse import SparseWBackend
from .._core.systems import AdjointSystem


@dataclass(frozen=True)
class AutoBackendPolicy:
    """Fixed first-version policy; timings still require empirical validation."""

    probe_iterations: int = 5
    allow_dense: bool = True
    dense_cpu_memory_budget_bytes: int = 256 * 1024 * 1024
    dense_cuda_free_memory_fraction: float = 0.25
    dense_work_margin: float = 1.0
    dense_maximum_zero_fraction: float = 0.5
    allow_sparse: bool = True
    sparse_requires_cuda: bool = True
    sparse_minimum_reuse_count: int = 2
    sparse_maximum_effective_density: float = 0.2
    sparse_minimum_predicted_jacobi_iterations: int = 32
    dropped_mass: float = 0.005
    diagonal_shift: float = 1e-10

    def __post_init__(self) -> None:
        if self.probe_iterations <= 0 or self.sparse_minimum_reuse_count <= 0:
            raise ValueError("auto iteration and reuse controls must be positive")
        if self.dense_cpu_memory_budget_bytes <= 0:
            raise ValueError("auto Dense CPU memory budget must be positive")
        if not 0.0 < self.dense_cuda_free_memory_fraction <= 1.0:
            raise ValueError("auto Dense CUDA memory fraction must lie in (0, 1]")
        if not math.isfinite(self.dense_work_margin) or self.dense_work_margin <= 0:
            raise ValueError("auto Dense work margin must be finite and positive")
        for value, name in (
            (self.dense_maximum_zero_fraction, "dense_maximum_zero_fraction"),
            (
                self.sparse_maximum_effective_density,
                "sparse_maximum_effective_density",
            ),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.sparse_minimum_predicted_jacobi_iterations <= 0:
            raise ValueError("auto Sparse iteration threshold must be positive")
        if not 0.0 <= self.dropped_mass < 1.0:
            raise ValueError("auto dropped_mass must lie in [0, 1)")
        if not math.isfinite(self.diagonal_shift) or self.diagonal_shift < 0.0:
            raise ValueError("auto diagonal_shift must be finite and nonnegative")


@dataclass(frozen=True)
class AutoBackwardDiagnostics:
    selected_method: str
    reason: str
    observed_reuse_count: int
    effective_support_density: float
    exact_zero_fraction: float
    estimated_dense_peak_bytes: int
    dense_memory_budget_bytes: int
    dense_eligible: bool
    sparse_eligible: bool
    probe_iterations: int
    probe_initial_relative_residual: float
    probe_relative_residual: float
    predicted_jacobi_iterations: int
    probe_solution_reused: bool
    full_operator_relative_residual: float
    selector_uses_dense_reference: bool = False
    selector_reruns_forward: bool = False
    exact_operator_uses_full_plan: bool = True


@dataclass(frozen=True)
class AutoLinearSolveResult(LinearSolveResult):
    diagnostics: AutoBackwardDiagnostics


def _default_sparse_backend_available() -> bool:
    try:
        return importlib.util.find_spec("nvmath.sparse.advanced") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _dense_memory_budget(system: AdjointSystem, policy: AutoBackendPolicy) -> int:
    if system.state.plan.device.type != "cuda":
        return int(policy.dense_cpu_memory_budget_bytes)
    try:
        free_bytes, _ = torch.cuda.mem_get_info(system.state.plan.device)
    except Exception:
        return 0
    return int(float(free_bytes) * policy.dense_cuda_free_memory_fraction)


def _dense_wins_before_probe(
    system: AdjointSystem,
    policy: AutoBackendPolicy,
) -> bool:
    source = system.source_dimension
    target = system.target_dimension
    dense_work = 2.0 * source * source * target + (2.0 / 3.0) * source**3
    probe_work = 2.0 * policy.probe_iterations * source * target
    return dense_work <= policy.dense_work_margin * probe_work


def _predict_jacobi_iterations(
    probe: PCGResult,
    *,
    tolerance: float,
    maximum_iterations: int,
) -> int:
    if probe.converged:
        return probe.iterations
    if (
        probe.iterations <= 0
        or probe.initial_relative_residual <= 0.0
        or probe.relative_residual <= 0.0
    ):
        return maximum_iterations
    ratio = probe.relative_residual / probe.initial_relative_residual
    if not math.isfinite(ratio) or ratio >= 1.0:
        return maximum_iterations
    contraction = ratio ** (1.0 / probe.iterations)
    if not 0.0 < contraction < 1.0:
        return maximum_iterations
    remaining = math.ceil(
        math.log(tolerance / probe.relative_residual) / math.log(contraction)
    )
    return min(maximum_iterations, probe.iterations + max(0, remaining))


class ProgressiveAutoBackend:
    """Select Dense, Jacobi-W, or Sparse-W without an oracle solve."""

    solver_name = "auto"

    def __init__(
        self,
        *,
        tolerance: float = 1e-8,
        maximum_iterations: int = 1_000,
        residual_check_interval: int = 1,
        validate_spd: bool = True,
        policy: AutoBackendPolicy | None = None,
        sparse_backend: AdjointBackend | None = None,
        sparse_backend_available: Callable[[], bool] | None = None,
    ) -> None:
        if (
            not math.isfinite(float(tolerance))
            or tolerance <= 0.0
            or maximum_iterations <= 0
            or residual_check_interval <= 0
        ):
            raise ValueError("invalid auto backward stopping configuration")
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.residual_check_interval = int(residual_check_interval)
        self.validate_spd = bool(validate_spd)
        self.policy = policy or AutoBackendPolicy()
        self._adapter_contract = BackendAdapterContract(
            solver_name=self.solver_name,
            accepted_device_types=None,
            accepted_dtypes=frozenset({torch.float32, torch.float64}),
            state_scope="slot",
            initial_guess_policy="accepted",
            requested_tolerance=self.tolerance,
            tolerance_floor_factor=0.0,
            maximum_iterations=self.maximum_iterations,
            fallback_solvers=("dense", "jacobi_w", "sparse_w"),
        )
        self._dense = DenseBackend(tolerance=self.tolerance)
        self._sparse = sparse_backend
        self._sparse_available = (
            sparse_backend_available or _default_sparse_backend_available
        )
        self._reuse_signatures: dict[Hashable, tuple[object, ...]] = {}
        self._reuse_counts: dict[Hashable, int] = {}
        self._diagnostics: dict[Hashable, AutoBackwardDiagnostics] = {}
        self._closed = False
        self._lock = RLock()

    @property
    def adapter_contract(self) -> BackendAdapterContract:
        return self._adapter_contract

    @property
    def diagnostics(self) -> dict[Hashable, AutoBackwardDiagnostics]:
        with self._lock:
            return dict(self._diagnostics)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    def _features(self, system: AdjointSystem) -> PlanScreeningFeatures:
        features = system.state.screening_features
        if features is not None:
            return features
        return collect_plan_screening_features(
            system.state.plan,
            iterations=system.state.iterations,
            residual_contraction=None,
            block_rows=128,
        )

    def _observe_reuse(
        self, system: AdjointSystem, slot_key: Hashable | None
    ) -> int:
        if slot_key is None:
            return 1
        signature = system.structural_signature
        previous = self._reuse_signatures.get(slot_key)
        if previous != signature:
            if previous is not None and self._sparse is not None:
                self._sparse.reset(slot_key)
            self._reuse_signatures[slot_key] = signature
            self._reuse_counts[slot_key] = 1
        else:
            self._reuse_counts[slot_key] += 1
        return self._reuse_counts[slot_key]

    def _sparse_backend(self) -> AdjointBackend:
        if self._sparse is None:
            self._sparse = SparseWBackend(
                dropped_mass=self.policy.dropped_mass,
                diagonal_shift=self.policy.diagonal_shift,
                tolerance=self.tolerance,
                maximum_iterations=max(
                    1, self.maximum_iterations - self.policy.probe_iterations
                ),
                residual_check_interval=self.residual_check_interval,
                validate_spd=self.validate_spd,
            )
        return self._sparse

    @staticmethod
    def _combine(
        probe: PCGResult,
        continuation: LinearSolveResult | PCGResult,
        *,
        external_initial_guess_used: bool,
        diagnostics: AutoBackwardDiagnostics,
    ) -> AutoLinearSolveResult:
        return AutoLinearSolveResult(
            solution=continuation.solution,
            iterations=probe.iterations + continuation.iterations,
            initial_relative_residual=probe.initial_relative_residual,
            relative_residual=continuation.relative_residual,
            converged=continuation.converged,
            initial_guess_used=external_initial_guess_used,
            diagnostics=diagnostics,
        )

    def _record(
        self,
        slot_key: Hashable | None,
        diagnostics: AutoBackwardDiagnostics,
    ) -> None:
        if slot_key is not None:
            self._diagnostics[slot_key] = diagnostics

    def solve(
        self,
        system: AdjointSystem,
        *,
        slot_key: Hashable | None,
        initial_guess: torch.Tensor | None = None,
    ) -> AutoLinearSolveResult:
        with self._lock:
            self._require_open()
            features = self._features(system)
            reuse_count = self._observe_reuse(system, slot_key)
            memory_budget = _dense_memory_budget(system, self.policy)
            dense_eligible = (
                self.policy.allow_dense
                and features.estimated_dense_peak_bytes <= memory_budget
                and features.exact_zero_fraction
                <= self.policy.dense_maximum_zero_fraction
            )
            if dense_eligible and _dense_wins_before_probe(system, self.policy):
                dense = self._dense.solve(
                    system, slot_key=slot_key, initial_guess=initial_guess
                )
                diagnostics = AutoBackwardDiagnostics(
                    selected_method="dense",
                    reason="dense_static_memory_and_work_gate",
                    observed_reuse_count=reuse_count,
                    effective_support_density=features.effective_support_density,
                    exact_zero_fraction=features.exact_zero_fraction,
                    estimated_dense_peak_bytes=features.estimated_dense_peak_bytes,
                    dense_memory_budget_bytes=memory_budget,
                    dense_eligible=True,
                    sparse_eligible=False,
                    probe_iterations=0,
                    probe_initial_relative_residual=0.0,
                    probe_relative_residual=0.0,
                    predicted_jacobi_iterations=0,
                    probe_solution_reused=False,
                    full_operator_relative_residual=dense.relative_residual,
                )
                self._record(slot_key, diagnostics)
                # ``auto`` owns one requested-tolerance contract. Dense may
                # internally accept a machine-precision floor, but the outer
                # adapter must not silently inherit that looser threshold.
                converged = dense.relative_residual <= self.tolerance
                return AutoLinearSolveResult(
                    solution=dense.solution,
                    iterations=dense.iterations,
                    initial_relative_residual=dense.initial_relative_residual,
                    relative_residual=dense.relative_residual,
                    converged=converged,
                    initial_guess_used=dense.initial_guess_used,
                    diagnostics=diagnostics,
                )

            probe_limit = min(
                self.policy.probe_iterations, self.maximum_iterations
            )
            probe = pcg(
                system.matvec,
                system.scaled_rhs,
                tolerance=self.tolerance,
                maximum_iterations=probe_limit,
                initial_guess=initial_guess,
                residual_check_interval=self.residual_check_interval,
                validate_spd=self.validate_spd,
            )
            predicted = _predict_jacobi_iterations(
                probe,
                tolerance=self.tolerance,
                maximum_iterations=self.maximum_iterations,
            )
            sparse_base_eligible = (
                self.policy.allow_sparse
                and slot_key is not None
                and reuse_count >= self.policy.sparse_minimum_reuse_count
                and features.effective_support_density
                <= self.policy.sparse_maximum_effective_density
                and predicted
                >= self.policy.sparse_minimum_predicted_jacobi_iterations
                and (
                    not self.policy.sparse_requires_cuda
                    or system.state.plan.device.type == "cuda"
                )
            )
            sparse_eligible = sparse_base_eligible and self._sparse_available()

            if probe.converged:
                diagnostics = AutoBackwardDiagnostics(
                    selected_method="jacobi_w",
                    reason="jacobi_probe_reached_tolerance",
                    observed_reuse_count=reuse_count,
                    effective_support_density=features.effective_support_density,
                    exact_zero_fraction=features.exact_zero_fraction,
                    estimated_dense_peak_bytes=features.estimated_dense_peak_bytes,
                    dense_memory_budget_bytes=memory_budget,
                    dense_eligible=dense_eligible,
                    sparse_eligible=sparse_eligible,
                    probe_iterations=probe.iterations,
                    probe_initial_relative_residual=probe.initial_relative_residual,
                    probe_relative_residual=probe.relative_residual,
                    predicted_jacobi_iterations=predicted,
                    probe_solution_reused=True,
                    full_operator_relative_residual=probe.relative_residual,
                )
                self._record(slot_key, diagnostics)
                return AutoLinearSolveResult(
                    solution=probe.solution,
                    iterations=probe.iterations,
                    initial_relative_residual=probe.initial_relative_residual,
                    relative_residual=probe.relative_residual,
                    converged=True,
                    initial_guess_used=initial_guess is not None,
                    diagnostics=diagnostics,
                )

            if sparse_eligible:
                continuation = self._sparse_backend().solve(
                    system,
                    slot_key=slot_key,
                    initial_guess=probe.solution,
                )
                selected_method = "sparse_w"
                reason = "observed_reuse_sparse_plan_and_slow_jacobi"
            else:
                remaining = self.maximum_iterations - probe.iterations
                if remaining <= 0:
                    continuation = PCGResult(
                        solution=probe.solution,
                        iterations=0,
                        initial_relative_residual=probe.relative_residual,
                        relative_residual=probe.relative_residual,
                        converged=False,
                    )
                else:
                    continuation = pcg(
                        system.matvec,
                        system.scaled_rhs,
                        tolerance=self.tolerance,
                        maximum_iterations=remaining,
                        initial_guess=probe.solution,
                        residual_check_interval=self.residual_check_interval,
                        validate_spd=self.validate_spd,
                    )
                selected_method = "jacobi_w"
                reason = "sparse_ineligible_continue_jacobi"

            diagnostics = AutoBackwardDiagnostics(
                selected_method=selected_method,
                reason=reason,
                observed_reuse_count=reuse_count,
                effective_support_density=features.effective_support_density,
                exact_zero_fraction=features.exact_zero_fraction,
                estimated_dense_peak_bytes=features.estimated_dense_peak_bytes,
                dense_memory_budget_bytes=memory_budget,
                dense_eligible=dense_eligible,
                sparse_eligible=sparse_eligible,
                probe_iterations=probe.iterations,
                probe_initial_relative_residual=probe.initial_relative_residual,
                probe_relative_residual=probe.relative_residual,
                predicted_jacobi_iterations=predicted,
                probe_solution_reused=True,
                full_operator_relative_residual=continuation.relative_residual,
            )
            self._record(slot_key, diagnostics)
            return self._combine(
                probe,
                continuation,
                external_initial_guess_used=initial_guess is not None,
                diagnostics=diagnostics,
            )

    def reset(self, slot_key: Hashable | None = None) -> None:
        with self._lock:
            self._require_open()
            if slot_key is None:
                self._reuse_signatures.clear()
                self._reuse_counts.clear()
                self._diagnostics.clear()
            else:
                self._reuse_signatures.pop(slot_key, None)
                self._reuse_counts.pop(slot_key, None)
                self._diagnostics.pop(slot_key, None)
            self._dense.reset(slot_key)
            if self._sparse is not None:
                self._sparse.reset(slot_key)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._reuse_signatures.clear()
            self._reuse_counts.clear()
            self._diagnostics.clear()
            self._dense.close()
            if self._sparse is not None:
                self._sparse.close()
