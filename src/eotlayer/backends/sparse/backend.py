"""Sparse sampled-graph cuDSS preconditioning for exact Schur PCG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
import math
from threading import RLock
from time import perf_counter
from typing import Hashable, Literal, Protocol

import scipy.sparse as sp
import torch

from .. import BackendAdapterContract, LinearSolveResult
from .cudss import CuDSSFixedCSR, CuDSSPlanMemorySnapshot
from ...pcg import pcg
from .support import (
    EffectiveResistanceSupport,
    MassThresholdSupport,
    SparseSupport,
    build_effective_resistance_support,
    build_mass_threshold_support,
    build_scaled_cuda_upper_grounded_laplacian_proxy,
    build_scaled_cuda_upper_normal_matrix,
    build_scaled_grounded_laplacian_proxy,
    build_scaled_sparse_normal_matrix,
    effective_resistance_support_to,
    refresh_fixed_mass_support,
    update_scaled_cuda_upper_normal_values_,
)
from ..._core.systems import AdjointSystem


class FixedCSRFactorization(Protocol):
    matrix_format: str

    def analyze(self, matrix: object) -> None: ...

    def factorize(self, matrix: object) -> None: ...

    def solve(self, right_hand_side: torch.Tensor) -> torch.Tensor: ...

    def close(self) -> None: ...


class CuDSSFixedPatternFactorizer:
    """cuDSS adapter whose dependency is imported only during CUDA setup."""

    matrix_format = "torch_cuda_upper_csr"

    def __init__(self) -> None:
        self._solver: CuDSSFixedCSR | None = None
        self._matrix: torch.Tensor | None = None
        self._memory_snapshot: CuDSSPlanMemorySnapshot | None = None
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("factorization is closed")

    @property
    def memory_snapshot(self) -> CuDSSPlanMemorySnapshot | None:
        return self._memory_snapshot

    @staticmethod
    def _validate_matrix(matrix: object) -> torch.Tensor:
        if (
            not isinstance(matrix, torch.Tensor)
            or matrix.layout != torch.sparse_csr
            or matrix.device.type != "cuda"
        ):
            raise TypeError("cuDSS factorization requires a CUDA CSR tensor")
        return matrix

    def analyze(self, matrix: object) -> None:
        self._require_open()
        if self._solver is not None:
            raise RuntimeError("symbolic analysis already exists")
        cuda_matrix = self._validate_matrix(matrix)
        self._matrix = cuda_matrix
        self._solver = CuDSSFixedCSR(cuda_matrix)
        self._memory_snapshot = self._solver.plan()

    def factorize(self, matrix: object) -> None:
        self._require_open()
        cuda_matrix = self._validate_matrix(matrix)
        if self._solver is None or self._matrix is None:
            raise RuntimeError("symbolic analysis must precede factorization")
        if (
            cuda_matrix.shape != self._matrix.shape
            or cuda_matrix.dtype != self._matrix.dtype
            or cuda_matrix.device != self._matrix.device
            or not torch.equal(cuda_matrix.crow_indices(), self._matrix.crow_indices())
            or not torch.equal(cuda_matrix.col_indices(), self._matrix.col_indices())
        ):
            raise ValueError("numeric factorization changed the fixed CSR pattern")
        if cuda_matrix is not self._matrix:
            self._matrix.values().copy_(cuda_matrix.values())
        self._solver.factorize()

    def solve(self, right_hand_side: torch.Tensor) -> torch.Tensor:
        self._require_open()
        if self._solver is None:
            raise RuntimeError("factorization is unavailable")
        return self._solver.solve(right_hand_side)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            solver = self._solver
            self._solver = None
            self._matrix = None
            if solver is not None:
                solver.close()


@dataclass(frozen=True)
class SparseWDiagnostics:
    support_method: str
    construction_protocol: str
    support_created: bool
    support_reused: bool
    symbolic_plan_created: bool
    symbolic_plan_reused: bool
    numeric_refactorization: bool
    symbolic_plan_calls: int
    numeric_factorization_calls: int
    preconditioner_apply_calls: int
    retained_edges: int
    total_edges: int
    retained_density: float
    retained_mass_fraction: float | None
    sample_budget: int | None
    spectral_epsilon: float | None
    theorem_constant: float | None
    grounded: bool
    grounded_target_index: int | None
    active_target_dimension: int
    preconditioner_shape: tuple[int, int]
    diagonal_shift: float
    setup_seconds: float | None = None
    pcg_seconds: float | None = None
    timing_synchronized: bool = False
    exact_operator_uses_full_plan: bool = True
    resistance_solver_backend: str | None = None
    resistance_estimation_seconds: float | None = None
    resistance_energy_error_certified: bool | None = None
    resistance_energy_error_upper_bound: float | None = None
    resistance_required_energy_error: float | None = None
    cudss_memory_snapshot: CuDSSPlanMemorySnapshot | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class SparseWSolveResult(LinearSolveResult):
    diagnostics: SparseWDiagnostics


@dataclass
class _PatternEntry:
    signature: tuple[object, ...]
    support: SparseSupport
    factorization: FixedCSRFactorization
    matrix: object
    symbolic_plan_calls: int = 1
    numeric_factorization_calls: int = 1
    preconditioner_apply_calls: int = 0


class SparseWBackend:
    """Sparse graph preconditioner around the exact full-plan matvec.

    Mass support may reuse a fixed symbolic pattern.  Paper-parameterized
    effective-resistance support is plan-bound and rebuilt on every solve.
    Forward and backward warm starts are owned exclusively by
    :class:`EOTSolverSession`.
    """

    solver_name = "sparse_w"

    def __init__(
        self,
        *,
        dropped_mass: float | None = None,
        diagonal_shift: float = 0.0,
        support_method: Literal["mass", "effective_resistance"] = "mass",
        spectral_epsilon: float | None = None,
        theorem_constant: float | None = None,
        resistance_method: Literal["exact", "jl"] = "exact",
        resistance_epsilon: float | None = None,
        resistance_laplacian_solver: Literal[
            "auto", "scipy_splu", "torch_pcg"
        ] = "auto",
        resistance_edge_block_size: int = 4096,
        resistance_rhs_block_size: int = 32,
        resistance_maximum_laplacian_iterations: int | None = None,
        support_seed: int = 0,
        maximum_input_edges: int = 1_000_000,
        tolerance: float = 1e-8,
        maximum_iterations: int = 1_000,
        residual_check_interval: int = 1,
        validate_spd: bool = True,
        record_timings: bool = False,
        factorizer_factory: Callable[[], FixedCSRFactorization] = CuDSSFixedPatternFactorizer,
    ) -> None:
        if support_method not in {"mass", "effective_resistance"}:
            raise ValueError(
                "support_method must be 'mass' or 'effective_resistance'"
            )
        if support_method == "mass":
            if dropped_mass is None or not 0.0 <= float(dropped_mass) < 1.0:
                raise ValueError(
                    "mass support requires dropped_mass in [0, 1)"
                )
            if spectral_epsilon is not None or theorem_constant is not None:
                raise ValueError(
                    "paper parameters are only valid for effective-resistance support"
                )
        else:
            if dropped_mass is not None:
                raise ValueError(
                    "effective-resistance support does not use dropped_mass"
                )
            if spectral_epsilon is None or theorem_constant is None:
                raise ValueError(
                    "effective-resistance support requires spectral_epsilon and "
                    "theorem_constant"
                )
            if diagonal_shift != 0.0:
                raise ValueError(
                    "paper-aligned effective-resistance support requires "
                    "diagonal_shift=0"
                )
            if resistance_method not in {"exact", "jl"}:
                raise ValueError("resistance_method must be 'exact' or 'jl'")
            if resistance_laplacian_solver not in {
                "auto",
                "scipy_splu",
                "torch_pcg",
            }:
                raise ValueError(
                    "resistance_laplacian_solver must be 'auto', "
                    "'scipy_splu', or 'torch_pcg'"
                )
            if resistance_edge_block_size <= 0 or resistance_rhs_block_size <= 0:
                raise ValueError("resistance block sizes must be positive")
            if (
                resistance_maximum_laplacian_iterations is not None
                and resistance_maximum_laplacian_iterations <= 0
            ):
                raise ValueError(
                    "resistance_maximum_laplacian_iterations must be positive"
                )
            if maximum_input_edges <= 0:
                raise ValueError("maximum_input_edges must be positive")
        if not math.isfinite(float(diagonal_shift)) or diagonal_shift < 0:
            raise ValueError("diagonal_shift must be finite and nonnegative")
        if (
            not math.isfinite(float(tolerance))
            or tolerance <= 0
            or maximum_iterations <= 0
            or residual_check_interval <= 0
        ):
            raise ValueError("invalid PCG stopping configuration")
        self.support_method = support_method
        self.dropped_mass = (
            None if dropped_mass is None else float(dropped_mass)
        )
        self.diagonal_shift = float(diagonal_shift)
        self.spectral_epsilon = spectral_epsilon
        self.theorem_constant = theorem_constant
        self.resistance_method = resistance_method
        self.resistance_epsilon = resistance_epsilon
        self.resistance_laplacian_solver = resistance_laplacian_solver
        self.resistance_edge_block_size = int(resistance_edge_block_size)
        self.resistance_rhs_block_size = int(resistance_rhs_block_size)
        self.resistance_maximum_laplacian_iterations = (
            None
            if resistance_maximum_laplacian_iterations is None
            else int(resistance_maximum_laplacian_iterations)
        )
        self.support_seed = int(support_seed)
        self.maximum_input_edges = int(maximum_input_edges)
        self.tolerance = float(tolerance)
        self.maximum_iterations = int(maximum_iterations)
        self.residual_check_interval = int(residual_check_interval)
        self.validate_spd = bool(validate_spd)
        self.record_timings = bool(record_timings)
        self.factorizer_factory = factorizer_factory
        self._adapter_contract = BackendAdapterContract(
            solver_name=self.solver_name,
            accepted_device_types=frozenset({"cuda"}),
            accepted_dtypes=frozenset({torch.float32, torch.float64}),
            state_scope="slot",
            initial_guess_policy="accepted",
            requested_tolerance=self.tolerance,
            tolerance_floor_factor=0.0,
            maximum_iterations=self.maximum_iterations,
        )
        self._entries: dict[Hashable, _PatternEntry] = {}
        self._lock = RLock()
        self._closed = False

    @property
    def adapter_contract(self) -> BackendAdapterContract:
        return self._adapter_contract

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("backend is closed")

    @staticmethod
    def _synchronize(tensor: torch.Tensor) -> None:
        if tensor.device.type == "cuda":
            torch.cuda.synchronize(tensor.device)

    def _timed(self, system: AdjointSystem, operation):
        if not self.record_timings:
            return operation(), None
        self._synchronize(system.state.plan)
        started = perf_counter()
        try:
            value = operation()
        finally:
            self._synchronize(system.state.plan)
        return value, perf_counter() - started

    def _signature(self, system: AdjointSystem) -> tuple[object, ...]:
        return (
            *system.structural_signature,
            self.support_method,
            self.dropped_mass,
            self.diagonal_shift,
            self.spectral_epsilon,
            self.theorem_constant,
            self.resistance_method,
            self.resistance_epsilon,
            self.resistance_laplacian_solver,
            self.resistance_edge_block_size,
            self.resistance_rhs_block_size,
            self.resistance_maximum_laplacian_iterations,
            self.support_seed,
            self.maximum_input_edges,
            self.factorizer_factory,
        )

    @staticmethod
    def _format(factorizer: FixedCSRFactorization) -> str:
        return getattr(factorizer, "matrix_format", "scipy_csr")

    def _build_matrix(
        self,
        system: AdjointSystem,
        support: SparseSupport,
        factorizer: FixedCSRFactorization,
    ) -> object:
        matrix_format = self._format(factorizer)
        if isinstance(support, EffectiveResistanceSupport):
            if matrix_format == "scipy_csr":
                return build_scaled_grounded_laplacian_proxy(
                    support,
                    system.source_total,
                    system.target_total,
                    layout=system.layout,
                ).tocsr()
            if matrix_format == "torch_cuda_upper_csr":
                return build_scaled_cuda_upper_grounded_laplacian_proxy(
                    support,
                    system.source_total,
                    system.target_total,
                    layout=system.layout,
                )
            raise ValueError(
                f"unknown fixed CSR matrix format: {matrix_format!r}"
            )
        if matrix_format == "scipy_csr":
            return build_scaled_sparse_normal_matrix(
                support,
                system.source_total,
                system.target_total,
                layout=system.layout,
                diagonal_shift=self.diagonal_shift,
            ).tocsr()
        if matrix_format == "torch_cuda_upper_csr":
            return build_scaled_cuda_upper_normal_matrix(
                support,
                system.source_total,
                system.target_total,
                layout=system.layout,
                diagonal_shift=self.diagonal_shift,
            )
        raise ValueError(f"unknown fixed CSR matrix format: {matrix_format!r}")

    def _new_entry(
        self, system: AdjointSystem
    ) -> tuple[_PatternEntry, SparseSupport]:
        if self.support_method == "mass":
            assert self.dropped_mass is not None
            support: SparseSupport = build_mass_threshold_support(
                system.state.plan, dropped_mass=self.dropped_mass
            )
        else:
            assert self.spectral_epsilon is not None
            assert self.theorem_constant is not None
            construction_plan = system.state.plan.detach()
            if self.resistance_method == "exact" or (
                self.resistance_laplacian_solver == "scipy_splu"
            ):
                construction_plan = construction_plan.to("cpu")
            support = build_effective_resistance_support(
                construction_plan,
                spectral_epsilon=self.spectral_epsilon,
                theorem_constant=self.theorem_constant,
                resistance_method=self.resistance_method,
                resistance_epsilon=self.resistance_epsilon,
                laplacian_solver=self.resistance_laplacian_solver,
                seed=self.support_seed,
                edge_block_size=self.resistance_edge_block_size,
                rhs_block_size=self.resistance_rhs_block_size,
                maximum_input_edges=self.maximum_input_edges,
                maximum_laplacian_iterations=(
                    self.resistance_maximum_laplacian_iterations
                ),
            )
            if not support.connected:
                raise ValueError(
                    "the sampled effective-resistance graph is disconnected"
                )
            if (
                support.edge_weights.device != system.state.plan.device
                or support.edge_weights.dtype != system.state.plan.dtype
            ):
                support = effective_resistance_support_to(
                    support,
                    device=system.state.plan.device,
                    dtype=system.state.plan.dtype,
                )
        factorizer = self.factorizer_factory()
        try:
            matrix = self._build_matrix(system, support, factorizer)
            factorizer.analyze(matrix)
            factorizer.factorize(matrix)
        except Exception:
            try:
                factorizer.close()
            except Exception:
                pass
            raise
        pattern = (
            replace(support, edge_weights=support.edge_weights.new_empty(0))
            if isinstance(support, MassThresholdSupport)
            else support
        )
        return (
            _PatternEntry(
                signature=self._signature(system),
                support=pattern,
                factorization=factorizer,
                matrix=matrix,
            ),
            support,
        )

    def _update_entry(
        self, entry: _PatternEntry, system: AdjointSystem
    ) -> SparseSupport:
        if isinstance(entry.support, EffectiveResistanceSupport):
            raise RuntimeError(
                "effective-resistance support is bound to one plan and must be rebuilt"
            )
        support = refresh_fixed_mass_support(entry.support, system.state.plan)
        if self._format(entry.factorization) == "torch_cuda_upper_csr":
            if not isinstance(entry.matrix, torch.Tensor):
                raise TypeError("cached CUDA factorization has no CUDA CSR matrix")
            update_scaled_cuda_upper_normal_values_(
                entry.matrix,
                support,
                system.source_total,
                system.target_total,
                layout=system.layout,
                diagonal_shift=self.diagonal_shift,
            )
            matrix = entry.matrix
        else:
            matrix = self._build_matrix(system, support, entry.factorization)
        entry.factorization.factorize(matrix)
        entry.matrix = matrix
        entry.numeric_factorization_calls += 1
        return support

    def _prepare(
        self, system: AdjointSystem, key: Hashable
    ) -> tuple[_PatternEntry, SparseSupport, bool]:
        entry = self._entries.get(key)
        signature = self._signature(system)
        if entry is not None and entry.signature != signature:
            self._entries.pop(key, None)
            entry.factorization.close()
            entry = None
        if entry is None:
            entry, support = self._new_entry(system)
            self._entries[key] = entry
            return entry, support, True
        try:
            support = self._update_entry(entry, system)
        except Exception:
            self._entries.pop(key, None)
            try:
                entry.factorization.close()
            except Exception:
                pass
            raise
        return entry, support, False

    def solve(
        self,
        system: AdjointSystem,
        *,
        slot_key: Hashable | None,
        initial_guess: torch.Tensor | None = None,
    ) -> SparseWSolveResult:
        with self._lock:
            self._require_open()
            ephemeral = (
                slot_key is None
                or self.support_method == "effective_resistance"
            )
            key: Hashable = object() if ephemeral else slot_key
            prepared, setup_seconds = self._timed(
                system, lambda: self._prepare(system, key)
            )
            entry, support, created = prepared
            if initial_guess is not None and (
                initial_guess.shape != system.scaled_rhs.shape
                or initial_guess.dtype != system.scaled_rhs.dtype
                or initial_guess.device != system.scaled_rhs.device
            ):
                initial_guess = None
            source_dimension = system.source_dimension
            full_dimension = source_dimension + system.target_dimension
            retained_edges = support.retained_edges
            retained_fraction = support.retained_fraction
            retained_mass_fraction = getattr(
                support, "retained_mass_fraction", None
            )
            paper_parameters = getattr(support, "paper_parameters", None)
            construction_protocol = getattr(
                support, "construction_protocol", "mass_threshold"
            )
            sample_budget = getattr(support, "sample_budget", None)
            resistance_estimate = getattr(support, "estimate", None)
            del support

            def apply_preconditioner(residual: torch.Tensor) -> torch.Tensor:
                rhs = torch.zeros(
                    full_dimension, dtype=residual.dtype, device=residual.device
                )
                rhs[:source_dimension].copy_(residual)
                solution = entry.factorization.solve(rhs)
                if (
                    solution.ndim != 1
                    or solution.numel() != full_dimension
                    or solution.dtype != residual.dtype
                    or solution.device != residual.device
                ):
                    raise ValueError("factorizer returned an incompatible solution")
                entry.preconditioner_apply_calls += 1
                return solution[:source_dimension]

            failed = False
            try:
                result, pcg_seconds = self._timed(
                    system,
                    lambda: pcg(
                        system.matvec,
                        system.scaled_rhs,
                        apply_preconditioner=apply_preconditioner,
                        tolerance=self.tolerance,
                        maximum_iterations=self.maximum_iterations,
                        initial_guess=initial_guess,
                        residual_check_interval=self.residual_check_interval,
                        validate_spd=self.validate_spd,
                    ),
                )
                failed = not result.converged
                diagnostics = SparseWDiagnostics(
                    support_method=self.support_method,
                    construction_protocol=construction_protocol,
                    support_created=created,
                    support_reused=not created,
                    symbolic_plan_created=created,
                    symbolic_plan_reused=not created,
                    numeric_refactorization=not created,
                    symbolic_plan_calls=entry.symbolic_plan_calls,
                    numeric_factorization_calls=entry.numeric_factorization_calls,
                    preconditioner_apply_calls=entry.preconditioner_apply_calls,
                    retained_edges=retained_edges,
                    total_edges=system.state.plan.numel(),
                    retained_density=retained_fraction,
                    retained_mass_fraction=retained_mass_fraction,
                    sample_budget=sample_budget,
                    spectral_epsilon=(
                        None
                        if paper_parameters is None
                        else paper_parameters.spectral_epsilon
                    ),
                    theorem_constant=(
                        None
                        if paper_parameters is None
                        else paper_parameters.theorem_constant
                    ),
                    grounded=system.grounded,
                    grounded_target_index=system.layout.grounded_target_index,
                    active_target_dimension=system.target_dimension,
                    preconditioner_shape=(full_dimension, full_dimension),
                    diagonal_shift=self.diagonal_shift,
                    setup_seconds=setup_seconds,
                    pcg_seconds=pcg_seconds,
                    timing_synchronized=self.record_timings,
                    resistance_solver_backend=(
                        None
                        if resistance_estimate is None
                        else resistance_estimate.solver_backend
                    ),
                    resistance_estimation_seconds=(
                        None
                        if resistance_estimate is None
                        else resistance_estimate.elapsed_seconds
                    ),
                    resistance_energy_error_certified=(
                        None
                        if resistance_estimate is None
                        else resistance_estimate.energy_error_certified
                    ),
                    resistance_energy_error_upper_bound=(
                        None
                        if resistance_estimate is None
                        else resistance_estimate.energy_error_upper_bound
                    ),
                    resistance_required_energy_error=(
                        None
                        if resistance_estimate is None
                        else resistance_estimate.required_energy_relative_error
                    ),
                    cudss_memory_snapshot=getattr(
                        entry.factorization, "memory_snapshot", None
                    ),
                )
                return SparseWSolveResult(
                    solution=result.solution,
                    iterations=result.iterations,
                    initial_relative_residual=result.initial_relative_residual,
                    relative_residual=result.relative_residual,
                    converged=result.converged,
                    initial_guess_used=initial_guess is not None,
                    diagnostics=diagnostics,
                )
            except Exception:
                failed = True
                raise
            finally:
                if ephemeral or failed:
                    self._entries.pop(key, None)
                    try:
                        entry.factorization.close()
                    except Exception:
                        if not failed:
                            raise

    def reset(self, slot_key: Hashable | None = None) -> None:
        with self._lock:
            self._require_open()
            if slot_key is None:
                entries = tuple(self._entries.values())
                self._entries.clear()
            else:
                entry = self._entries.pop(slot_key, None)
                entries = () if entry is None else (entry,)
            first_error: Exception | None = None
            for entry in entries:
                try:
                    entry.factorization.close()
                except Exception as error:
                    first_error = first_error or error
            if first_error is not None:
                raise first_error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            entries = tuple(self._entries.values())
            self._entries.clear()
            self._closed = True
            first_error: Exception | None = None
            for entry in entries:
                try:
                    entry.factorization.close()
                except Exception as error:
                    first_error = first_error or error
            if first_error is not None:
                raise first_error


# Descriptive compatibility alias used by older WDA experiment code.
FixedPatternMassSparsePCGBackend = SparseWBackend
MassSparseDiagnostics = SparseWDiagnostics
MassSparseSolveResult = SparseWSolveResult
