"""Canonical exact-adjoint linear-solver backends."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Hashable, Literal, Protocol

import torch

from ..pcg import pcg
from .._core.systems import AdjointSystem


CANONICAL_METHODS = ("dense", "jacobi_w", "ic0", "sparse_w")
METHOD_REGISTRY = {name: name for name in CANONICAL_METHODS}

_RESIDUAL_OPERATOR = "full_plan_scaled_source_schur"


class BackendContractError(RuntimeError):
    """A backend or its result violates the common adjoint contract."""


@dataclass(frozen=True)
class BackendAdapterContract:
    """Frozen capabilities and numerical promises for one adjoint backend."""

    solver_name: str
    accepted_device_types: frozenset[str] | None
    accepted_dtypes: frozenset[torch.dtype]
    state_scope: Literal["stateless", "slot"]
    initial_guess_policy: Literal["accepted", "ignored"]
    requested_tolerance: float
    tolerance_floor_factor: float
    maximum_iterations: int | None
    fallback_solvers: tuple[str, ...] = ()
    residual_operator: str = _RESIDUAL_OPERATOR

    def __post_init__(self) -> None:
        if not isinstance(self.solver_name, str) or not self.solver_name:
            raise ValueError("solver_name must be a nonempty string")
        if self.accepted_device_types is not None and (
            not isinstance(self.accepted_device_types, frozenset)
            or not self.accepted_device_types
            or any(
                not isinstance(device_type, str) or not device_type
                for device_type in self.accepted_device_types
            )
        ):
            raise ValueError("accepted_device_types must be None or a nonempty frozenset")
        if (
            not isinstance(self.accepted_dtypes, frozenset)
            or not self.accepted_dtypes
            or not self.accepted_dtypes <= frozenset({torch.float32, torch.float64})
        ):
            raise ValueError("accepted_dtypes must contain supported floating dtypes")
        if self.state_scope not in {"stateless", "slot"}:
            raise ValueError("state_scope must be stateless or slot")
        if self.initial_guess_policy not in {"accepted", "ignored"}:
            raise ValueError("initial_guess_policy must be accepted or ignored")
        _validate_tolerance(self.requested_tolerance)
        if (
            isinstance(self.tolerance_floor_factor, bool)
            or not isinstance(self.tolerance_floor_factor, Real)
            or not math.isfinite(float(self.tolerance_floor_factor))
            or self.tolerance_floor_factor < 0
        ):
            raise ValueError("tolerance_floor_factor must be finite and nonnegative")
        if self.maximum_iterations is not None and (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, Integral)
            or self.maximum_iterations <= 0
        ):
            raise ValueError("maximum_iterations must be None or a positive integer")
        if (
            not isinstance(self.fallback_solvers, tuple)
            or len(set(self.fallback_solvers)) != len(self.fallback_solvers)
            or any(not isinstance(name, str) or not name for name in self.fallback_solvers)
        ):
            raise ValueError("fallback_solvers must contain unique nonempty names")
        if self.solver_name in self.fallback_solvers:
            raise ValueError("fallback_solvers must not repeat solver_name")
        if self.residual_operator != _RESIDUAL_OPERATOR:
            raise ValueError(
                f"residual_operator must be {_RESIDUAL_OPERATOR!r}"
            )

    def effective_tolerance(self, dtype: torch.dtype) -> float:
        if dtype not in self.accepted_dtypes:
            raise BackendContractError(
                f"backend {self.solver_name!r} does not accept dtype {dtype}"
            )
        return max(
            float(self.requested_tolerance),
            float(self.tolerance_floor_factor) * torch.finfo(dtype).eps,
        )


@dataclass(frozen=True)
class LinearSolveResult:
    solution: torch.Tensor
    iterations: int
    initial_relative_residual: float
    relative_residual: float
    converged: bool
    initial_guess_used: bool = field(default=False, kw_only=True)


class AdjointBackend(Protocol):
    solver_name: str

    @property
    def adapter_contract(self) -> BackendAdapterContract: ...

    def solve(
        self,
        system: AdjointSystem,
        *,
        slot_key: Hashable | None,
        initial_guess: torch.Tensor | None = None,
    ) -> LinearSolveResult: ...

    def reset(self, slot_key: Hashable | None = None) -> None: ...

    def close(self) -> None: ...


def _validate_tolerance(tolerance: float) -> float:
    if not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    return float(tolerance)


def _relative_residual(system: AdjointSystem, candidate: torch.Tensor) -> float:
    residual = system.matvec(candidate) - system.scaled_rhs
    normalizer = max(
        float(torch.linalg.vector_norm(system.scaled_rhs)),
        torch.finfo(system.scaled_rhs.dtype).tiny,
    )
    return float(torch.linalg.vector_norm(residual)) / normalizer


def _same_reported_scalar(
    reported: float, measured: float, dtype: torch.dtype
) -> bool:
    allowance = 64.0 * torch.finfo(dtype).eps
    return math.isclose(reported, measured, rel_tol=allowance, abs_tol=allowance)


def _system_tensor_versions(
    system: AdjointSystem,
) -> tuple[tuple[torch.Tensor, int | None], ...]:
    tensors = (
        system.state.plan,
        system.source_total,
        system.target_total,
        system.active_target_indices,
        system.plan_block,
        system.rhs_source,
        system.rhs_target,
        system.inverse_sqrt_source,
        system.inverse_target,
        system.scaled_rhs,
        system.grad_plan if isinstance(system.grad_plan, torch.Tensor) else None,
    )
    unique: list[tuple[torch.Tensor, int | None]] = []
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is not None and id(tensor) not in seen:
            seen.add(id(tensor))
            try:
                version = tensor._version
            except RuntimeError:  # inference tensors have no version counter
                version = None
            unique.append((tensor, version))
    return tuple(unique)


def _modified_backend_input(
    system_versions: tuple[tuple[torch.Tensor, int | None], ...],
    initial_guess: torch.Tensor | None,
    initial_snapshot: torch.Tensor | None,
    initial_version: int | None,
) -> str | None:
    for tensor, version in system_versions:
        if version is not None and tensor._version != version:
            return "backend modified an adjoint-system tensor in place"
    if initial_guess is not None:
        try:
            current_version = initial_guess._version
        except RuntimeError:
            current_version = None
        if (
            initial_version is not None
            and current_version != initial_version
        ) or not torch.equal(initial_guess, initial_snapshot):
            return "backend modified initial_guess in place"
    return None


def solve_with_backend_contract(
    backend: AdjointBackend,
    system: AdjointSystem,
    *,
    slot_key: Hashable | None,
    initial_guess: torch.Tensor | None = None,
) -> LinearSolveResult:
    """Execute and independently certify one backend solve.

    Backends may change the solve or preconditioner only. Acceptance always
    uses the complete current plan through :meth:`AdjointSystem.matvec`.
    """

    contract = backend.adapter_contract
    if not isinstance(contract, BackendAdapterContract):
        raise BackendContractError("backend adapter_contract has the wrong type")
    if backend.solver_name != contract.solver_name:
        raise BackendContractError("backend solver_name disagrees with its contract")
    device_type = system.scaled_rhs.device.type
    if (
        contract.accepted_device_types is not None
        and device_type not in contract.accepted_device_types
    ):
        raise BackendContractError(
            f"backend {contract.solver_name!r} does not accept device {device_type!r}"
        )
    if system.scaled_rhs.dtype not in contract.accepted_dtypes:
        raise BackendContractError(
            f"backend {contract.solver_name!r} does not accept dtype "
            f"{system.scaled_rhs.dtype}"
        )
    if initial_guess is not None and (
        not isinstance(initial_guess, torch.Tensor)
        or initial_guess.shape != system.scaled_rhs.shape
        or initial_guess.dtype != system.scaled_rhs.dtype
        or initial_guess.device != system.scaled_rhs.device
        or not bool(torch.all(torch.isfinite(initial_guess)))
    ):
        raise BackendContractError("initial_guess is incompatible with the adjoint system")

    system_versions = _system_tensor_versions(system)
    initial_snapshot = (
        None if initial_guess is None else initial_guess.detach().clone()
    )
    if initial_guess is None:
        initial_version = None
    else:
        try:
            initial_version = initial_guess._version
        except RuntimeError:
            initial_version = None

    try:
        result = backend.solve(
            system,
            slot_key=slot_key,
            initial_guess=initial_guess,
        )
    except Exception as error:
        modification = _modified_backend_input(
            system_versions,
            initial_guess,
            initial_snapshot,
            initial_version,
        )
        if modification is not None:
            raise BackendContractError(modification) from error
        raise
    modification = _modified_backend_input(
        system_versions,
        initial_guess,
        initial_snapshot,
        initial_version,
    )
    if modification is not None:
        raise BackendContractError(modification)
    if not isinstance(result, LinearSolveResult):
        raise BackendContractError("backend must return LinearSolveResult")
    solution = result.solution
    if (
        not isinstance(solution, torch.Tensor)
        or solution.shape != system.scaled_rhs.shape
        or solution.dtype != system.scaled_rhs.dtype
        or solution.device != system.scaled_rhs.device
    ):
        raise BackendContractError("backend solution has incompatible shape, dtype, or device")
    if not bool(torch.all(torch.isfinite(solution))):
        raise BackendContractError("backend solution must be finite")
    if any(solution.is_set_to(tensor) for tensor, _ in system_versions) or (
        initial_guess is not None and solution.is_set_to(initial_guess)
    ):
        raise BackendContractError("backend solution must own independent storage")
    if (
        isinstance(result.iterations, bool)
        or not isinstance(result.iterations, Integral)
        or result.iterations < 0
        or (
            contract.maximum_iterations is not None
            and result.iterations > contract.maximum_iterations
        )
    ):
        raise BackendContractError("backend iteration count violates its declared cap")
    if not isinstance(result.converged, bool):
        raise BackendContractError("backend converged flag must be bool")
    if not isinstance(result.initial_guess_used, bool):
        raise BackendContractError("backend initial_guess_used flag must be bool")
    if result.initial_guess_used and (
        initial_guess is None or contract.initial_guess_policy == "ignored"
    ):
        raise BackendContractError("backend misreported initial-guess use")
    for name, value in (
        ("initial_relative_residual", result.initial_relative_residual),
        ("relative_residual", result.relative_residual),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise BackendContractError(f"backend {name} must be finite and nonnegative")

    initial_candidate = (
        initial_snapshot
        if result.initial_guess_used
        else torch.zeros_like(system.scaled_rhs)
    )
    with torch.no_grad():
        measured_initial = _relative_residual(system, initial_candidate)
        measured_final = _relative_residual(system, solution)
    if not _same_reported_scalar(
        float(result.initial_relative_residual),
        measured_initial,
        system.scaled_rhs.dtype,
    ):
        raise BackendContractError("backend misreported initial residual")
    if not _same_reported_scalar(
        float(result.relative_residual), measured_final, system.scaled_rhs.dtype
    ):
        raise BackendContractError("backend misreported final residual")
    expected_converged = measured_final <= contract.effective_tolerance(
        system.scaled_rhs.dtype
    )
    if result.converged != expected_converged:
        raise BackendContractError("backend convergence flag disagrees with exact residual")

    diagnostics = getattr(result, "diagnostics", None)
    selected = getattr(diagnostics, "selected_method", None)
    if contract.fallback_solvers:
        allowed = (contract.solver_name, *contract.fallback_solvers)
        if selected not in allowed:
            raise BackendContractError(
                "backend selected solver is absent or not declared by its contract"
            )
    elif selected is not None and selected != contract.solver_name:
        raise BackendContractError(
            "backend selected solver is not declared by its contract"
        )
    return result


class DenseBackend:
    """Dense Schur solve, accepted only against the exact full-plan matvec.

    A direct factorization can select a huge roundoff-dominated component when
    the entropic plan is numerically disconnected.  If that happens, retry
    with a diagonal perturbation starting at machine precision.  The
    perturbation stabilizes only the numerical solve: convergence is still
    measured with the original, unshifted full-plan operator.
    """

    solver_name = "dense"

    def __init__(self, *, tolerance: float = 1e-8) -> None:
        self.tolerance = _validate_tolerance(tolerance)
        self._adapter_contract = BackendAdapterContract(
            solver_name=self.solver_name,
            accepted_device_types=None,
            accepted_dtypes=frozenset({torch.float32, torch.float64}),
            state_scope="stateless",
            initial_guess_policy="ignored",
            requested_tolerance=self.tolerance,
            tolerance_floor_factor=100.0,
            maximum_iterations=None,
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
        del slot_key, initial_guess
        if self._closed:
            raise RuntimeError("backend is closed")
        matrix = system.dense_matrix()
        rhs_norm = float(torch.linalg.vector_norm(system.scaled_rhs))
        normalizer = max(rhs_norm, torch.finfo(system.scaled_rhs.dtype).tiny)
        threshold = max(
            self.tolerance, 100.0 * torch.finfo(system.scaled_rhs.dtype).eps
        )

        def relative_residual(candidate: torch.Tensor) -> tuple[torch.Tensor, float]:
            residual = system.matvec(candidate) - system.scaled_rhs
            relative = float(torch.linalg.vector_norm(residual)) / normalizer
            return residual, relative

        def solve_and_measure() -> tuple[torch.Tensor, float]:
            candidate = torch.linalg.solve(matrix, system.scaled_rhs)
            residual, relative = relative_residual(candidate)
            # Standard iterative refinement is used only when the direct
            # factorization misses the exact-operator residual target.  The
            # correction may use a numerically shifted factorization, but
            # acceptance is always measured with the original full-plan
            # operator through ``system.matvec``.
            for _ in range(6):
                if not math.isfinite(relative) or relative <= threshold:
                    break
                correction = torch.linalg.solve(matrix, -residual)
                refined = candidate + correction
                refined_residual, refined_relative = relative_residual(refined)
                if (
                    not math.isfinite(refined_relative)
                    or refined_relative >= relative
                ):
                    break
                candidate = refined
                residual = refined_residual
                relative = refined_relative
            return candidate, relative

        best_solution: torch.Tensor | None = None
        best_relative = math.inf
        try:
            solution, relative = solve_and_measure()
            if math.isfinite(relative):
                best_solution, best_relative = solution, relative
        except RuntimeError:
            relative = math.inf

        if relative > threshold:
            scale = max(1.0, float(torch.max(torch.abs(matrix))))
            added_shift = 0.0
            trial_shift = torch.finfo(matrix.dtype).eps * scale
            for _ in range(8):
                matrix.diagonal().add_(trial_shift - added_shift)
                added_shift = trial_shift
                try:
                    candidate, candidate_relative = solve_and_measure()
                except RuntimeError:
                    candidate_relative = math.inf
                if candidate_relative < best_relative:
                    best_solution = candidate
                    best_relative = candidate_relative
                if candidate_relative <= threshold:
                    break
                trial_shift *= 10.0

        if best_solution is None:
            raise RuntimeError("dense Schur solve failed for every stabilization trial")
        solution = best_solution
        relative = best_relative
        return LinearSolveResult(
            solution=solution,
            iterations=0,
            initial_relative_residual=0.0 if rhs_norm == 0.0 else 1.0,
            relative_residual=relative,
            converged=relative <= threshold,
            initial_guess_used=False,
        )

    def reset(self, slot_key: Hashable | None = None) -> None:
        del slot_key
        if self._closed:
            raise RuntimeError("backend is closed")

    def close(self) -> None:
        self._closed = True


class JacobiWBackend:
    """Warm-startable PCG on the exact Jacobi-scaled Schur system."""

    solver_name = "jacobi_w"

    def __init__(
        self,
        *,
        tolerance: float = 1e-8,
        maximum_iterations: int = 1_000,
        residual_check_interval: int = 1,
        validate_spd: bool = True,
    ) -> None:
        self.tolerance = _validate_tolerance(tolerance)
        if maximum_iterations <= 0 or residual_check_interval <= 0:
            raise ValueError("PCG iteration controls must be positive")
        self.maximum_iterations = int(maximum_iterations)
        self.residual_check_interval = int(residual_check_interval)
        self.validate_spd = bool(validate_spd)
        self._adapter_contract = BackendAdapterContract(
            solver_name=self.solver_name,
            accepted_device_types=None,
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
        result = pcg(
            system.matvec,
            system.scaled_rhs,
            tolerance=self.tolerance,
            maximum_iterations=self.maximum_iterations,
            initial_guess=initial_guess,
            residual_check_interval=self.residual_check_interval,
            validate_spd=self.validate_spd,
        )
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


def create_backend(method: str, **kwargs) -> AdjointBackend:
    """Instantiate one canonical method without importing CUDA eagerly."""

    if method == "dense":
        return DenseBackend(**kwargs)
    if method == "jacobi_w":
        return JacobiWBackend(**kwargs)
    if method == "ic0":
        from .sparse.ic0 import IC0Backend

        return IC0Backend(**kwargs)
    if method == "sparse_w":
        from .sparse.backend import SparseWBackend

        return SparseWBackend(**kwargs)
    raise ValueError(
        f"unknown method {method!r}; expected one of {', '.join(CANONICAL_METHODS)}"
    )


# Compatibility for code that previously called the unpreconditioned solver
# PCG. Its canonical experiment name is now Jacobi-W / ``jacobi_w``.
PCGBackend = JacobiWBackend
