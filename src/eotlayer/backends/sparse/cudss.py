"""Lazy nvmath/cuDSS wrapper for one fixed CUDA CSR matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
import operator

import numpy as np
import torch


_CUDSS_MEMORY_SOURCE = "DirectSolverPlanInfo.memory_estimates"
_CUDSS_MEMORY_FIELDS = (
    "permanent_device_memory",
    "peak_device_memory",
    "permanent_host_memory",
    "peak_host_memory",
    "hybrid_min_device_memory",
    "hybrid_max_device_memory",
)
_CUDSS_WORKSPACE_UNAVAILABLE = (
    "nvmath DirectSolverPlanInfo exposes aggregate cuDSS memory estimates, "
    "not a distinct workspace_bytes value"
)


def _ensure_size_written_buffer(thread_state: object) -> None:
    """Initialize nvmath 0.6's per-thread cuDSS query buffer.

    nvmath 0.6 initializes ``_tls.size_written`` only in the importing thread.
    PyTorch may execute a custom backward in another worker thread, where
    querying ``FactorizationInfo.info`` otherwise raises ``AttributeError``.
    Newer nvmath versions can initialize the field themselves; preserving an
    existing value keeps this compatibility shim a no-op there.
    """

    if not hasattr(thread_state, "size_written"):
        setattr(thread_state, "size_written", np.empty((1,), dtype=np.uint64))


def _initialize_nvmath_cudss_thread_state() -> None:
    """Feature-detect and initialize the private nvmath 0.6 thread state."""

    try:
        from nvmath.sparse._internal import cudss_data_ifc
    except (ImportError, AttributeError):
        return
    thread_state = getattr(cudss_data_ifc, "_tls", None)
    if thread_state is not None:
        _ensure_size_written_buffer(thread_state)


def _memory_query_reason(prefix: str, error: Exception) -> str:
    detail = str(error).strip() or type(error).__name__
    return f"{prefix}: {detail}"


def _memory_bytes(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("memory bytes must be an integer, not bool")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise TypeError("memory bytes must be an integer") from error
    if normalized < 0:
        raise ValueError("memory bytes must be nonnegative")
    return int(normalized)


@dataclass(frozen=True)
class CuDSSPlanMemorySnapshot:
    """Immutable plan-time memory metadata detached from native cuDSS state.

    nvmath's ``DirectSolverPlanInfo`` is backed by the live solver.  Keeping
    that object in a solve result and querying it after ``free()`` is unsafe.
    This value object contains only Python integers and strings copied while
    the solver is live, so it remains valid after factorization teardown.
    """

    permanent_device_bytes: int | None
    peak_device_bytes: int | None
    permanent_host_bytes: int | None
    peak_host_bytes: int | None
    hybrid_min_device_bytes: int | None
    hybrid_max_device_bytes: int | None
    memory_estimates_unavailable_reason: str | None
    workspace_bytes: int | None
    workspace_unavailable_reason: str | None
    source: str = _CUDSS_MEMORY_SOURCE

    def __post_init__(self) -> None:
        estimates = (
            self.permanent_device_bytes,
            self.peak_device_bytes,
            self.permanent_host_bytes,
            self.peak_host_bytes,
            self.hybrid_min_device_bytes,
            self.hybrid_max_device_bytes,
        )
        if self.memory_estimates_unavailable_reason is None:
            if any(value is None for value in estimates):
                raise ValueError(
                    "available cuDSS memory estimates require every byte value"
                )
            normalized = tuple(_memory_bytes(value) for value in estimates)
            for name, value in zip(
                (
                    "permanent_device_bytes",
                    "peak_device_bytes",
                    "permanent_host_bytes",
                    "peak_host_bytes",
                    "hybrid_min_device_bytes",
                    "hybrid_max_device_bytes",
                ),
                normalized,
            ):
                object.__setattr__(self, name, value)
        else:
            if not self.memory_estimates_unavailable_reason.strip():
                raise ValueError("cuDSS estimate unavailability requires a reason")
            if any(value is not None for value in estimates):
                raise ValueError(
                    "unavailable cuDSS estimates cannot contain byte values"
                )
        if self.workspace_bytes is None:
            if (
                not isinstance(self.workspace_unavailable_reason, str)
                or not self.workspace_unavailable_reason.strip()
            ):
                raise ValueError("workspace unavailability requires a reason")
        else:
            object.__setattr__(self, "workspace_bytes", _memory_bytes(self.workspace_bytes))
            if self.workspace_unavailable_reason is not None:
                raise ValueError(
                    "an available cuDSS workspace value cannot have a reason"
                )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("cuDSS memory snapshot source must be nonempty")

    @property
    def estimates_available(self) -> bool:
        return self.memory_estimates_unavailable_reason is None

    @classmethod
    def unavailable(cls, reason: str) -> "CuDSSPlanMemorySnapshot":
        return cls(
            permanent_device_bytes=None,
            peak_device_bytes=None,
            permanent_host_bytes=None,
            peak_host_bytes=None,
            hybrid_min_device_bytes=None,
            hybrid_max_device_bytes=None,
            memory_estimates_unavailable_reason=reason,
            workspace_bytes=None,
            workspace_unavailable_reason=_CUDSS_WORKSPACE_UNAVAILABLE,
        )


def _structured_memory_record(raw: object) -> object:
    dtype = getattr(raw, "dtype", None)
    names = getattr(dtype, "names", None)
    if not names:
        return raw
    size = getattr(raw, "size", None)
    if size == 0:
        raise ValueError("memory_estimates is empty")
    shape = getattr(raw, "shape", None)
    if shape == () or shape is None:
        return raw
    return raw.reshape(-1)[0]


def _extract_plan_memory_values(raw: object) -> tuple[int, ...]:
    raw = _structured_memory_record(raw)
    values: list[int] = []
    missing: list[str] = []
    for index, name in enumerate(_CUDSS_MEMORY_FIELDS):
        try:
            if isinstance(raw, Mapping):
                value = raw[name]
            elif getattr(getattr(raw, "dtype", None), "names", None):
                value = raw[name]
            elif hasattr(raw, name):
                value = getattr(raw, name)
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                value = raw[index]
            else:
                missing.append(name)
                continue
        except (KeyError, IndexError):
            missing.append(name)
            continue
        try:
            values.append(_memory_bytes(value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid {name}: {error}") from error
    if missing:
        raise ValueError(
            "missing cuDSS memory estimate fields: " + ", ".join(missing)
        )
    if len(values) != len(_CUDSS_MEMORY_FIELDS):
        raise ValueError("incomplete cuDSS memory estimates")
    return tuple(values)


def snapshot_cudss_plan_memory(plan_info: object | None) -> CuDSSPlanMemorySnapshot:
    """Copy plan memory metadata before its native solver can be freed."""

    if plan_info is None:
        return CuDSSPlanMemorySnapshot.unavailable("cuDSS plan info is unavailable")
    sentinel = object()
    try:
        raw = getattr(plan_info, "memory_estimates", sentinel)
    except Exception as error:
        return CuDSSPlanMemorySnapshot.unavailable(
            _memory_query_reason("cuDSS memory_estimates query failed", error)
        )
    if raw is sentinel:
        return CuDSSPlanMemorySnapshot.unavailable(
            "cuDSS plan info does not expose memory_estimates"
        )
    if raw is None:
        return CuDSSPlanMemorySnapshot.unavailable(
            "cuDSS memory_estimates is unavailable or not supported"
        )
    try:
        values = _extract_plan_memory_values(raw)
    except (TypeError, ValueError) as error:
        return CuDSSPlanMemorySnapshot.unavailable(
            f"invalid cuDSS memory_estimates: {error}"
        )
    return CuDSSPlanMemorySnapshot(
        permanent_device_bytes=values[0],
        peak_device_bytes=values[1],
        permanent_host_bytes=values[2],
        peak_host_bytes=values[3],
        hybrid_min_device_bytes=values[4],
        hybrid_max_device_bytes=values[5],
        memory_estimates_unavailable_reason=None,
        workspace_bytes=None,
        workspace_unavailable_reason=_CUDSS_WORKSPACE_UNAVAILABLE,
    )


def _multithreading_library() -> str | None:
    for package in ("nvidia-cudss-cu12", "nvidia-cudss-cu13"):
        try:
            installed = distribution(package)
        except PackageNotFoundError:
            continue
        for relative in (
            "nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0",
            "nvidia/cu13/lib/libcudss_mtlayer_gomp.so.0",
        ):
            candidate = installed.locate_file(relative)
            if candidate.is_file():
                return str(candidate)
    return None


class CuDSSFixedCSR:
    """Symbolically plan once and numerically refactor one upper CUDA CSR."""

    def __init__(self, matrix: torch.Tensor) -> None:
        if (
            not isinstance(matrix, torch.Tensor)
            or matrix.layout != torch.sparse_csr
            or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or matrix.device.type != "cuda"
            or matrix.dtype not in {torch.float32, torch.float64}
            or matrix.crow_indices().dtype != torch.int32
            or matrix.col_indices().dtype != torch.int32
        ):
            raise ValueError(
                "cuDSS requires square float32/float64 CUDA CSR with int32 indices"
            )
        try:
            from nvmath.sparse import advanced
        except ImportError as error:
            raise RuntimeError("Sparse-W on CUDA requires nvmath-python and cuDSS") from error

        self.matrix = matrix
        self.dimension = int(matrix.shape[0])
        self._rhs = torch.zeros(self.dimension, dtype=matrix.dtype, device=matrix.device)
        options = advanced.DirectSolverOptions(
            sparse_system_type=advanced.DirectSolverMatrixType.SPD,
            sparse_system_view=advanced.DirectSolverMatrixViewType.UPPER,
            multithreading_lib=_multithreading_library(),
            blocking=True,
        )
        self._solver = advanced.DirectSolver(matrix, self._rhs, options=options)
        self._memory_snapshot: CuDSSPlanMemorySnapshot | None = None
        self._planned = False
        self._factorized = False
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("cuDSS solver is closed")

    @property
    def memory_snapshot(self) -> CuDSSPlanMemorySnapshot | None:
        return self._memory_snapshot

    def plan(self) -> CuDSSPlanMemorySnapshot | None:
        self._require_open()
        _initialize_nvmath_cudss_thread_state()
        if not self._planned:
            plan_info = self._solver.plan()
            self._memory_snapshot = snapshot_cudss_plan_memory(plan_info)
            self._planned = True
        return self._memory_snapshot

    def factorize(self) -> None:
        self._require_open()
        _initialize_nvmath_cudss_thread_state()
        if not self._planned:
            raise RuntimeError("plan() must precede factorize()")
        info = self._solver.factorize()
        if int(info.info) != 0:
            raise RuntimeError("cuDSS reported a nonzero factorization status")
        self._factorized = True

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        self._require_open()
        _initialize_nvmath_cudss_thread_state()
        if not self._factorized:
            raise RuntimeError("factorize() must precede solve()")
        if (
            rhs.ndim != 1
            or rhs.numel() != self.dimension
            or rhs.dtype != self.matrix.dtype
            or rhs.device != self.matrix.device
            or not torch.all(torch.isfinite(rhs))
        ):
            raise ValueError("right_hand_side is incompatible with the cuDSS matrix")
        self._rhs.copy_(rhs)
        return self._solver.solve()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._solver.free()
