"""The mode-explicit PyTorch interface for standard balanced EOT."""

from __future__ import annotations

import math
from numbers import Real
from typing import Hashable

import torch

from .backends.auto import AutoBackwardDiagnostics, ProgressiveAutoBackend
from .backends import CANONICAL_METHODS, create_backend
from ._core.diagnostics import SolveDiagnostics
from ._core.envelope import envelope_transport_value
from .forward import ForwardConfig
from ._core.implicit import implicit_transport_plan
from ._core.session import EOTSolverSession
from ._core.unrolled import unrolled_transport_plan


MODES = ("unrolled", "implicit", "envelope")
IMPLICIT_SOLVERS = (*CANONICAL_METHODS, "auto")
DEFAULT_FORWARD_TOL = 1e-10
DEFAULT_FORWARD_ITER_MAX = 10_000
DEFAULT_EPSILON_SCALING_DECAY = 0.5
DEFAULT_BACKWARD_SOLVER = "dense"
DEFAULT_BACKWARD_TOL = 1e-8
DEFAULT_BACKWARD_ITER_MAX = 1_000


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive; supply a real scalar")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _positive_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _open_unit_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar strictly between 0 and 1")
    if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return float(value)


def _select_vector(values: torch.Tensor, index: int, batch_size: int) -> torch.Tensor:
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[0] == batch_size:
        return values[index]
    raise ValueError("marginals must be shared vectors or batched matrices")


def _validate_inputs(
    cost: torch.Tensor, source: torch.Tensor, target: torch.Tensor
) -> None:
    for name, value in (("cost", cost), ("source", source), ("target", target)):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{name} must be a dense strided tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use torch.float32 or torch.float64")
        if value.dtype != cost.dtype or value.device != cost.device:
            raise ValueError("cost and marginals must use the same dtype and device")
    if cost.ndim not in (2, 3):
        raise ValueError("cost must have shape (m,n) or (batch,m,n)")
    if any(size == 0 for size in cost.shape):
        raise ValueError("cost dimensions and batch size must be nonempty")
    for name, value, size in (
        ("source", source, cost.shape[-2]),
        ("target", target, cost.shape[-1]),
    ):
        allowed = ((size,),) if cost.ndim == 2 else ((size,), (cost.shape[0], size))
        if tuple(value.shape) not in allowed:
            raise ValueError(f"{name} shape must be one of {allowed}")


class EOTLayer(torch.nn.Module):
    """A standard balanced EOT problem embedded as a PyTorch layer.

    ``unrolled`` returns a plan after exactly ``iter_num`` Sinkhorn updates.
    PyTorch retains and differentiates the complete iteration graph.

    ``implicit`` returns a plan meeting the forward marginal tolerance. Its
    first-order backward evaluates the fixed-point adjoint formula at that
    approximate plan, using the declared marginals in the linear system.

    ``envelope`` evaluates the regularized EOT value at the accepted plan. Its
    first-order backward uses that plan and grounded dual potentials.

    ``epsilon_scaling_start`` optionally enables a geometric continuation
    schedule for converged modes.  The schedule always terminates at the
    declared ``epsilon`` before a plan can be accepted.

    See ``docs/api-contract.md`` for the frozen numerical and diagnostic
    contract, including the effective Dense tolerance and precision limits.
    """

    def __init__(
        self,
        epsilon: float,
        mode: str,
        *,
        iter_num: int | None = None,
        forward_tol: float | None = None,
        forward_iter_max: int | None = None,
        epsilon_scaling_start: float | None = None,
        epsilon_scaling_decay: float | None = None,
        backward_solver: str | None = None,
        backward_tol: float | None = None,
        backward_iter_max: int | None = None,
    ) -> None:
        super().__init__()
        self._epsilon = _positive_float(epsilon, "epsilon")
        if mode not in MODES:
            raise ValueError("mode must be 'unrolled', 'implicit', or 'envelope'")
        self._mode = mode
        self._session: EOTSolverSession | None = None
        self._auto_backend: ProgressiveAutoBackend | None = None
        self._closed = False
        self._diagnostics: tuple[SolveDiagnostics, ...] = ()

        if mode == "unrolled":
            if iter_num is None:
                raise ValueError("iter_num is required for unrolled mode")
            self._iter_num = _positive_integer(iter_num, "iter_num")
            for name, value in (
                ("forward_tol", forward_tol),
                ("forward_iter_max", forward_iter_max),
                ("epsilon_scaling_start", epsilon_scaling_start),
                ("epsilon_scaling_decay", epsilon_scaling_decay),
                ("backward_solver", backward_solver),
                ("backward_tol", backward_tol),
                ("backward_iter_max", backward_iter_max),
            ):
                if value is not None:
                    raise ValueError(f"{name} is not valid for unrolled mode")
            self._forward_tol = None
            self._forward_iter_max = None
            self._epsilon_scaling_start = None
            self._epsilon_scaling_decay = None
            self._backward_solver = None
            self._backward_tol = None
            self._backward_iter_max = None
            self._forward_config = None
            return

        if iter_num is not None:
            raise ValueError("iter_num is only valid for unrolled mode")
        self._iter_num = None
        self._forward_tol = _positive_float(
            DEFAULT_FORWARD_TOL if forward_tol is None else forward_tol,
            "forward_tol",
        )
        self._forward_iter_max = _positive_integer(
            DEFAULT_FORWARD_ITER_MAX
            if forward_iter_max is None
            else forward_iter_max,
            "forward_iter_max",
        )
        self._epsilon_scaling_start = (
            None
            if epsilon_scaling_start is None
            else _positive_float(epsilon_scaling_start, "epsilon_scaling_start")
        )
        if epsilon_scaling_decay is not None and epsilon_scaling_start is None:
            raise ValueError(
                "epsilon_scaling_decay requires epsilon_scaling_start"
            )
        self._epsilon_scaling_decay = (
            None
            if self._epsilon_scaling_start is None
            else (
                DEFAULT_EPSILON_SCALING_DECAY
                if epsilon_scaling_decay is None
                else _open_unit_float(
                    epsilon_scaling_decay, "epsilon_scaling_decay"
                )
            )
        )
        if (
            self._epsilon_scaling_start is not None
            and self._epsilon_scaling_start < self._epsilon
        ):
            raise ValueError(
                "epsilon_scaling_start must be greater than or equal to epsilon"
            )
        self._forward_config = ForwardConfig(
            tolerance=self._forward_tol,
            maximum_iterations=self._forward_iter_max,
            epsilon_scaling_start=self._epsilon_scaling_start,
            epsilon_scaling_decay=(
                DEFAULT_EPSILON_SCALING_DECAY
                if self._epsilon_scaling_decay is None
                else self._epsilon_scaling_decay
            ),
        )

        if mode == "envelope":
            for name, value in (
                ("backward_solver", backward_solver),
                ("backward_tol", backward_tol),
                ("backward_iter_max", backward_iter_max),
            ):
                if value is not None:
                    raise ValueError(f"{name} is not valid for envelope mode")
            self._backward_solver = None
            self._backward_tol = None
            self._backward_iter_max = None
            return

        selected_solver = (
            DEFAULT_BACKWARD_SOLVER
            if backward_solver is None
            else backward_solver
        )
        if selected_solver not in IMPLICIT_SOLVERS:
            raise ValueError(
                "backward_solver must be one of " + ", ".join(IMPLICIT_SOLVERS)
            )
        self._backward_solver = selected_solver
        if selected_solver == "auto":
            self._forward_config = ForwardConfig(
                tolerance=self._forward_tol,
                maximum_iterations=self._forward_iter_max,
                epsilon_scaling_start=self._epsilon_scaling_start,
                epsilon_scaling_decay=(
                    DEFAULT_EPSILON_SCALING_DECAY
                    if self._epsilon_scaling_decay is None
                    else self._epsilon_scaling_decay
                ),
                collect_screening_features=True,
            )
        self._backward_tol = _positive_float(
            DEFAULT_BACKWARD_TOL if backward_tol is None else backward_tol,
            "backward_tol",
        )
        self._backward_iter_max = _positive_integer(
            DEFAULT_BACKWARD_ITER_MAX
            if backward_iter_max is None
            else backward_iter_max,
            "backward_iter_max",
        )
        if selected_solver == "auto":
            self._auto_backend = ProgressiveAutoBackend(
                tolerance=self._backward_tol,
                maximum_iterations=self._backward_iter_max,
            )
            backend = self._auto_backend
        else:
            backend_options: dict[str, object] = {
                "tolerance": self._backward_tol
            }
            if selected_solver != "dense":
                backend_options["maximum_iterations"] = self._backward_iter_max
            if selected_solver in {"ic0", "sparse_w"}:
                backend_options.update(dropped_mass=0.005, diagonal_shift=1e-10)
            backend = create_backend(selected_solver, **backend_options)
        self._session = EOTSolverSession(adjoint_backend=backend)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def iter_num(self) -> int | None:
        return self._iter_num

    @property
    def forward_tol(self) -> float | None:
        return self._forward_tol

    @property
    def forward_iter_max(self) -> int | None:
        return self._forward_iter_max

    @property
    def epsilon_scaling_start(self) -> float | None:
        return self._epsilon_scaling_start

    @property
    def epsilon_scaling_decay(self) -> float | None:
        return self._epsilon_scaling_decay

    @property
    def backward_solver(self) -> str | None:
        return self._backward_solver

    @property
    def backward_tol(self) -> float | None:
        return self._backward_tol

    @property
    def backward_iter_max(self) -> int | None:
        return self._backward_iter_max

    @property
    def closed(self) -> bool:
        if self._session is not None:
            return self._session.closed
        return self._closed

    @property
    def auto_diagnostics(self) -> dict[Hashable, AutoBackwardDiagnostics]:
        """Last automatic-backend decision for each internal problem slot."""

        if self._auto_backend is None:
            return {}
        return self._auto_backend.diagnostics

    @property
    def diagnostics(self) -> tuple[dict[str, object], ...]:
        """Scalar snapshots for the latest forward attempt, in batch order.

        Backward updates the record belonging to its own forward graph. Read
        this property again after backward; previously returned snapshots do
        not change. Metadata-invalid calls clear the records.
        """

        return tuple(record.snapshot() for record in self._diagnostics)

    def _forward_single(
        self,
        cost: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        *,
        diagnostics: SolveDiagnostics,
        slot_key: Hashable | None = None,
    ) -> torch.Tensor:
        if self._mode == "unrolled":
            assert self._iter_num is not None
            plan = unrolled_transport_plan(
                cost,
                source,
                target,
                epsilon=self._epsilon,
                iterations=self._iter_num,
            )
            if not bool(torch.isfinite(plan).all()):
                raise RuntimeError("unrolled Sinkhorn produced a nonfinite plan")
            diagnostics.record_unrolled(plan, source, target, self._iter_num)
            return plan
        assert self._forward_config is not None
        if self._mode == "implicit":
            assert self._session is not None
            return implicit_transport_plan(
                cost,
                source,
                target,
                epsilon=self._epsilon,
                forward_config=self._forward_config,
                session=self._session,
                slot_key=slot_key,
                diagnostics=diagnostics,
            )
        return envelope_transport_value(
            cost,
            source,
            target,
            epsilon=self._epsilon,
            forward_config=self._forward_config,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        cost: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one problem or a batch with shared/batched marginals."""

        self._diagnostics = ()
        if self.closed:
            raise RuntimeError("EOTLayer is closed")
        _validate_inputs(cost, source, target)
        batch_size = 1 if cost.ndim == 2 else cost.shape[0]
        self._diagnostics = tuple(
            SolveDiagnostics(
                self._mode, index, self._forward_tol,
                self._backward_solver, self._backward_tol,
            )
            for index in range(batch_size)
        )
        outputs = []
        for index, record in enumerate(self._diagnostics):
            try:
                outputs.append(self._forward_single(
                    cost if cost.ndim == 2 else cost[index],
                    source if cost.ndim == 2 else _select_vector(source, index, batch_size),
                    target if cost.ndim == 2 else _select_vector(target, index, batch_size),
                    diagnostics=record,
                    slot_key=index if self._backward_solver == "auto" else None,
                ))
            except Exception as error:
                record.fail("forward", error)
                raise
        return outputs[0] if cost.ndim == 2 else torch.stack(outputs)

    def close(self) -> None:
        """Release resources owned by an implicit backward session."""

        if self._session is not None:
            self._session.close()
        self._closed = True
