"""Scalar diagnostic snapshots for one public-layer problem instance."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import torch

from ..backward import TransportBackwardResult
from ..forward import TransportState, _errors


class SolveDiagnostics:
    """Track one forward/backward pair without retaining tensors or graphs."""

    def __init__(
        self,
        mode: str,
        batch_index: int,
        forward_tolerance: float | None,
        backward_solver: str | None,
        backward_tolerance: float | None,
    ) -> None:
        self._values = {
            "mode": mode,
            "batch_index": int(batch_index),
            "forward": {
                "status": "pending",
                "iterations": None,
                "absolute_marginal_error": None,
                "normalized_marginal_error": None,
                "tolerance": (
                    None if forward_tolerance is None else float(forward_tolerance)
                ),
                "error": None,
            },
            "backward": {
                "status": (
                    "pending" if mode in {"implicit", "envelope"} else "not_applicable"
                ),
                "solver": backward_solver,
                "selected_solver": None,
                "iterations": None,
                "initial_relative_residual": None,
                "relative_residual": None,
                "requested_tolerance": (
                    None if backward_tolerance is None else float(backward_tolerance)
                ),
                "effective_tolerance": None,
                "initial_guess_used": None,
                "error": None,
            },
        }

    def snapshot(self) -> dict[str, object]:
        """Return an independent copy that callers may freely mutate."""

        return deepcopy(self._values)

    def record_forward(self, state: TransportState) -> None:
        self._values["forward"].update(
            status="converged" if state.converged else "not_converged",
            iterations=int(state.iterations),
            absolute_marginal_error=float(state.absolute_marginal_error),
            normalized_marginal_error=float(state.normalized_marginal_error),
            error=None,
        )

    def record_unrolled(
        self,
        plan: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        iterations: int,
    ) -> None:
        with torch.no_grad():
            absolute, normalized = _errors(
                plan.sum(1) - source,
                plan.sum(0) - target,
                source,
                target,
            )
        self._values["forward"].update(
            status="completed",
            iterations=int(iterations),
            absolute_marginal_error=absolute,
            normalized_marginal_error=normalized,
            tolerance=None,
            error=None,
        )

    def record_backward(self, result: TransportBackwardResult) -> None:
        backward = self._values["backward"]
        linear_result = result.linear_result
        selected_solver = backward["solver"]
        if selected_solver == "auto":
            selected_solver = linear_result.diagnostics.selected_method
        effective_tolerance = backward["requested_tolerance"]
        if backward["solver"] == "dense" and effective_tolerance is not None:
            effective_tolerance = max(
                effective_tolerance,
                100.0 * torch.finfo(linear_result.solution.dtype).eps,
            )
        backward.update(
            status="converged" if result.converged else "not_converged",
            selected_solver=selected_solver,
            iterations=int(result.iterations),
            initial_relative_residual=float(result.initial_relative_residual),
            relative_residual=float(result.relative_residual),
            effective_tolerance=effective_tolerance,
            initial_guess_used=bool(linear_result.initial_guess_used),
            error=None,
        )

    def begin_backward(self) -> None:
        """Discard measurements from an earlier attempt on a retained graph."""

        self._values["backward"].update(
            status="pending", selected_solver=None, iterations=None,
            initial_relative_residual=None, relative_residual=None,
            effective_tolerance=None, initial_guess_used=None, error=None,
        )

    def record_envelope_backward(self) -> None:
        self._values["backward"].update(status="completed", error=None)

    def fail(self, stage: Literal["forward", "backward"], exc: Exception) -> None:
        """Preserve observed diagnostics and attach the reported failure."""

        values = self._values[stage]
        status = "not_converged" if values["status"] == "not_converged" else "error"
        values.update(status=status, error=str(exc))
