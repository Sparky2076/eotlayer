"""Optional Flash-Sinkhorn specialization for squared-Euclidean EOT.

The module contains no eager Flash-Sinkhorn import.  CPU-only users can import
the canonical :mod:`eotlayer` package without installing Triton or CUDA
dependencies.  The public root interface remains unchanged; eligible case
adapters opt into this backend explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Callable

import torch

from .constraints import BalancedConstraints


@dataclass(frozen=True)
class _FlashAPI:
    """Pinned low-level Flash-Sinkhorn functions used by this adapter."""

    solve_alternating: Callable[..., object]
    apply_plan_vec: Callable[..., torch.Tensor]
    apply_plan_mat: Callable[..., torch.Tensor]
    version: str


def _load_flash_api() -> _FlashAPI:
    """Load the optional dependency only when the structured path is used."""

    try:
        package = importlib.import_module("flash_sinkhorn")
        solvers = importlib.import_module("flash_sinkhorn.sinkhorn_solvers")
        apply = importlib.import_module("flash_sinkhorn.kernels.apply_flash")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "flash_sqeuclidean requires the optional Flash-Sinkhorn dependency; "
            "install it with: pip install -e '.[flash]'"
        ) from error
    return _FlashAPI(
        solve_alternating=solvers.sinkhorn_flashstyle_alternating,
        apply_plan_vec=apply.apply_plan_vec_flashstyle,
        apply_plan_mat=apply.apply_plan_mat_flashstyle,
        version=str(getattr(package, "__version__", "unknown")),
    )


def _validate_inputs(
    x: torch.Tensor,
    source: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float,
    forward_tol: float,
    forward_iter_max: int,
    require_cuda: bool,
    require_float32: bool,
) -> None:
    for value, name in ((x, "x"), (y, "y")):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(f"{name} must be a matrix")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if value.numel() == 0 or not torch.all(torch.isfinite(value)):
            raise ValueError(f"{name} must be finite and nonempty")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    constraints = BalancedConstraints(source, target)
    if x.shape[0] != constraints.source.numel():
        raise ValueError("x rows must match the source marginal")
    if y.shape[0] != constraints.target.numel():
        raise ValueError("y rows must match the target marginal")
    for value, name in ((y, "y"), (source, "source"), (target, "target")):
        if value.dtype != x.dtype or value.device != x.device:
            raise ValueError(f"{name} must match x dtype and device")
    if require_cuda and not x.is_cuda:
        raise ValueError("flash_sqeuclidean requires CUDA tensors")
    if require_float32 and x.dtype != torch.float32:
        raise TypeError("flash_sqeuclidean requires explicit torch.float32 inputs")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(float(cost_scale)) or float(cost_scale) <= 0:
        raise ValueError("cost_scale must be finite and positive")
    if not math.isfinite(float(forward_tol)) or float(forward_tol) <= 0:
        raise ValueError("forward_tol must be finite and positive")
    if (
        not isinstance(forward_iter_max, int)
        or isinstance(forward_iter_max, bool)
        or forward_iter_max <= 0
    ):
        raise ValueError("forward_iter_max must be a positive integer")


@dataclass(frozen=True)
class FlashSquaredEuclideanPlanOperator:
    """Represent ``P`` through supports and shifted dual potentials."""

    x: torch.Tensor
    y: torch.Tensor
    f_hat: torch.Tensor
    g_hat: torch.Tensor
    log_source: torch.Tensor
    log_target: torch.Tensor
    epsilon: float
    cost_scale: float
    api: _FlashAPI
    allow_tf32: bool = False
    use_exp2: bool = True
    autotune: bool = True

    def __post_init__(self) -> None:
        n, dimension = self.x.shape
        m, target_dimension = self.y.shape
        if dimension != target_dimension:
            raise ValueError("x and y must have the same feature dimension")
        for value, expected, name in (
            (self.f_hat, n, "f_hat"),
            (self.log_source, n, "log_source"),
            (self.g_hat, m, "g_hat"),
            (self.log_target, m, "log_target"),
        ):
            if value.ndim != 1 or value.shape[0] != expected:
                raise ValueError(f"{name} has the wrong shape")

    @property
    def shape(self) -> torch.Size:
        return torch.Size((self.x.shape[0], self.y.shape[0]))

    @property
    def dtype(self) -> torch.dtype:
        return self.x.dtype

    @property
    def device(self) -> torch.device:
        return self.x.device

    def _validate_vector(
        self, vector: torch.Tensor, expected: int, operation: str
    ) -> None:
        if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
            raise ValueError(f"{operation} operand must be a vector")
        if vector.shape[0] != expected:
            raise ValueError(f"{operation} operand must have length {expected}")
        if vector.dtype != self.dtype or vector.device != self.device:
            raise ValueError(f"{operation} operand must match operator metadata")

    def _apply_matrix(self, matrix: torch.Tensor, *, axis: int) -> torch.Tensor:
        return self.api.apply_plan_mat(
            self.x,
            self.y,
            self.f_hat,
            self.g_hat,
            self.log_source,
            self.log_target,
            matrix,
            eps=float(self.epsilon),
            axis=axis,
            cost_scale=float(self.cost_scale),
            allow_tf32=bool(self.allow_tf32),
            use_exp2=bool(self.use_exp2),
            autotune=bool(self.autotune),
        )

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector, self.shape[1], "matvec")
        return self._apply_columns(vector[:, None], transpose=False)[:, 0]

    def rmatvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector, self.shape[0], "rmatvec")
        return self._apply_columns(vector[:, None], transpose=True)[:, 0]

    def _apply_columns(self, matrix: torch.Tensor, *, transpose: bool) -> torch.Tensor:
        expected = self.shape[0] if transpose else self.shape[1]
        operation = "rmatmat" if transpose else "matmat"
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            raise ValueError(f"{operation} operand must be a matrix")
        if matrix.shape[0] != expected:
            raise ValueError(
                f"{operation} operand must have leading dimension {expected}"
            )
        if matrix.dtype != self.dtype or matrix.device != self.device:
            raise ValueError(f"{operation} operand must match operator metadata")
        output_rows = self.shape[1] if transpose else self.shape[0]
        if matrix.shape[1] == 0:
            return matrix.new_empty((output_rows, 0))
        geometry_dimension = self.x.shape[1]
        axis = 0 if transpose else 1
        outputs = []
        for start in range(0, matrix.shape[1], geometry_dimension):
            block = matrix[:, start : start + geometry_dimension]
            width = block.shape[1]
            if width < geometry_dimension:
                padding = block.new_zeros((block.shape[0], geometry_dimension - width))
                block = torch.cat((block, padding), dim=1)
            outputs.append(self._apply_matrix(block, axis=axis)[:, :width])
        return torch.cat(outputs, dim=1)

    def matmat(self, matrix: torch.Tensor) -> torch.Tensor:
        # The upstream fused matrix kernel ties feature width to geometry width.
        # Column-wise vector application preserves the general operator contract.
        return self._apply_columns(matrix, transpose=False)

    def rmatmat(self, matrix: torch.Tensor) -> torch.Tensor:
        return self._apply_columns(matrix, transpose=True)

    def materialize(self) -> torch.Tensor:
        """Explicit debug path; this deliberately allocates an ``n x m`` plan."""

        logits = (
            self.log_source[:, None]
            + self.log_target[None, :]
            + (
                self.f_hat[:, None]
                + self.g_hat[None, :]
                + 2.0 * self.cost_scale * (self.x @ self.y.transpose(0, 1))
            )
            / self.epsilon
        )
        return torch.exp(logits)

    def marginal_errors(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.matvec(target.new_ones(target.shape))
        column = self.rmatvec(source.new_ones(source.shape))
        row_l1 = torch.linalg.vector_norm(row - source, ord=1)
        column_l1 = torch.linalg.vector_norm(column - target, ord=1)
        absolute = torch.maximum(row_l1, column_l1)
        normalized = torch.maximum(row_l1 / source.sum(), column_l1 / target.sum())
        return absolute, normalized


class _FlashSquaredEuclideanEnvelopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        source: torch.Tensor,
        y: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        cost_scale: float,
        forward_tol: float,
        forward_iter_max: int,
        allow_tf32: bool,
        use_exp2: bool,
        autotune: bool,
        api: _FlashAPI,
    ) -> torch.Tensor:
        result = api.solve_alternating(
            x,
            y,
            source,
            target,
            eps=float(epsilon),
            n_iters=int(forward_iter_max),
            cost_scale=float(cost_scale),
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
            autotune=bool(autotune),
            threshold=float(forward_tol) * float(epsilon),
            check_every=5,
            return_n_iters=True,
        )
        relative_source, relative_target, iterations = result
        del iterations
        alpha = float(cost_scale) * x.square().sum(1)
        beta = float(cost_scale) * y.square().sum(1)
        f_hat = relative_source - alpha
        g_hat = relative_target - beta
        log_source = torch.log(source)
        log_target = torch.log(target)
        operator = FlashSquaredEuclideanPlanOperator(
            x=x,
            y=y,
            f_hat=f_hat,
            g_hat=g_hat,
            log_source=log_source,
            log_target=log_target,
            epsilon=float(epsilon),
            cost_scale=float(cost_scale),
            api=api,
            allow_tf32=bool(allow_tf32),
            use_exp2=bool(use_exp2),
            autotune=bool(autotune),
        )
        _, normalized_error = operator.marginal_errors(source, target)
        if float(normalized_error) > float(forward_tol):
            raise RuntimeError(
                "flash_sqeuclidean forward did not satisfy the normalized "
                f"marginal tolerance (error={float(normalized_error):.3e}, "
                f"tolerance={float(forward_tol):.3e})"
            )

        raw_source = relative_source + float(epsilon) * log_source
        raw_target = relative_target + float(epsilon) * log_target
        gauge = raw_target[-1]
        grounded_source = raw_source + gauge
        grounded_target = raw_target - gauge
        mass = source.sum()
        value = (
            (source * grounded_source).sum()
            + (target * grounded_target).sum()
            - float(epsilon) * mass
        )
        ctx.save_for_backward(
            x,
            source,
            y,
            target,
            f_hat,
            g_hat,
            log_source,
            log_target,
            grounded_source,
            grounded_target,
        )
        ctx.epsilon = float(epsilon)
        ctx.cost_scale = float(cost_scale)
        ctx.allow_tf32 = bool(allow_tf32)
        ctx.use_exp2 = bool(use_exp2)
        ctx.autotune = bool(autotune)
        ctx.api = api
        return value

    @staticmethod
    def backward(ctx, grad_value: torch.Tensor):
        (
            x,
            source,
            y,
            target,
            f_hat,
            g_hat,
            log_source,
            log_target,
            grounded_source,
            grounded_target,
        ) = ctx.saved_tensors
        operator = FlashSquaredEuclideanPlanOperator(
            x=x,
            y=y,
            f_hat=f_hat,
            g_hat=g_hat,
            log_source=log_source,
            log_target=log_target,
            epsilon=ctx.epsilon,
            cost_scale=ctx.cost_scale,
            api=ctx.api,
            allow_tf32=ctx.allow_tf32,
            use_exp2=ctx.use_exp2,
            autotune=ctx.autotune,
        )
        transported_y = operator.matmat(y)
        transported_x = operator.rmatmat(x)
        coordinate_scale = 2.0 * ctx.cost_scale
        grad_x = coordinate_scale * (source[:, None] * x - transported_y)
        grad_y = coordinate_scale * (target[:, None] * y - transported_x)
        return (
            grad_value * grad_x,
            grad_value * grounded_source,
            grad_value * grad_y,
            grad_value * grounded_target,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _flash_squared_euclidean_envelope_with_api(
    x: torch.Tensor,
    source: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float = 1.0,
    forward_tol: float = 1e-5,
    forward_iter_max: int = 2_000,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
    api: _FlashAPI,
    require_cuda: bool,
    require_float32: bool,
) -> torch.Tensor:
    _validate_inputs(
        x,
        source,
        y,
        target,
        epsilon=epsilon,
        cost_scale=cost_scale,
        forward_tol=forward_tol,
        forward_iter_max=forward_iter_max,
        require_cuda=require_cuda,
        require_float32=require_float32,
    )
    return _FlashSquaredEuclideanEnvelopeFunction.apply(
        x,
        source,
        y,
        target,
        float(epsilon),
        float(cost_scale),
        float(forward_tol),
        int(forward_iter_max),
        bool(allow_tf32),
        bool(use_exp2),
        bool(autotune),
        api,
    )


def flash_squared_euclidean_envelope(
    x: torch.Tensor,
    source: torch.Tensor,
    y: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    cost_scale: float = 1.0,
    forward_tol: float = 1e-5,
    forward_iter_max: int = 2_000,
    allow_tf32: bool = False,
    use_exp2: bool = True,
    autotune: bool = True,
) -> torch.Tensor:
    """Return our raw-entropy EOT value through Flash-Sinkhorn kernels.

    This is an internal, explicit specialization for balanced FP32 CUDA point
    clouds with cost ``cost_scale * ||x_i-y_j||^2``.  It neither changes the
    canonical :class:`EOTLayer` signature nor silently falls back to dense OT.
    """

    _validate_inputs(
        x,
        source,
        y,
        target,
        epsilon=epsilon,
        cost_scale=cost_scale,
        forward_tol=forward_tol,
        forward_iter_max=forward_iter_max,
        require_cuda=True,
        require_float32=True,
    )
    return _flash_squared_euclidean_envelope_with_api(
        x,
        source,
        y,
        target,
        epsilon=epsilon,
        cost_scale=cost_scale,
        forward_tol=forward_tol,
        forward_iter_max=forward_iter_max,
        allow_tf32=allow_tf32,
        use_exp2=use_exp2,
        autotune=autotune,
        api=_load_flash_api(),
        require_cuda=True,
        require_float32=True,
    )
