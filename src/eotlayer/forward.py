"""Converged log-domain forward solves for supported EOT constraints."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ._core.constraints import (
    BalancedConstraints,
    SlackMatchingConstraints,
    constraint_vectors,
)
from ._core.problem import EOTProblem


@dataclass(frozen=True)
class ForwardConfig:
    """Stopping and optional epsilon-continuation policy.

    ``epsilon_scaling_start=None`` keeps the fixed-epsilon algorithm.  A
    positive start value enables geometric continuation to the problem's
    target epsilon.  Both values use the same physical units.
    """

    tolerance: float = 1e-10
    maximum_iterations: int = 10_000
    fixed_iterations: int | None = None
    check_interval: int = 10
    collect_screening_features: bool = False
    screening_block_rows: int = 128
    epsilon_scaling_start: float | None = None
    epsilon_scaling_decay: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.tolerance)) or self.tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            self.maximum_iterations <= 0
            or self.check_interval <= 0
            or self.screening_block_rows <= 0
        ):
            raise ValueError("iteration controls must be positive")
        if self.fixed_iterations is not None and self.fixed_iterations <= 0:
            raise ValueError("fixed_iterations must be positive")
        if self.epsilon_scaling_start is not None and (
            not math.isfinite(float(self.epsilon_scaling_start))
            or self.epsilon_scaling_start <= 0
        ):
            raise ValueError("epsilon_scaling_start must be finite and positive")
        if (
            not math.isfinite(float(self.epsilon_scaling_decay))
            or not 0.0 < self.epsilon_scaling_decay < 1.0
        ):
            raise ValueError("epsilon_scaling_decay must lie strictly between 0 and 1")
        if not isinstance(self.collect_screening_features, bool):
            raise ValueError("collect_screening_features must be boolean")


@dataclass(frozen=True)
class PlanScreeningFeatures:
    """Bounded reductions collected from a final materialized plan.

    These features screen backend eligibility; they are not an accuracy oracle
    and never use a Dense adjoint solution.
    """

    shape: tuple[int, int]
    sinkhorn_iterations: int
    sinkhorn_residual_contraction: float | None
    effective_support_density: float
    exact_zero_fraction: float
    minimum_positive_entry: float
    maximum_entry: float
    dynamic_range: float
    estimated_dense_peak_bytes: int


@dataclass(frozen=True)
class TransportState:
    """Detached state of one forward EOT solve."""

    problem: EOTProblem
    plan: torch.Tensor
    iterations: int
    absolute_marginal_error: float
    normalized_marginal_error: float
    converged: bool
    log_left: torch.Tensor
    log_right: torch.Tensor
    source_slack: torch.Tensor | None
    target_slack: torch.Tensor | None
    screening_features: PlanScreeningFeatures | None

    @property
    def cost(self) -> torch.Tensor:
        return self.problem.cost

    @property
    def constraints(self):
        return self.problem.constraints

    @property
    def epsilon(self) -> float:
        return self.problem.epsilon

    @property
    def marginal_error(self) -> float:
        """Backward-compatible name for the normalized marginal error."""

        return self.normalized_marginal_error


def _validate_problem(problem: EOTProblem) -> None:
    # Revalidate even when a caller intentionally constructed a deferred-check
    # problem.  Solver entry points are never allowed to consume invalid data.
    EOTProblem(problem.cost, problem.constraints, problem.epsilon)


def _validate_initial_scalings(
    source: torch.Tensor,
    target: torch.Tensor,
    initial_log_left: torch.Tensor | None,
    initial_log_right: torch.Tensor | None,
) -> None:
    if (initial_log_left is None) != (initial_log_right is None):
        raise ValueError("both initial log scalings must be provided together")
    if initial_log_left is None:
        return
    assert initial_log_right is not None
    if initial_log_left.shape != source.shape or initial_log_right.shape != target.shape:
        raise ValueError("initial log scalings have the wrong shape")
    if (
        initial_log_left.dtype != source.dtype
        or initial_log_right.dtype != target.dtype
        or initial_log_left.device != source.device
        or initial_log_right.device != target.device
        or not torch.all(torch.isfinite(initial_log_left))
        or not torch.all(torch.isfinite(initial_log_right))
    ):
        raise ValueError("initial log scalings must be finite and match constraints")


def _should_check(iteration: int, limit: int, config: ForwardConfig) -> bool:
    return iteration == limit or (
        config.fixed_iterations is None and iteration % config.check_interval == 0
    )


def _epsilon_for_iteration(
    target_epsilon: float,
    config: ForwardConfig,
    iteration: int,
    limit: int,
) -> float:
    """Return the continuation value while guaranteeing a target-epsilon output."""

    start = config.epsilon_scaling_start
    if start is None or start == target_epsilon:
        return target_epsilon
    if iteration == limit:
        return target_epsilon
    return max(
        target_epsilon,
        float(start) * float(config.epsilon_scaling_decay) ** (iteration - 1),
    )


def _rescale_log_scalings(
    log_left: torch.Tensor,
    log_right: torch.Tensor,
    previous_epsilon: float,
    current_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve dual potentials when changing the continuation epsilon."""

    if current_epsilon == previous_epsilon:
        return log_left, log_right
    factor = previous_epsilon / current_epsilon
    return factor * log_left, factor * log_right


def _errors(
    row_residual: torch.Tensor,
    column_residual: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    row_l1 = torch.linalg.vector_norm(row_residual, ord=1)
    column_l1 = torch.linalg.vector_norm(column_residual, ord=1)
    absolute = float(torch.maximum(row_l1, column_l1))
    row_normalized = row_l1 / source.sum()
    column_normalized = column_l1 / target.sum()
    normalized = float(torch.maximum(row_normalized, column_normalized))
    return absolute, normalized


def _detached_problem(problem: EOTProblem) -> EOTProblem:
    constraints = problem.constraints
    if isinstance(constraints, BalancedConstraints):
        detached_constraints = BalancedConstraints(
            constraints.source.detach(), constraints.target.detach()
        )
    elif isinstance(constraints, SlackMatchingConstraints):
        detached_constraints = SlackMatchingConstraints(
            constraints.source_total.detach(),
            constraints.target_total.detach(),
            constraints.slack_logit,
        )
    else:  # pragma: no cover - protected by EOTProblem validation
        raise TypeError(f"unsupported constraint type: {type(constraints)!r}")
    return EOTProblem(problem.cost.detach(), detached_constraints, problem.epsilon)


def collect_plan_screening_features(
    plan: torch.Tensor,
    *,
    iterations: int,
    residual_contraction: float | None,
    block_rows: int,
) -> PlanScreeningFeatures:
    """Collect concentration and memory features without support sorting."""

    rows, columns = plan.shape
    column_mass = torch.zeros(columns, dtype=plan.dtype, device=plan.device)
    column_square_mass = torch.zeros_like(column_mass)
    row_effective_sum = plan.new_zeros(())
    zero_count = torch.zeros((), dtype=torch.int64, device=plan.device)
    minimum_positive = plan.new_full((), float("inf"))
    maximum = plan.new_zeros(())
    tiny = torch.finfo(plan.dtype).tiny
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        block = plan[start:stop]
        square = block.square()
        row_mass = block.sum(1)
        row_square_mass = square.sum(1)
        row_effective_sum = row_effective_sum + torch.sum(
            row_mass.square() / torch.clamp(row_square_mass, min=tiny)
        )
        column_mass.add_(block.sum(0))
        column_square_mass.add_(square.sum(0))
        zero_count.add_(torch.count_nonzero(block == 0))
        maximum = torch.maximum(maximum, torch.max(block))
        minimum_positive = torch.minimum(
            minimum_positive,
            torch.min(
                torch.where(
                    block > 0,
                    block,
                    torch.full_like(block, float("inf")),
                )
            ),
        )

    column_effective = column_mass.square() / torch.clamp(
        column_square_mass, min=tiny
    )
    summary = torch.stack(
        (
            row_effective_sum,
            torch.sum(column_effective),
            zero_count.to(plan.dtype),
            minimum_positive,
            maximum,
        )
    ).cpu()
    row_density = float(summary[0]) / float(rows * columns)
    column_density = float(summary[1]) / float(rows * columns)
    effective_density = min(1.0, max(row_density, column_density))
    zero_count_value = int(summary[2])
    minimum_positive_value = float(summary[3])
    maximum_value = float(summary[4])
    if not math.isfinite(minimum_positive_value):
        minimum_positive_value = 0.0
    dynamic_range = (
        maximum_value / minimum_positive_value
        if minimum_positive_value > 0.0
        else float("inf")
    )
    dense_matrix_bytes = rows * rows * plan.element_size()
    estimated_dense_peak_bytes = (
        4 * dense_matrix_bytes + 8 * rows * plan.element_size()
    )
    return PlanScreeningFeatures(
        shape=(rows, columns),
        sinkhorn_iterations=int(iterations),
        sinkhorn_residual_contraction=residual_contraction,
        effective_support_density=effective_density,
        exact_zero_fraction=zero_count_value / plan.numel(),
        minimum_positive_entry=minimum_positive_value,
        maximum_entry=maximum_value,
        dynamic_range=dynamic_range,
        estimated_dense_peak_bytes=estimated_dense_peak_bytes,
    )


def _balanced_forward(
    problem: EOTProblem,
    config: ForwardConfig,
    initial_log_left: torch.Tensor | None,
    initial_log_right: torch.Tensor | None,
) -> TransportState:
    constraints = problem.constraints
    assert isinstance(constraints, BalancedConstraints)
    source, target = constraints.source, constraints.target
    _validate_initial_scalings(source, target, initial_log_left, initial_log_right)
    log_left = torch.zeros_like(source) if initial_log_left is None else initial_log_left.clone()
    log_right = torch.zeros_like(target) if initial_log_right is None else initial_log_right.clone()
    limit = int(config.fixed_iterations or config.maximum_iterations)
    target_epsilon = float(problem.epsilon)
    previous_epsilon = target_epsilon
    log_kernel = -problem.cost / target_epsilon
    plan = torch.empty_like(problem.cost)
    absolute_error = float("inf")
    normalized_error = float("inf")
    converged = False
    previous_checked_error: float | None = None
    previous_checked_iteration: int | None = None
    residual_contraction: float | None = None

    for iteration in range(1, limit + 1):
        current_epsilon = _epsilon_for_iteration(
            target_epsilon, config, iteration, limit
        )
        if current_epsilon != previous_epsilon:
            log_left, log_right = _rescale_log_scalings(
                log_left, log_right, previous_epsilon, current_epsilon
            )
            log_kernel = -problem.cost / current_epsilon
            previous_epsilon = current_epsilon
        log_left = torch.log(source) - torch.logsumexp(
            log_kernel + log_right[None, :], dim=1
        )
        log_right = torch.log(target) - torch.logsumexp(
            log_kernel + log_left[:, None], dim=0
        )
        if current_epsilon == target_epsilon and _should_check(
            iteration, limit, config
        ):
            plan = torch.exp(log_left[:, None] + log_kernel + log_right[None, :])
            absolute_error, normalized_error = _errors(
                plan.sum(1) - source, plan.sum(0) - target, source, target
            )
            if (
                previous_checked_error is not None
                and previous_checked_iteration is not None
                and previous_checked_error > 0.0
                and normalized_error > 0.0
            ):
                elapsed = iteration - previous_checked_iteration
                residual_contraction = (
                    normalized_error / previous_checked_error
                ) ** (1.0 / elapsed)
            previous_checked_error = normalized_error
            previous_checked_iteration = iteration
            converged = normalized_error <= config.tolerance
            if config.fixed_iterations is None and converged:
                break

    screening = (
        collect_plan_screening_features(
            plan,
            iterations=iteration,
            residual_contraction=residual_contraction,
            block_rows=config.screening_block_rows,
        )
        if config.collect_screening_features
        else None
    )

    return TransportState(
        problem=_detached_problem(problem),
        plan=plan.detach(),
        iterations=iteration,
        absolute_marginal_error=absolute_error,
        normalized_marginal_error=normalized_error,
        converged=converged,
        log_left=log_left.detach(),
        log_right=log_right.detach(),
        source_slack=None,
        target_slack=None,
        screening_features=screening,
    )


def _slack_forward(
    problem: EOTProblem,
    config: ForwardConfig,
    initial_log_left: torch.Tensor | None,
    initial_log_right: torch.Tensor | None,
) -> TransportState:
    constraints = problem.constraints
    assert isinstance(constraints, SlackMatchingConstraints)
    source, target = constraints.source_total, constraints.target_total
    _validate_initial_scalings(source, target, initial_log_left, initial_log_right)
    slack_logit = problem.cost.new_tensor(constraints.slack_logit)
    log_left = torch.zeros_like(source) if initial_log_left is None else initial_log_left.clone()
    log_right = torch.zeros_like(target) if initial_log_right is None else initial_log_right.clone()
    limit = int(config.fixed_iterations or config.maximum_iterations)
    target_epsilon = float(problem.epsilon)
    previous_epsilon = target_epsilon
    log_kernel = -problem.cost / target_epsilon
    plan = torch.empty_like(problem.cost)
    source_slack = torch.empty_like(source)
    target_slack = torch.empty_like(target)
    absolute_error = float("inf")
    normalized_error = float("inf")
    converged = False
    previous_checked_error: float | None = None
    previous_checked_iteration: int | None = None
    residual_contraction: float | None = None

    for iteration in range(1, limit + 1):
        current_epsilon = _epsilon_for_iteration(
            target_epsilon, config, iteration, limit
        )
        if current_epsilon != previous_epsilon:
            log_left, log_right = _rescale_log_scalings(
                log_left, log_right, previous_epsilon, current_epsilon
            )
            log_kernel = -problem.cost / current_epsilon
            previous_epsilon = current_epsilon
        row_terms = torch.cat(
            (
                log_kernel + log_right[None, :],
                slack_logit.expand(source.numel(), 1),
            ),
            dim=1,
        )
        log_left = torch.log(source) - torch.logsumexp(row_terms, dim=1)
        column_terms = torch.cat(
            (
                log_kernel + log_left[:, None],
                slack_logit.expand(1, target.numel()),
            ),
            dim=0,
        )
        log_right = torch.log(target) - torch.logsumexp(column_terms, dim=0)
        if current_epsilon == target_epsilon and _should_check(
            iteration, limit, config
        ):
            plan = torch.exp(log_left[:, None] + log_kernel + log_right[None, :])
            source_slack = torch.exp(slack_logit + log_left)
            target_slack = torch.exp(slack_logit + log_right)
            absolute_error, normalized_error = _errors(
                plan.sum(1) + source_slack - source,
                plan.sum(0) + target_slack - target,
                source,
                target,
            )
            if (
                previous_checked_error is not None
                and previous_checked_iteration is not None
                and previous_checked_error > 0.0
                and normalized_error > 0.0
            ):
                elapsed = iteration - previous_checked_iteration
                residual_contraction = (
                    normalized_error / previous_checked_error
                ) ** (1.0 / elapsed)
            previous_checked_error = normalized_error
            previous_checked_iteration = iteration
            converged = normalized_error <= config.tolerance
            if config.fixed_iterations is None and converged:
                break

    screening = (
        collect_plan_screening_features(
            plan,
            iterations=iteration,
            residual_contraction=residual_contraction,
            block_rows=config.screening_block_rows,
        )
        if config.collect_screening_features
        else None
    )

    return TransportState(
        problem=_detached_problem(problem),
        plan=plan.detach(),
        iterations=iteration,
        absolute_marginal_error=absolute_error,
        normalized_marginal_error=normalized_error,
        converged=converged,
        log_left=log_left.detach(),
        log_right=log_right.detach(),
        source_slack=source_slack.detach(),
        target_slack=target_slack.detach(),
        screening_features=screening,
    )


def transport_forward(
    problem: EOTProblem,
    *,
    config: ForwardConfig | None = None,
    initial_log_left: torch.Tensor | None = None,
    initial_log_right: torch.Tensor | None = None,
) -> TransportState:
    """Solve one EOT problem without retaining an autograd graph."""

    if not isinstance(problem, EOTProblem):
        raise TypeError("transport_forward requires an EOTProblem")
    _validate_problem(problem)
    config = config or ForwardConfig()
    if (
        config.epsilon_scaling_start is not None
        and config.epsilon_scaling_start < problem.epsilon
    ):
        raise ValueError(
            "epsilon_scaling_start must be greater than or equal to the target epsilon"
        )
    with torch.no_grad():
        if isinstance(problem.constraints, BalancedConstraints):
            return _balanced_forward(
                problem, config, initial_log_left, initial_log_right
            )
        if isinstance(problem.constraints, SlackMatchingConstraints):
            return _slack_forward(problem, config, initial_log_left, initial_log_right)
    raise TypeError(f"unsupported constraint type: {type(problem.constraints)!r}")
