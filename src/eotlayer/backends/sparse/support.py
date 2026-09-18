"""Sparse transport-plan proxies and their matrix assemblies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from time import perf_counter
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu
import torch

from ..._core.layout import SchurLayout


_SELECTION_BLOCK_SIZE = 512
_MASS_THRESHOLD_STEPS = 32


def _threshold_step(rows, targets, lower, upper):
    midpoint = lower + (upper - lower) * 0.5
    mass = torch.where(rows > midpoint, rows, 0.0).sum(1, keepdim=True)
    sufficient = mass >= targets
    return torch.where(sufficient, midpoint, lower), torch.where(sufficient, upper, midpoint)


@lru_cache(maxsize=1)
def _cuda_threshold_step():
    # Standard Inductor fusion, not a new selection kernel. Compilation and
    # possible recompilation happen inside the calling constructor and are
    # retained in its cold/warmup timing. No silent eager fallback on error.
    return torch.compile(_threshold_step, fullgraph=True, dynamic=True)


@torch.no_grad()
def _accumulate_threshold_mass_mask(
    block: torch.Tensor,
    threshold: torch.Tensor,
    destination: torch.Tensor,
    *,
    dimension: int,
) -> bool:
    """Use threshold masks and reductions, with no sorting on acceptance.

    Bisect the monotone retained-mass function sum(P * (P > threshold)).
    Coverage alone is insufficient: removing the smallest retained weight
    must also violate the target. Together these tests identify a minimal
    largest-weight prefix. Ambiguous rounding, a split tie, or an unresolved
    search requests the original full-sort block. The fixed search budget
    never relaxes the retained-mass target.
    """

    rows = (block if dimension == 1 else block.t()).contiguous()
    targets = threshold.reshape(-1, 1)
    lower = torch.zeros_like(targets)
    upper = rows.amax(1, keepdim=True)
    step = _cuda_threshold_step() if rows.is_cuda else _threshold_step
    # No host scalar read, sorting, or per-entry index construction in this
    # loop; all row thresholds advance together via elementwise operations.
    for _ in range(_MASS_THRESHOLD_STEPS):
        lower, upper = step(rows, targets, lower, upper)
    mask = rows > lower
    mass = torch.where(mask, rows, 0.0).sum(1, keepdim=True)
    smallest = torch.where(mask, rows, float('inf')).amin(1, keepdim=True)
    roundoff = 8 * rows.shape[1] * torch.finfo(rows.dtype).eps * rows.sum(1, keepdim=True)
    safe = ((mass - targets > roundoff)
            & (targets - (mass - smallest) > roundoff)
            & torch.isfinite(smallest))
    if not bool(safe.all()):
        return False
    destination.logical_or_(mask if dimension == 1 else mask.t())
    return True


@dataclass(frozen=True)
class MassThresholdSupport:
    shape: tuple[int, int]
    edge_rows: torch.Tensor
    edge_columns: torch.Tensor
    edge_weights: torch.Tensor
    dropped_mass_target: float | None
    retained_mass_fraction: float
    retained_edges: int
    retained_fraction: float
    maximum_row_dropped_mass_fraction: float | None = None
    maximum_column_dropped_mass_fraction: float | None = None
    selection_method: str = "mass"
    strength_threshold: float | None = None


@dataclass(frozen=True)
class EffectiveResistanceEstimate:
    """Effective-resistance estimate for all positive edges of one plan."""

    shape: tuple[int, int]
    edge_rows: torch.Tensor
    edge_columns: torch.Tensor
    edge_weights: torch.Tensor
    resistances: torch.Tensor
    method: Literal["exact", "jl"]
    sketch_dimension: int | None
    solve_rhs_count: int
    leverage_sum: float
    elapsed_seconds: float
    required_energy_relative_error: float | None = None
    energy_error_certified: bool = False
    solver_backend: str = "scipy_splu"
    maximum_solver_iterations: int | None = None
    maximum_relative_residual: float | None = None
    condition_number_upper_bound: float | None = None
    energy_error_upper_bound: float | None = None


@dataclass(frozen=True)
class EffectiveResistancePaperParameters:
    """Executable parameters from Spielman--Srivastava (2011).

    ``spectral_epsilon`` is the requested final spectral distortion.  For
    exact resistances, ``sampling_epsilon`` is identical.  With JL resistance
    estimates, Corollary 6 introduces ``approximation_factor`` and the
    sampling theorem is instantiated at
    ``sampling_epsilon = spectral_epsilon / approximation_factor``.

    The paper leaves the universal constant ``C`` and the sufficiently-large
    graph threshold unspecified.  The caller must therefore declare ``C``;
    this object records theorem parameterization, not a finite-run spectral
    certificate.
    """

    vertex_count: int
    spectral_epsilon: float
    theorem_constant: float
    resistance_epsilon: float | None
    approximation_factor: float
    sampling_epsilon: float
    sketch_dimension: int | None
    sample_budget: int
    solver_energy_tolerance: float | None = None


@dataclass(frozen=True)
class EffectiveResistanceSupport:
    """Reweighted edge sample with degrees computed from the sampled graph.

    ``sampling_probabilities`` and ``sample_counts`` index the input edges in
    ``estimate``.  ``edge_weights`` are the reweighted sampled values, not a
    subset of the original plan values.  This object is bound to the plan for
    which it was constructed and is intentionally unsupported by
    :func:`refresh_fixed_mass_support`.
    """

    shape: tuple[int, int]
    edge_rows: torch.Tensor
    edge_columns: torch.Tensor
    edge_weights: torch.Tensor
    source_degrees: torch.Tensor
    target_degrees: torch.Tensor
    estimate: EffectiveResistanceEstimate
    sampling_probabilities: torch.Tensor
    sample_counts: torch.Tensor
    sample_budget: int
    seed: int
    connected_components: int
    retained_edges: int
    retained_fraction: float
    selection_method: str = "effective_resistance"
    construction_protocol: Literal["heuristic", "spielman_srivastava"] = "heuristic"
    paper_parameters: EffectiveResistancePaperParameters | None = None
    spectral_certificate: bool = False

    @property
    def connected(self) -> bool:
        return self.connected_components == 1


SparseSupport = MassThresholdSupport | EffectiveResistanceSupport


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def effective_resistance_paper_parameters(
    *,
    vertex_count: int,
    spectral_epsilon: float,
    theorem_constant: float,
    resistance_method: Literal["exact", "jl"],
    resistance_epsilon: float | None = None,
) -> EffectiveResistancePaperParameters:
    """Instantiate the paper's dimension and sample-count formulae.

    Theorem 1 requires ``1/sqrt(N) < epsilon <= 1`` and samples
    ``ceil(9 C^2 N log(N) / epsilon^2)`` edges.  Lemma 7 uses
    ``ceil(24 log(N) / eta^2)`` JL rows.  For approximate resistances,
    Lemma 9 gives a multiplicative interval ``(1 +/- eta)^2``; choosing
    ``alpha = (1-eta)^-2`` satisfies both hypotheses of Corollary 6.
    """

    vertex_count = _positive_integer(vertex_count, "vertex_count")
    if vertex_count <= 1:
        raise ValueError("vertex_count must exceed one")
    spectral_epsilon = float(spectral_epsilon)
    theorem_constant = float(theorem_constant)
    if not math.isfinite(spectral_epsilon) or not 0.0 < spectral_epsilon <= 1.0:
        raise ValueError("spectral_epsilon must lie in (0, 1]")
    if not math.isfinite(theorem_constant) or theorem_constant <= 0.0:
        raise ValueError("theorem_constant must be finite and positive")
    if resistance_method not in {"exact", "jl"}:
        raise ValueError("resistance_method must be 'exact' or 'jl'")

    if resistance_method == "exact":
        if resistance_epsilon is not None:
            raise ValueError(
                "resistance_epsilon is only defined for JL resistance estimates"
            )
        approximation_factor = 1.0
        sampling_epsilon = spectral_epsilon
        sketch_dimension = None
    else:
        if resistance_epsilon is None:
            raise ValueError(
                "paper-aligned JL estimation requires resistance_epsilon"
            )
        resistance_epsilon = float(resistance_epsilon)
        if (
            not math.isfinite(resistance_epsilon)
            or not 0.0 < resistance_epsilon < 1.0
        ):
            raise ValueError("resistance_epsilon must lie in (0, 1)")
        approximation_factor = 1.0 / (1.0 - resistance_epsilon) ** 2
        sampling_epsilon = spectral_epsilon / approximation_factor
        sketch_dimension = math.ceil(
            24.0 * math.log(vertex_count) / resistance_epsilon**2
        )

    if sampling_epsilon <= 1.0 / math.sqrt(vertex_count):
        raise ValueError(
            "the paper requires sampling_epsilon > 1/sqrt(vertex_count); "
            "increase spectral_epsilon, decrease resistance_epsilon, or use a "
            "larger graph"
        )
    sample_budget = math.ceil(
        9.0
        * theorem_constant**2
        * vertex_count
        * math.log(vertex_count)
        / sampling_epsilon**2
    )
    return EffectiveResistancePaperParameters(
        vertex_count=vertex_count,
        spectral_epsilon=spectral_epsilon,
        theorem_constant=theorem_constant,
        resistance_epsilon=resistance_epsilon,
        approximation_factor=approximation_factor,
        sampling_epsilon=sampling_epsilon,
        sketch_dimension=sketch_dimension,
        sample_budget=sample_budget,
    )


def _grounded_plan_graph(
    plan: torch.Tensor,
    *,
    maximum_input_edges: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    sp.csc_matrix,
]:
    """Return positive plan edges and the last-vertex-grounded Laplacian.

    This is deliberately a CPU reference path.  Enumerating every positive
    entry of a dense entropic plan is not a scalable GPU implementation, so a
    hard edge-count guard prevents accidental use at unsupported sizes.
    """

    _validate_plan(plan, None)
    maximum_input_edges = _positive_integer(
        maximum_input_edges, "maximum_input_edges"
    )
    if plan.device.type != "cpu":
        raise ValueError(
            "effective-resistance reference construction currently requires a CPU plan"
        )
    edges = torch.nonzero(plan > 0, as_tuple=False)
    edge_count = int(edges.shape[0])
    if edge_count > maximum_input_edges:
        raise ValueError(
            "effective-resistance reference construction would enumerate "
            f"{edge_count} edges, above maximum_input_edges={maximum_input_edges}"
        )
    rows = edges[:, 0].detach().cpu().numpy().astype(np.int64, copy=False)
    columns = edges[:, 1].detach().cpu().numpy().astype(np.int64, copy=False)
    weights = (
        plan[edges[:, 0], edges[:, 1]]
        .detach()
        .to(torch.float64)
        .cpu()
        .numpy()
    )
    source_dimension, target_dimension = plan.shape
    vertex_count = source_dimension + target_dimension
    vertices_u = rows
    vertices_v = source_dimension + columns
    adjacency = sp.csr_matrix(
        (
            np.ones(2 * edge_count, dtype=np.float64),
            (
                np.concatenate((vertices_u, vertices_v)),
                np.concatenate((vertices_v, vertices_u)),
            ),
        ),
        shape=(vertex_count, vertex_count),
    )
    component_count, _ = csgraph.connected_components(adjacency, directed=False)
    if component_count != 1:
        raise ValueError(
            "effective-resistance estimation requires a connected positive-edge graph"
        )
    laplacian = sp.coo_matrix(
        (
            np.concatenate((weights, weights, -weights, -weights)),
            (
                np.concatenate((vertices_u, vertices_v, vertices_u, vertices_v)),
                np.concatenate((vertices_u, vertices_v, vertices_v, vertices_u)),
            ),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsc()
    # The final target vertex is the grounded gauge.  Every other vertex keeps
    # its original integer index, which simplifies both exact and JL solves.
    grounded = laplacian[:-1, :-1].tocsc()
    return rows, columns, weights, vertices_u, vertices_v, grounded


def _torch_graph_component_count(
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    *,
    vertex_count: int,
) -> int:
    """Count components without copying a sampled CUDA graph to the CPU."""

    labels = torch.arange(vertex_count, device=edge_u.device, dtype=torch.int64)
    for _ in range(vertex_count):
        updated = labels.clone()
        updated.scatter_reduce_(0, edge_u, labels[edge_v], reduce="amin")
        updated.scatter_reduce_(0, edge_v, labels[edge_u], reduce="amin")
        # Pointer jumping makes paths contract geometrically while preserving
        # the component minimum propagated by the two scatter reductions.
        updated = torch.minimum(updated, updated[updated])
        if torch.equal(updated, labels):
            return int(torch.unique(updated).numel())
        labels = updated
    raise RuntimeError("GPU connected-components iteration did not converge")


def _apply_grounded_plan_laplacian(
    plan: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Apply the last-target-grounded bipartite Laplacian matrix-free."""

    source_dimension, target_dimension = plan.shape
    expected_rows = source_dimension + target_dimension - 1
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError(
            f"vectors must have shape ({expected_rows}, number_of_rhs)"
        )
    source_vectors = vectors[:source_dimension]
    target_vectors = vectors[source_dimension:]
    source_degrees = plan.sum(1)
    target_degrees = plan.sum(0)[:-1]
    active_plan = plan[:, :-1]
    return torch.cat(
        (
            source_degrees[:, None] * source_vectors
            - active_plan @ target_vectors,
            target_degrees[:, None] * target_vectors
            - active_plan.transpose(0, 1) @ source_vectors,
        ),
        dim=0,
    )


@dataclass(frozen=True)
class _TorchLaplacianSolve:
    solution: torch.Tensor
    iteration_counts: torch.Tensor
    maximum_relative_residual: float
    condition_number_upper_bound: float
    energy_error_upper_bound: float


def _solve_grounded_plan_laplacian_pcg(
    plan: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    required_energy_relative_error: float,
    maximum_iterations: int | None,
) -> _TorchLaplacianSolve:
    """Solve grounded Laplacian systems with a rigorous residual gate.

    For a connected weighted graph with ``N`` vertices and grounded
    Laplacian ``L_g``, a path argument gives

    ``lambda_min(L_g) >= w_min / (N-1)^2``.

    Gershgorin gives ``lambda_max(L_g) <= 2 d_max``.  Consequently the
    observed Euclidean relative residual bounds the relative energy error by
    ``sqrt(kappa_bound)``.  This is conservative but makes the paper's Lemma 9
    solve requirement executable rather than replacing it with an unrelated
    residual tolerance.
    """

    if right_hand_side.ndim != 2:
        raise ValueError("right_hand_side must be a matrix of RHS columns")
    required_energy_relative_error = float(required_energy_relative_error)
    if (
        not math.isfinite(required_energy_relative_error)
        or required_energy_relative_error <= 0.0
    ):
        raise ValueError("required_energy_relative_error must be positive")

    positive_weights = plan[plan > 0]
    if positive_weights.numel() == 0:
        raise ValueError("the weighted graph has no positive edges")
    vertex_count = sum(plan.shape)
    grounded_dimension = vertex_count - 1
    minimum_eigenvalue_lower_bound = positive_weights.min() / grounded_dimension**2
    maximum_degree = torch.maximum(plan.sum(1).max(), plan.sum(0).max())
    maximum_eigenvalue_upper_bound = 2.0 * maximum_degree
    condition_bound_tensor = (
        maximum_eigenvalue_upper_bound / minimum_eigenvalue_lower_bound
    )
    condition_bound = float(condition_bound_tensor)
    residual_target = required_energy_relative_error / math.sqrt(condition_bound)
    if residual_target <= 0.0 or not math.isfinite(residual_target):
        raise ValueError("the derived residual target is not finite and positive")

    if maximum_iterations is None:
        maximum_iterations = grounded_dimension
    maximum_iterations = _positive_integer(
        maximum_iterations, "maximum_laplacian_iterations"
    )

    diagonal = torch.cat((plan.sum(1), plan.sum(0)[:-1]))[:, None]
    if bool((diagonal <= 0).any()):
        raise ValueError("grounded Laplacian has a nonpositive diagonal")
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    rhs_norm = torch.linalg.vector_norm(right_hand_side, dim=0)
    nonzero_rhs = rhs_norm > 0
    relative_residual = torch.zeros_like(rhs_norm)
    relative_residual[nonzero_rhs] = 1.0
    active = nonzero_rhs.clone()
    preconditioned = residual / diagonal
    direction = torch.where(active[None, :], preconditioned, 0.0)
    residual_product = (residual * preconditioned).sum(0)
    iteration_counts = torch.zeros(
        right_hand_side.shape[1], dtype=torch.int64, device=plan.device
    )

    for iteration in range(1, maximum_iterations + 1):
        if not bool(active.any()):
            break
        matrix_direction = _apply_grounded_plan_laplacian(plan, direction)
        denominator = (direction * matrix_direction).sum(0)
        invalid = active & ((denominator <= 0) | ~torch.isfinite(denominator))
        if bool(invalid.any()):
            raise RuntimeError(
                "GPU Laplacian PCG lost positive curvature before reaching "
                "the paper-required solve accuracy"
            )
        safe_denominator = torch.where(active, denominator, 1.0)
        step = torch.where(active, residual_product / safe_denominator, 0.0)
        solution = solution + direction * step[None, :]
        residual = residual - matrix_direction * step[None, :]
        relative_residual = torch.where(
            nonzero_rhs,
            torch.linalg.vector_norm(residual, dim=0)
            / torch.where(nonzero_rhs, rhs_norm, 1.0),
            0.0,
        )
        next_active = nonzero_rhs & (relative_residual > residual_target)
        newly_converged = active & ~next_active
        iteration_counts[newly_converged] = iteration
        if not bool(next_active.any()):
            active = next_active
            break
        next_preconditioned = residual / diagonal
        next_product = (residual * next_preconditioned).sum(0)
        safe_product = torch.where(active, residual_product, 1.0)
        beta = torch.where(next_active, next_product / safe_product, 0.0)
        direction = torch.where(
            next_active[None, :],
            next_preconditioned + direction * beta[None, :],
            0.0,
        )
        preconditioned = next_preconditioned
        residual_product = next_product
        active = next_active

    if bool(active.any()):
        worst = float(relative_residual[active].max())
        raise RuntimeError(
            "GPU Laplacian PCG did not meet the paper-required energy-error "
            f"gate in {maximum_iterations} iterations: worst relative residual "
            f"{worst:.3e}, required <= {residual_target:.3e}"
        )
    maximum_relative_residual = (
        float(relative_residual[nonzero_rhs].max())
        if bool(nonzero_rhs.any())
        else 0.0
    )
    energy_error_upper_bound = maximum_relative_residual * math.sqrt(
        condition_bound
    )
    return _TorchLaplacianSolve(
        solution=solution,
        iteration_counts=iteration_counts,
        maximum_relative_residual=maximum_relative_residual,
        condition_number_upper_bound=condition_bound,
        energy_error_upper_bound=energy_error_upper_bound,
    )


@torch.no_grad()
def estimate_effective_resistances_torch_jl(
    plan: torch.Tensor,
    *,
    sketch_dimension: int,
    required_energy_relative_error: float,
    seed: int = 0,
    edge_block_size: int = 4096,
    rhs_block_size: int = 32,
    maximum_input_edges: int = 1_000_000,
    maximum_laplacian_iterations: int | None = None,
) -> EffectiveResistanceEstimate:
    """Implement the paper's JL resistance estimator with torch operators.

    The routine constructs ``Y = Q W^(1/2) B`` in edge blocks, solves the
    grounded systems in RHS blocks, and accumulates
    ``||Z(chi_u-chi_v)||^2`` without materializing either ``Q`` or the full
    embedding.  On CUDA all large tensors and all Laplacian solves remain on
    the GPU.
    """

    _validate_plan(plan, None)
    sketch_dimension = _positive_integer(sketch_dimension, "sketch_dimension")
    edge_block_size = _positive_integer(edge_block_size, "edge_block_size")
    rhs_block_size = _positive_integer(rhs_block_size, "rhs_block_size")
    maximum_input_edges = _positive_integer(
        maximum_input_edges, "maximum_input_edges"
    )
    if plan.dtype not in {torch.float32, torch.float64}:
        raise TypeError("torch JL resistance estimation requires float32 or float64")
    if plan.is_cuda:
        torch.cuda.synchronize(plan.device)
    started = perf_counter()

    flat_plan = plan.reshape(-1)
    positive_indices = torch.nonzero(flat_plan > 0, as_tuple=False).flatten()
    edge_count = int(positive_indices.numel())
    if edge_count > maximum_input_edges:
        raise ValueError(
            "effective-resistance construction would enumerate "
            f"{edge_count} edges, above maximum_input_edges={maximum_input_edges}"
        )
    source_dimension, target_dimension = plan.shape
    vertex_count = source_dimension + target_dimension
    ground_vertex = vertex_count - 1
    edge_rows = torch.div(
        positive_indices, target_dimension, rounding_mode="floor"
    )
    edge_columns = positive_indices.remainder(target_dimension)
    edge_vertices_v = source_dimension + edge_columns
    edge_weights = flat_plan[positive_indices]
    if edge_count == plan.numel():
        component_count = 1
    else:
        component_count = _torch_graph_component_count(
            edge_rows,
            edge_vertices_v,
            vertex_count=vertex_count,
        )
    if component_count != 1:
        raise ValueError(
            "effective-resistance estimation requires a connected positive-edge graph"
        )

    generator = torch.Generator(device=plan.device)
    generator.manual_seed(int(seed))
    resistances = torch.zeros_like(edge_weights)
    maximum_solver_iterations_used = 0
    maximum_relative_residual = 0.0
    maximum_energy_error_upper_bound = 0.0
    condition_number_upper_bound: float | None = None

    for sketch_left in range(0, sketch_dimension, rhs_block_size):
        sketch_right = min(sketch_left + rhs_block_size, sketch_dimension)
        width = sketch_right - sketch_left
        right_hand_side = torch.zeros(
            (ground_vertex, width), dtype=plan.dtype, device=plan.device
        )
        for edge_left in range(0, edge_count, edge_block_size):
            edge_right = min(edge_left + edge_block_size, edge_count)
            local_weights = edge_weights[edge_left:edge_right]
            signs = torch.randint(
                0,
                2,
                (edge_right - edge_left, width),
                generator=generator,
                device=plan.device,
                dtype=torch.int8,
            ).to(dtype=plan.dtype)
            signs.mul_(2.0).sub_(1.0)
            signs.mul_(
                torch.sqrt(local_weights / sketch_dimension)[:, None]
            )
            local_u = edge_rows[edge_left:edge_right]
            local_v = edge_vertices_v[edge_left:edge_right]
            right_hand_side.index_add_(0, local_u, signs)
            active_v = local_v != ground_vertex
            right_hand_side.index_add_(
                0, local_v[active_v], -signs[active_v]
            )

        solve = _solve_grounded_plan_laplacian_pcg(
            plan,
            right_hand_side,
            required_energy_relative_error=required_energy_relative_error,
            maximum_iterations=maximum_laplacian_iterations,
        )
        maximum_solver_iterations_used = max(
            maximum_solver_iterations_used,
            int(solve.iteration_counts.max()),
        )
        maximum_relative_residual = max(
            maximum_relative_residual, solve.maximum_relative_residual
        )
        maximum_energy_error_upper_bound = max(
            maximum_energy_error_upper_bound, solve.energy_error_upper_bound
        )
        condition_number_upper_bound = solve.condition_number_upper_bound

        for edge_left in range(0, edge_count, edge_block_size):
            edge_right = min(edge_left + edge_block_size, edge_count)
            local_u = edge_rows[edge_left:edge_right]
            local_v = edge_vertices_v[edge_left:edge_right]
            difference = solve.solution[local_u].clone()
            active_v = local_v != ground_vertex
            difference[active_v] -= solve.solution[local_v[active_v]]
            resistances[edge_left:edge_right].add_(
                (difference * difference).sum(1)
            )

    if plan.is_cuda:
        torch.cuda.synchronize(plan.device)
    if not bool(torch.isfinite(resistances).all()) or bool((resistances <= 0).any()):
        raise RuntimeError(
            "torch JL resistance estimation produced nonpositive or nonfinite values"
        )
    return EffectiveResistanceEstimate(
        shape=tuple(plan.shape),
        edge_rows=edge_rows,
        edge_columns=edge_columns,
        edge_weights=edge_weights,
        resistances=resistances,
        method="jl",
        sketch_dimension=sketch_dimension,
        solve_rhs_count=sketch_dimension,
        leverage_sum=float((edge_weights * resistances).sum()),
        elapsed_seconds=perf_counter() - started,
        required_energy_relative_error=float(required_energy_relative_error),
        energy_error_certified=(
            maximum_energy_error_upper_bound
            <= float(required_energy_relative_error)
        ),
        solver_backend="torch_jacobi_pcg",
        maximum_solver_iterations=maximum_solver_iterations_used,
        maximum_relative_residual=maximum_relative_residual,
        condition_number_upper_bound=condition_number_upper_bound,
        energy_error_upper_bound=maximum_energy_error_upper_bound,
    )


def estimate_effective_resistances(
    plan: torch.Tensor,
    *,
    method: Literal["exact", "jl"] = "jl",
    sketch_dimension: int = 32,
    seed: int = 0,
    edge_block_size: int = 4096,
    maximum_input_edges: int = 1_000_000,
) -> EffectiveResistanceEstimate:
    """Estimate resistance of every positive edge of a single plan.

    ``exact`` reuses one grounded sparse factorization but solves one right-
    hand side per edge.  ``jl`` follows the Spielman--Srivastava random
    projection construction and solves ``sketch_dimension`` right-hand sides.
    Both are CPU references.  The JL output is an estimator, not a finite-run
    spectral certificate.
    """

    if method not in {"exact", "jl"}:
        raise ValueError("method must be 'exact' or 'jl'")
    edge_block_size = _positive_integer(edge_block_size, "edge_block_size")
    if method == "jl":
        sketch_dimension = _positive_integer(
            sketch_dimension, "sketch_dimension"
        )
    started = perf_counter()
    rows, columns, weights, vertices_u, vertices_v, grounded = (
        _grounded_plan_graph(
            plan, maximum_input_edges=maximum_input_edges
        )
    )
    edge_count = len(weights)
    factor = splu(grounded)
    ground_vertex = grounded.shape[0]

    if method == "exact":
        resistances = np.empty(edge_count, dtype=np.float64)
        for left in range(0, edge_count, edge_block_size):
            right = min(left + edge_block_size, edge_count)
            width = right - left
            rhs = np.zeros((ground_vertex, width), dtype=np.float64)
            local = np.arange(width)
            np.add.at(rhs, (vertices_u[left:right], local), 1.0)
            active_v = vertices_v[left:right] != ground_vertex
            np.add.at(
                rhs,
                (vertices_v[left:right][active_v], local[active_v]),
                -1.0,
            )
            solution = factor.solve(rhs)
            resistances[left:right] = np.einsum(
                "ij,ij->j", rhs, solution
            )
        solve_rhs_count = edge_count
        stored_sketch_dimension = None
    else:
        rng = np.random.default_rng(seed)
        rhs = np.zeros(
            (ground_vertex, sketch_dimension), dtype=np.float64
        )
        for left in range(0, edge_count, edge_block_size):
            right = min(left + edge_block_size, edge_count)
            signs = np.where(
                rng.random((right - left, sketch_dimension)) < 0.5,
                -1.0,
                1.0,
            )
            signs *= np.sqrt(weights[left:right, None] / sketch_dimension)
            np.add.at(rhs, vertices_u[left:right], signs)
            active_v = vertices_v[left:right] != ground_vertex
            np.add.at(rhs, vertices_v[left:right][active_v], -signs[active_v])
        solution = factor.solve(rhs)
        embedding = np.zeros(
            (ground_vertex + 1, sketch_dimension), dtype=np.float64
        )
        embedding[:ground_vertex] = solution
        resistances = np.empty(edge_count, dtype=np.float64)
        for left in range(0, edge_count, edge_block_size):
            right = min(left + edge_block_size, edge_count)
            difference = (
                embedding[vertices_u[left:right]]
                - embedding[vertices_v[left:right]]
            )
            resistances[left:right] = np.einsum(
                "ij,ij->i", difference, difference
            )
        solve_rhs_count = sketch_dimension
        stored_sketch_dimension = sketch_dimension

    if not np.all(np.isfinite(resistances)) or np.any(resistances <= 0):
        raise RuntimeError(
            "effective-resistance estimation produced nonpositive or nonfinite values"
        )
    dtype = plan.dtype
    return EffectiveResistanceEstimate(
        shape=tuple(plan.shape),
        edge_rows=torch.from_numpy(rows.copy()),
        edge_columns=torch.from_numpy(columns.copy()),
        edge_weights=torch.from_numpy(weights.copy()).to(dtype=dtype),
        resistances=torch.from_numpy(resistances).to(dtype=dtype),
        method=method,
        sketch_dimension=stored_sketch_dimension,
        solve_rhs_count=solve_rhs_count,
        leverage_sum=float(weights @ resistances),
        elapsed_seconds=perf_counter() - started,
        energy_error_certified=False,
    )


def _sample_effective_resistance_estimate(
    plan: torch.Tensor,
    estimate: EffectiveResistanceEstimate,
    *,
    sample_budget: int,
    seed: int,
    construction_protocol: Literal["heuristic", "spielman_srivastava"],
    paper_parameters: EffectiveResistancePaperParameters | None,
) -> EffectiveResistanceSupport:
    """Apply the paper's with-replacement sampling and reweighting step."""

    sample_budget = _positive_integer(sample_budget, "sample_budget")
    device = estimate.edge_weights.device
    weights = estimate.edge_weights.to(torch.float64)
    resistances = estimate.resistances.to(torch.float64)
    scores = weights * resistances
    if not bool(torch.isfinite(scores).all()) or bool((scores <= 0).any()):
        raise RuntimeError("effective-resistance sampling scores are invalid")
    probabilities = scores / scores.sum()
    if bool((probabilities <= 0).any()):
        raise RuntimeError("effective-resistance probabilities underflowed")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1)
    sampled_indices = torch.multinomial(
        probabilities,
        sample_budget,
        replacement=True,
        generator=generator,
    )
    counts = torch.bincount(sampled_indices, minlength=len(weights))
    retained = counts > 0
    sampled_weights = (
        weights[retained]
        * counts[retained]
        / (sample_budget * probabilities[retained])
    )
    rows = estimate.edge_rows[retained]
    columns = estimate.edge_columns[retained]
    edge_weights = sampled_weights.to(dtype=plan.dtype)
    source_degrees = torch.zeros(
        plan.shape[0], dtype=plan.dtype, device=device
    )
    target_degrees = torch.zeros(
        plan.shape[1], dtype=plan.dtype, device=device
    )
    source_degrees.index_add_(0, rows, edge_weights)
    target_degrees.index_add_(0, columns, edge_weights)

    vertex_count = sum(plan.shape)
    component_count = _torch_graph_component_count(
        rows,
        plan.shape[0] + columns,
        vertex_count=vertex_count,
    )
    retained_edges = int(edge_weights.numel())
    return EffectiveResistanceSupport(
        shape=estimate.shape,
        edge_rows=rows,
        edge_columns=columns,
        edge_weights=edge_weights,
        source_degrees=source_degrees,
        target_degrees=target_degrees,
        estimate=estimate,
        sampling_probabilities=probabilities.to(dtype=plan.dtype),
        sample_counts=counts,
        sample_budget=sample_budget,
        seed=int(seed),
        connected_components=int(component_count),
        retained_edges=retained_edges,
        retained_fraction=retained_edges / len(weights),
        construction_protocol=construction_protocol,
        paper_parameters=paper_parameters,
        spectral_certificate=False,
    )


def build_heuristic_effective_resistance_support(
    plan: torch.Tensor,
    *,
    sample_budget: int,
    resistance_method: Literal["exact", "jl"] = "jl",
    sketch_dimension: int = 32,
    seed: int = 0,
    edge_block_size: int = 4096,
    maximum_input_edges: int = 1_000_000,
) -> EffectiveResistanceSupport:
    """Build an explicitly heuristic resistance-sampled support.

    The sampling and reweighting formulae agree with ``Sparsify``, but the
    caller chooses ``sample_budget`` and ``sketch_dimension``.  Consequently
    this route never claims the paper's spectral guarantee.
    """

    estimate = estimate_effective_resistances(
        plan,
        method=resistance_method,
        sketch_dimension=sketch_dimension,
        seed=seed,
        edge_block_size=edge_block_size,
        maximum_input_edges=maximum_input_edges,
    )
    return _sample_effective_resistance_estimate(
        plan,
        estimate,
        sample_budget=sample_budget,
        seed=seed,
        construction_protocol="heuristic",
        paper_parameters=None,
    )


def build_effective_resistance_support(
    plan: torch.Tensor,
    *,
    spectral_epsilon: float,
    theorem_constant: float,
    resistance_method: Literal["exact", "jl"] = "exact",
    resistance_epsilon: float | None = None,
    laplacian_solver: Literal["auto", "scipy_splu", "torch_pcg"] = "auto",
    seed: int = 0,
    edge_block_size: int = 4096,
    rhs_block_size: int = 32,
    maximum_input_edges: int = 1_000_000,
    maximum_laplacian_iterations: int | None = None,
) -> EffectiveResistanceSupport:
    """Instantiate Spielman--Srivastava sampling without hidden budgets.

    The sample count and, for JL, projection dimension are derived from the
    paper.  ``theorem_constant`` is mandatory because the source theorem does
    not publish a numerical value for its universal constant ``C``.  The
    returned object remains ``spectral_certificate=False``: the theorem also
    assumes a sufficiently large graph without specifying a finite threshold,
    and SciPy's floating-point solve is not a formal STSolve energy certificate.
    """

    _validate_plan(plan, None)
    parameters = effective_resistance_paper_parameters(
        vertex_count=sum(plan.shape),
        spectral_epsilon=spectral_epsilon,
        theorem_constant=theorem_constant,
        resistance_method=resistance_method,
        resistance_epsilon=resistance_epsilon,
    )
    if laplacian_solver not in {"auto", "scipy_splu", "torch_pcg"}:
        raise ValueError(
            "laplacian_solver must be 'auto', 'scipy_splu', or 'torch_pcg'"
        )
    if resistance_method == "exact" and laplacian_solver == "torch_pcg":
        raise ValueError("torch_pcg is defined only for JL resistance estimation")
    if plan.is_cuda and resistance_method == "exact":
        raise ValueError(
            "exact per-edge resistance estimation is a CPU audit path; use "
            "resistance_method='jl' for the GPU paper path"
        )

    sketch_dimension = parameters.sketch_dimension or 1
    if resistance_method == "jl":
        positive_weights = plan[plan > 0].to(torch.float64)
        if positive_weights.numel() == 0:
            raise ValueError("the weighted graph has no positive edges")
        eta = parameters.resistance_epsilon
        assert eta is not None
        solver_delta = eta / 3.0 * math.sqrt(
            2.0
            * (1.0 - eta)
            * float(positive_weights.min())
            / (
                (1.0 + eta)
                * parameters.vertex_count**3
                * float(positive_weights.max())
            )
        )
        parameters = replace(
            parameters, solver_energy_tolerance=solver_delta
        )
        selected_solver = (
            "torch_pcg" if laplacian_solver == "auto" and plan.is_cuda
            else "scipy_splu" if laplacian_solver == "auto"
            else laplacian_solver
        )
        if selected_solver == "torch_pcg":
            estimate = estimate_effective_resistances_torch_jl(
                plan,
                sketch_dimension=sketch_dimension,
                required_energy_relative_error=solver_delta,
                seed=seed,
                edge_block_size=edge_block_size,
                rhs_block_size=rhs_block_size,
                maximum_input_edges=maximum_input_edges,
                maximum_laplacian_iterations=maximum_laplacian_iterations,
            )
        else:
            if plan.device.type != "cpu":
                raise ValueError("scipy_splu requires a CPU plan")
            estimate = estimate_effective_resistances(
                plan,
                method="jl",
                sketch_dimension=sketch_dimension,
                seed=seed,
                edge_block_size=edge_block_size,
                maximum_input_edges=maximum_input_edges,
            )
            estimate = replace(
                estimate, required_energy_relative_error=solver_delta
            )
    else:
        if laplacian_solver not in {"auto", "scipy_splu"}:
            raise ValueError("exact resistance estimation requires scipy_splu")
        estimate = estimate_effective_resistances(
            plan,
            method="exact",
            sketch_dimension=sketch_dimension,
            seed=seed,
            edge_block_size=edge_block_size,
            maximum_input_edges=maximum_input_edges,
        )
    return _sample_effective_resistance_estimate(
        plan,
        estimate,
        sample_budget=parameters.sample_budget,
        seed=seed,
        construction_protocol="spielman_srivastava",
        paper_parameters=parameters,
    )


def _support_mass_loss(
    plan: torch.Tensor,
    edge_rows: torch.Tensor,
    edge_columns: torch.Tensor,
    edge_weights: torch.Tensor,
) -> tuple[float, float]:
    """Return worst relative mass loss over actual plan rows and columns."""

    retained_rows = torch.zeros(
        plan.shape[0], dtype=plan.dtype, device=plan.device
    )
    retained_columns = torch.zeros(
        plan.shape[1], dtype=plan.dtype, device=plan.device
    )
    retained_rows.index_add_(0, edge_rows, edge_weights)
    retained_columns.index_add_(0, edge_columns, edge_weights)
    row_totals = plan.sum(1)
    column_totals = plan.sum(0)
    row_loss = torch.clamp(row_totals - retained_rows, min=0.0) / row_totals
    column_loss = (
        torch.clamp(column_totals - retained_columns, min=0.0) / column_totals
    )
    return float(row_loss.max()), float(column_loss.max())


def _validate_plan(plan: torch.Tensor, dropped_mass: float | None) -> None:
    if not isinstance(plan, torch.Tensor) or plan.ndim != 2 or plan.numel() == 0:
        raise ValueError("plan must be a nonempty matrix")
    if not plan.is_floating_point():
        raise TypeError("plan must use a floating-point dtype")
    if dropped_mass is not None and not 0.0 <= float(dropped_mass) < 1.0:
        raise ValueError("dropped_mass must lie in [0, 1)")
    if not torch.all(torch.isfinite(plan)) or torch.any(plan < 0):
        raise ValueError("plan must be finite and nonnegative")
    if torch.any(plan.sum(1) <= 0) or torch.any(plan.sum(0) <= 0):
        raise ValueError("every plan row and column must carry positive mass")


def _accumulate_minimal_mass_mask(
    plan: torch.Tensor,
    support: torch.Tensor,
    *,
    dimension: int,
    retained_fraction: float,
    block_size: int,
    selection_method: str = "sort",
) -> None:
    batch_dimension = 1 - dimension
    totals = plan.sum(dim=dimension, keepdim=True)
    for start in range(0, plan.shape[batch_dimension], block_size):
        stop = min(start + block_size, plan.shape[batch_dimension])
        slices = [slice(None), slice(None)]
        slices[batch_dimension] = slice(start, stop)
        block_slice = tuple(slices)
        block = plan[block_slice]
        threshold = retained_fraction * totals[block_slice]
        selected = False
        if selection_method == "mass_threshold" and retained_fraction < 1.0:
            selected = _accumulate_threshold_mass_mask(
                block, threshold, support[block_slice], dimension=dimension
            )
        if not selected:
            # Keep the exact old orientation and block shape, including its
            # backend-specific (unstable) tie ordering. Never claim that
            # sort supplies a cross-device canonical tie break.
            values, indices = torch.sort(block, dim=dimension, descending=True)
            cumulative_before = torch.cumsum(values, dim=dimension)
            cumulative_before.sub_(values)
            keep_sorted = cumulative_before < threshold
            del values, cumulative_before
            block_support = torch.zeros_like(block, dtype=torch.bool)
            block_support.scatter_(dimension, indices, keep_sorted)
            support[block_slice].logical_or_(block_support)
            del indices, keep_sorted, block_support


def build_mass_threshold_support(
    plan: torch.Tensor,
    *,
    dropped_mass: float,
    selection_block_size: int = _SELECTION_BLOCK_SIZE,
    selection_method: str = "sort",
) -> MassThresholdSupport:
    """Retain the union of minimal row-wise and column-wise mass supports.

    ``mass_threshold`` is an opt-in matrix-mask/reduction candidate with
    mass, minimality and rounding checks. ``sort`` remains the production
    default until workload-level validation. Both keep original weights and
    actual plan marginals; the search budget never relaxes dropped_mass.
    """

    _validate_plan(plan, dropped_mass)
    if selection_method not in {"mass_threshold", "sort"}:
        raise ValueError("selection_method must be mass_threshold or sort")
    if selection_block_size <= 0:
        raise ValueError("selection_block_size must be positive")
    support = torch.zeros_like(plan, dtype=torch.bool)
    retained_fraction = 1.0 - float(dropped_mass)
    _accumulate_minimal_mass_mask(
        plan,
        support,
        dimension=1,
        retained_fraction=retained_fraction,
        block_size=selection_block_size,
        selection_method=selection_method,
    )
    _accumulate_minimal_mass_mask(
        plan,
        support,
        dimension=0,
        retained_fraction=retained_fraction,
        block_size=selection_block_size,
        selection_method=selection_method,
    )
    edges = torch.nonzero(support, as_tuple=False)
    edge_rows = edges[:, 0]
    edge_columns = edges[:, 1]
    edge_weights = plan[edge_rows, edge_columns]
    retained_edges = int(edge_weights.numel())
    maximum_row_loss, maximum_column_loss = _support_mass_loss(
        plan, edge_rows, edge_columns, edge_weights
    )
    return MassThresholdSupport(
        shape=(plan.shape[0], plan.shape[1]),
        edge_rows=edge_rows,
        edge_columns=edge_columns,
        edge_weights=edge_weights,
        dropped_mass_target=float(dropped_mass),
        retained_mass_fraction=float(edge_weights.sum() / plan.sum()),
        retained_edges=retained_edges,
        retained_fraction=retained_edges / plan.numel(),
        maximum_row_dropped_mass_fraction=maximum_row_loss,
        maximum_column_dropped_mass_fraction=maximum_column_loss,
    )


def build_relative_strength_support(
    plan: torch.Tensor, *, strength_threshold: float
) -> MassThresholdSupport:
    """Retain the union of strong row/column connections, with original weights.

    Adaptation of classical AMG relative strength to the bipartite plan:
    P_ij >= theta * max_j(P_ij) OR P_ij >= theta * max_i(P_ij).
    Zero edges are excluded. This is NOT a retained-mass guarantee or a
    spectral certificate. Only reductions/comparisons select the support;
    no sort, top-k, iteration, reweighting, or connectivity repair is used.
    """
    if not math.isfinite(float(strength_threshold)) or not 0.0 < strength_threshold <= 1.0:
        raise ValueError("strength_threshold must be finite and lie in (0, 1]")
    _validate_plan(plan, None)
    with torch.no_grad():
        row_threshold = strength_threshold * plan.amax(1, keepdim=True)
        column_threshold = strength_threshold * plan.amax(0, keepdim=True)
        mask = plan >= row_threshold
        mask.logical_or_(plan >= column_threshold)
        mask.logical_and_(plan > 0)
        edges = mask.nonzero(as_tuple=False)
    rows, columns = edges[:, 0], edges[:, 1]
    weights = plan[rows, columns]
    row_loss, column_loss = _support_mass_loss(plan, rows, columns, weights)
    return MassThresholdSupport(
        shape=tuple(plan.shape), edge_rows=rows, edge_columns=columns,
        edge_weights=weights, dropped_mass_target=None,
        retained_mass_fraction=float(weights.sum().detach() / plan.sum().detach()),
        retained_edges=int(weights.numel()), retained_fraction=weights.numel() / plan.numel(),
        maximum_row_dropped_mass_fraction=row_loss,
        maximum_column_dropped_mass_fraction=column_loss,
        selection_method="relative_strength", strength_threshold=float(strength_threshold),
    )


def refresh_fixed_mass_support(
    pattern: MassThresholdSupport, plan: torch.Tensor
) -> MassThresholdSupport:
    """Gather current values on an unchanged row-major edge pattern."""

    if not isinstance(pattern, MassThresholdSupport):
        raise TypeError(
            "refresh_fixed_mass_support only accepts mass/strength supports; "
            "effective-resistance probabilities and weights belong to one plan"
        )
    _validate_plan(plan, pattern.dropped_mass_target)
    if tuple(plan.shape) != pattern.shape:
        raise ValueError("plan and support dimensions do not agree")
    if pattern.edge_rows.device != plan.device or pattern.edge_columns.device != plan.device:
        raise ValueError("plan and support must use the same device")
    edge_weights = plan[pattern.edge_rows, pattern.edge_columns]
    maximum_row_loss, maximum_column_loss = _support_mass_loss(
        plan, pattern.edge_rows, pattern.edge_columns, edge_weights
    )
    return MassThresholdSupport(
        shape=pattern.shape,
        edge_rows=pattern.edge_rows,
        edge_columns=pattern.edge_columns,
        edge_weights=edge_weights,
        dropped_mass_target=pattern.dropped_mass_target,
        retained_mass_fraction=float(edge_weights.sum() / plan.sum()),
        retained_edges=pattern.retained_edges,
        retained_fraction=pattern.retained_fraction,
        maximum_row_dropped_mass_fraction=maximum_row_loss,
        maximum_column_dropped_mass_fraction=maximum_column_loss,
        selection_method=pattern.selection_method,
        strength_threshold=pattern.strength_threshold,
    )


def _validate_normal_inputs(
    support: SparseSupport,
    source: torch.Tensor,
    target: torch.Tensor,
    layout: SchurLayout,
    diagonal_shift: float,
) -> None:
    if not math.isfinite(float(diagonal_shift)) or diagonal_shift < 0:
        raise ValueError("diagonal_shift must be finite and nonnegative")
    if support.shape != (source.numel(), target.numel()):
        raise ValueError("support and marginal dimensions do not agree")
    if layout.target_dimension != target.numel():
        raise ValueError("layout and target dimensions do not agree")
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError("marginals must use the same dtype and device")
    if support.edge_weights.dtype != source.dtype:
        raise TypeError("support weights and marginals must use the same dtype")
    if (
        support.edge_rows.device != source.device
        or support.edge_columns.device != source.device
        or support.edge_weights.device != source.device
    ):
        raise ValueError("support and marginals must use the same device")


def _active_edges(
    support: SparseSupport, layout: SchurLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compact = torch.full(
        (layout.target_dimension,),
        -1,
        dtype=torch.long,
        device=support.edge_columns.device,
    )
    active_indices = torch.tensor(
        layout.active_target_indices,
        dtype=torch.long,
        device=support.edge_columns.device,
    )
    compact[active_indices] = torch.arange(
        active_indices.numel(), dtype=torch.long, device=active_indices.device
    )
    compact_columns = compact[support.edge_columns]
    active = compact_columns >= 0
    return (
        active,
        support.edge_rows[active],
        support.edge_columns[active],
        compact_columns[active],
    )


@dataclass(frozen=True)
class _NormalizedActiveEdgeBlock:
    """One bounded hot-update block after removing grounded edges."""

    input_edge_count: int
    active_offset: int
    source_rows: torch.Tensor
    normalized_values: torch.Tensor


def _iter_normalized_active_edge_blocks(
    support: SparseSupport,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    *,
    layout: SchurLayout,
    block_size: int,
):
    """Yield normalized active edge values without a support-sized temporary."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    compact = torch.full(
        (layout.target_dimension,),
        -1,
        dtype=torch.long,
        device=support.edge_columns.device,
    )
    active_indices = torch.tensor(
        layout.active_target_indices,
        dtype=torch.long,
        device=support.edge_columns.device,
    )
    compact[active_indices] = torch.arange(
        active_indices.numel(), dtype=torch.long, device=active_indices.device
    )
    active_offset = 0
    for start in range(0, support.retained_edges, block_size):
        stop = min(start + block_size, support.retained_edges)
        edge_rows = support.edge_rows[start:stop]
        edge_columns = support.edge_columns[start:stop]
        active = compact[edge_columns] >= 0
        source_rows = edge_rows[active]
        original_columns = edge_columns[active]
        normalized_values = support.edge_weights[start:stop][active]
        normalized_values = normalized_values / torch.sqrt(
            source_marginal[source_rows] * target_marginal[original_columns]
        )
        yield _NormalizedActiveEdgeBlock(
            input_edge_count=stop - start,
            active_offset=active_offset,
            source_rows=source_rows,
            normalized_values=normalized_values,
        )
        active_offset += int(normalized_values.numel())


def build_scaled_sparse_normal_matrix(
    support: MassThresholdSupport,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    *,
    layout: SchurLayout,
    diagonal_shift: float = 0.0,
) -> sp.csc_matrix:
    """Build ``[[I,R_hat],[R_hat.T,I]] + delta I`` on CPU."""

    _validate_normal_inputs(
        support, source_marginal, target_marginal, layout, diagonal_shift
    )
    active, rows, original_columns, columns = _active_edges(support, layout)
    values = support.edge_weights[active] / torch.sqrt(
        source_marginal[rows] * target_marginal[original_columns]
    )
    normalized = sp.coo_matrix(
        (
            values.detach().to(torch.float64).cpu().numpy(),
            (
                rows.detach().cpu().numpy(),
                columns.detach().cpu().numpy(),
            ),
        ),
        shape=(source_marginal.numel(), len(layout.active_target_indices)),
    ).tocsc()
    diagonal = 1.0 + float(diagonal_shift)
    source_identity = diagonal * sp.identity(
        source_marginal.numel(), dtype=np.float64, format="csc"
    )
    target_identity = diagonal * sp.identity(
        len(layout.active_target_indices), dtype=np.float64, format="csc"
    )
    matrix = sp.bmat(
        ((source_identity, normalized), (normalized.T, target_identity)),
        format="csc",
    )
    matrix.sort_indices()
    return matrix


def build_scaled_grounded_laplacian_proxy(
    support: EffectiveResistanceSupport,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    *,
    layout: SchurLayout,
) -> sp.csc_matrix:
    """Build the normalized grounded block from a reweighted graph sample.

    Unlike the mass-threshold proxy, an effective-resistance sample changes
    graph degrees.  Its diagonal blocks are therefore ``D_a_tilde / D_a``
    and ``D_b_tilde / D_b`` rather than identities.  The positive off-
    diagonal sign is congruent to the sampled bipartite Laplacian and matches
    the exact EOT block-system convention.
    """

    if not isinstance(support, EffectiveResistanceSupport):
        raise TypeError(
            "build_scaled_grounded_laplacian_proxy requires an "
            "EffectiveResistanceSupport"
        )
    _validate_normal_inputs(
        support, source_marginal, target_marginal, layout, 0.0
    )
    if source_marginal.device.type != "cpu":
        raise ValueError(
            "effective-resistance proxy assembly currently requires CPU tensors"
        )
    if not support.connected:
        raise ValueError(
            "the sampled effective-resistance graph is disconnected and cannot "
            "define an SPD grounded proxy"
        )
    if (
        support.source_degrees.dtype != source_marginal.dtype
        or support.target_degrees.dtype != target_marginal.dtype
        or support.source_degrees.device != source_marginal.device
        or support.target_degrees.device != target_marginal.device
    ):
        raise ValueError("sampled degrees must match marginal dtype and device")
    if torch.any(support.source_degrees <= 0) or torch.any(
        support.target_degrees <= 0
    ):
        raise ValueError("a connected sampled graph must have positive degrees")

    active, rows, original_columns, columns = _active_edges(support, layout)
    normalized_values = support.edge_weights[active] / torch.sqrt(
        source_marginal[rows] * target_marginal[original_columns]
    )
    normalized = sp.coo_matrix(
        (
            normalized_values.detach().to(torch.float64).numpy(),
            (rows.detach().numpy(), columns.detach().numpy()),
        ),
        shape=(source_marginal.numel(), len(layout.active_target_indices)),
    ).tocsc()
    active_indices = torch.tensor(
        layout.active_target_indices, dtype=torch.long
    )
    source_diagonal = (
        support.source_degrees / source_marginal
    ).to(torch.float64).numpy()
    target_diagonal = (
        support.target_degrees[active_indices]
        / target_marginal[active_indices]
    ).to(torch.float64).numpy()
    matrix = sp.bmat(
        (
            (sp.diags(source_diagonal, format="csc"), normalized),
            (normalized.T, sp.diags(target_diagonal, format="csc")),
        ),
        format="csc",
    )
    matrix.sort_indices()
    return matrix


def effective_resistance_support_to(
    support: EffectiveResistanceSupport,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> EffectiveResistanceSupport:
    """Move the sampled graph tensors without changing its paper sample.

    A CPU reference estimate may remain on CPU as audit evidence.  Tensors
    required to assemble and apply the sampled preconditioner move to the
    requested execution device.  A torch/GPU estimate already on that device
    is left unchanged by the backend and does not call this helper.
    """

    if not isinstance(support, EffectiveResistanceSupport):
        raise TypeError("support must be an EffectiveResistanceSupport")
    return replace(
        support,
        edge_rows=support.edge_rows.to(device=device),
        edge_columns=support.edge_columns.to(device=device),
        edge_weights=support.edge_weights.to(device=device, dtype=dtype),
        source_degrees=support.source_degrees.to(device=device, dtype=dtype),
        target_degrees=support.target_degrees.to(device=device, dtype=dtype),
    )


def build_scaled_cuda_upper_grounded_laplacian_proxy(
    support: EffectiveResistanceSupport,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    *,
    layout: SchurLayout,
) -> torch.Tensor:
    """Build the paper-sampled grounded proxy as an upper CUDA CSR matrix.

    The sampled degrees form the diagonal blocks and the reweighted sampled
    edges form the off-diagonal block.  A positive off-diagonal sign is used
    because it is congruent to the sampled bipartite Laplacian through a sign
    flip on target variables and gives the same source Schur complement.
    """

    if not isinstance(support, EffectiveResistanceSupport):
        raise TypeError(
            "effective-resistance CUDA assembly requires an "
            "EffectiveResistanceSupport"
        )
    _validate_normal_inputs(
        support, source_marginal, target_marginal, layout, 0.0
    )
    if source_marginal.device.type != "cuda":
        raise ValueError("effective-resistance CUDA assembly requires CUDA tensors")
    if not support.connected:
        raise ValueError(
            "the sampled effective-resistance graph is disconnected and cannot "
            "define an SPD grounded proxy"
        )
    if (
        support.source_degrees.dtype != source_marginal.dtype
        or support.target_degrees.dtype != target_marginal.dtype
        or support.source_degrees.device != source_marginal.device
        or support.target_degrees.device != target_marginal.device
    ):
        raise ValueError("sampled degrees must match marginal dtype and device")
    if torch.any(support.source_degrees <= 0) or torch.any(
        support.target_degrees <= 0
    ):
        raise ValueError("a connected sampled graph must have positive degrees")

    active, rows, original_columns, columns = _active_edges(support, layout)
    values = support.edge_weights[active] / torch.sqrt(
        source_marginal[rows] * target_marginal[original_columns]
    )
    active_indices = torch.tensor(
        layout.active_target_indices,
        dtype=torch.long,
        device=source_marginal.device,
    )
    diagonal_values = torch.cat(
        (
            support.source_degrees / source_marginal,
            support.target_degrees[active_indices]
            / target_marginal[active_indices],
        )
    )
    source_dimension = source_marginal.numel()
    dimension = source_dimension + active_indices.numel()
    edge_count = int(values.numel())
    if dimension + edge_count > torch.iinfo(torch.int32).max:
        raise OverflowError("CUDA CSR exceeds int32 capacity")

    row_counts = torch.ones(
        dimension, dtype=torch.int32, device=source_marginal.device
    )
    row_counts[:source_dimension].add_(
        torch.bincount(rows, minlength=source_dimension).to(torch.int32)
    )
    crow = torch.empty(
        dimension + 1, dtype=torch.int32, device=source_marginal.device
    )
    crow[0] = 0
    torch.cumsum(row_counts, dim=0, out=crow[1:])
    col = torch.empty(
        dimension + edge_count,
        dtype=torch.int32,
        device=source_marginal.device,
    )
    matrix_values = torch.empty(
        dimension + edge_count,
        dtype=source_marginal.dtype,
        device=source_marginal.device,
    )
    diagonal_positions = crow[:-1].to(torch.long)
    col.index_copy_(
        0,
        diagonal_positions,
        torch.arange(
            dimension, dtype=torch.int32, device=source_marginal.device
        ),
    )
    matrix_values.index_copy_(0, diagonal_positions, diagonal_values)
    edge_positions = torch.arange(
        edge_count, dtype=torch.long, device=rows.device
    )
    edge_positions.add_(rows).add_(1)
    col.index_copy_(
        0, edge_positions, (source_dimension + columns).to(torch.int32)
    )
    matrix_values.index_copy_(0, edge_positions, values)
    return torch.sparse_csr_tensor(
        crow,
        col,
        matrix_values,
        size=(dimension, dimension),
        dtype=source_marginal.dtype,
        device=source_marginal.device,
        check_invariants=False,
    )


def build_scaled_cuda_upper_normal_matrix(
    support: MassThresholdSupport,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    *,
    layout: SchurLayout,
    diagonal_shift: float = 0.0,
) -> torch.Tensor:
    """Build the fixed upper-triangular CUDA CSR consumed by cuDSS."""

    _validate_normal_inputs(
        support, source_marginal, target_marginal, layout, diagonal_shift
    )
    if source_marginal.device.type != "cuda":
        raise ValueError("CUDA normal assembly requires CUDA tensors")
    active, rows, original_columns, columns = _active_edges(support, layout)
    values = support.edge_weights[active] / torch.sqrt(
        source_marginal[rows] * target_marginal[original_columns]
    )
    source_dimension = source_marginal.numel()
    dimension = source_dimension + len(layout.active_target_indices)
    edge_count = int(values.numel())
    if dimension + edge_count > torch.iinfo(torch.int32).max:
        raise OverflowError("CUDA CSR exceeds int32 capacity")
    row_counts = torch.ones(dimension, dtype=torch.int32, device=source_marginal.device)
    row_counts[:source_dimension].add_(
        torch.bincount(rows, minlength=source_dimension).to(torch.int32)
    )
    crow = torch.empty(dimension + 1, dtype=torch.int32, device=source_marginal.device)
    crow[0] = 0
    torch.cumsum(row_counts, dim=0, out=crow[1:])
    col = torch.empty(dimension + edge_count, dtype=torch.int32, device=source_marginal.device)
    matrix_values = torch.full(
        (dimension + edge_count,),
        1.0 + diagonal_shift,
        dtype=source_marginal.dtype,
        device=source_marginal.device,
    )
    diagonal_positions = crow[:-1].to(torch.long)
    col.index_copy_(
        0,
        diagonal_positions,
        torch.arange(dimension, dtype=torch.int32, device=source_marginal.device),
    )
    edge_positions = torch.arange(edge_count, dtype=torch.long, device=rows.device)
    edge_positions.add_(rows).add_(1)
    col.index_copy_(0, edge_positions, (source_dimension + columns).to(torch.int32))
    matrix_values.index_copy_(0, edge_positions, values)
    return torch.sparse_csr_tensor(
        crow,
        col,
        matrix_values,
        size=(dimension, dimension),
        dtype=source_marginal.dtype,
        device=source_marginal.device,
        check_invariants=False,
    )


def update_scaled_cuda_upper_normal_values_(
    matrix: torch.Tensor,
    support: MassThresholdSupport,
    source_marginal: torch.Tensor,
    target_marginal: torch.Tensor,
    *,
    layout: SchurLayout,
    diagonal_shift: float = 0.0,
    update_block_size: int = 262_144,
) -> None:
    """Update only values of an existing upper CUDA CSR pattern."""

    _validate_normal_inputs(
        support, source_marginal, target_marginal, layout, diagonal_shift
    )
    if update_block_size <= 0:
        raise ValueError("update_block_size must be positive")
    source_dimension = source_marginal.numel()
    dimension = source_dimension + len(layout.active_target_indices)
    if (
        not isinstance(matrix, torch.Tensor)
        or matrix.layout != torch.sparse_csr
        or matrix.shape != (dimension, dimension)
        or matrix.device.type != "cuda"
        or matrix.device != source_marginal.device
        or matrix.dtype != source_marginal.dtype
    ):
        raise ValueError("fixed normal update requires a matching CUDA CSR")
    matrix_values = matrix.values()
    matrix_values.fill_(1.0 + diagonal_shift)
    active_edge_count = 0
    for block in _iter_normalized_active_edge_blocks(
        support,
        source_marginal,
        target_marginal,
        layout=layout,
        block_size=update_block_size,
    ):
        count = int(block.normalized_values.numel())
        if count:
            edge_positions = torch.arange(
                block.active_offset,
                block.active_offset + count,
                dtype=torch.long,
                device=block.source_rows.device,
            )
            edge_positions.add_(block.source_rows).add_(1)
            matrix_values.index_copy_(
                0, edge_positions, block.normalized_values
            )
            active_edge_count += count
    if matrix_values.numel() != dimension + active_edge_count:
        raise ValueError("support and fixed CUDA CSR pattern do not agree")
