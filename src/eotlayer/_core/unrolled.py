"""Finite-iteration, graph-retaining Sinkhorn for standard balanced EOT."""

from __future__ import annotations

import torch

from .constraints import BalancedConstraints
from .problem import EOTProblem


def unrolled_transport_plan(
    cost: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    iterations: int,
) -> torch.Tensor:
    """Return a balanced EOT plan after exactly ``iterations`` updates.

    Unlike :func:`eotlayer.forward.transport_forward`, this function must not
    detach tensors or enter ``torch.no_grad``: ordinary PyTorch autograd owns
    the complete finite-iteration backward pass.
    """

    EOTProblem(cost, BalancedConstraints(source, target), epsilon)
    log_kernel = -cost / float(epsilon)
    log_left = torch.zeros_like(source)
    log_right = torch.zeros_like(target)
    for _ in range(iterations):
        log_left = torch.log(source) - torch.logsumexp(
            log_kernel + log_right[None, :], dim=1
        )
        log_right = torch.log(target) - torch.logsumexp(
            log_kernel + log_left[:, None], dim=0
        )
    return torch.exp(log_left[:, None] + log_kernel + log_right[None, :])
