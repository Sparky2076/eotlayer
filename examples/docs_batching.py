"""Shared and batched marginal examples with public shape checks."""

# docs-start
import torch
from eotlayer import EOTLayer

base_cost = torch.tensor(
    [[0.10, 0.80, 0.55], [0.65, 0.20, 0.45], [0.70, 0.50, 0.15]],
    dtype=torch.float64,
)
cost = torch.stack([base_cost, base_cost.T]).requires_grad_(True)
source = base_cost.new_tensor([0.20, 0.30, 0.50])
target = base_cost.new_tensor([0.40, 0.35, 0.25])
weights = base_cost.new_tensor(
    [[0.2, -0.4, 0.7], [0.5, 0.1, -0.3], [-0.6, 0.8, 0.4]]
)

layer = EOTLayer(epsilon=0.3, mode="implicit", backward_solver="dense")
plans = layer(cost, source, target)
(plans * weights).sum().backward()
print("Plan shape:", tuple(plans.shape))
print("Cost gradient shape:", tuple(cost.grad.shape))
print("Per-item statuses:", [d["backward"]["status"] for d in layer.diagnostics])

batched_source = base_cost.new_tensor([[0.20, 0.30, 0.50], [0.30, 0.40, 0.30]])
other_plans = layer(cost.detach(), batched_source, target)
print("Mixed shared/batched marginals:", tuple(other_plans.shape))
# docs-end

assert plans.shape == (2, 3, 3) and other_plans.shape == (2, 3, 3)
assert torch.isfinite(cost.grad).all()
assert torch.allclose(plans.sum(2), source.expand(2, -1), atol=1e-9, rtol=0)
assert torch.allclose(plans.sum(1), target.expand(2, -1), atol=1e-9, rtol=0)
assert torch.allclose(other_plans.sum(2), batched_source, atol=1e-9, rtol=0)
layer.close()
