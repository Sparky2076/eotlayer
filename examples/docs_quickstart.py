"""A deterministic plan VJP example used by the documentation."""

# docs-start
import torch
from eotlayer import EOTLayer

cost = torch.tensor(
    [[0.10, 0.80, 0.55], [0.65, 0.20, 0.45], [0.70, 0.50, 0.15]],
    dtype=torch.float64,
    requires_grad=True,
)
source = torch.tensor([0.20, 0.30, 0.50], dtype=cost.dtype)
target = torch.tensor([0.40, 0.35, 0.25], dtype=cost.dtype)

layer = EOTLayer(
    epsilon=0.3,
    mode="implicit",
    forward_tol=1e-10,
    backward_solver="jacobi_w",
    backward_tol=1e-8,
)
plan = layer(cost, source, target)
weights = cost.new_tensor(
    [[0.2, -0.4, 0.7], [0.5, 0.1, -0.3], [-0.6, 0.8, 0.4]]
)
loss = (plan * weights).sum()
loss.backward()

print("Transport plan:", plan.detach())
print("Cost gradient:", cost.grad)
print("Backward:", layer.diagnostics[0]["backward"])
# docs-end

assert torch.allclose(plan.sum(1), source, atol=1e-9, rtol=0)
assert torch.allclose(plan.sum(0), target, atol=1e-9, rtol=0)
assert cost.grad is not None and torch.isfinite(cost.grad).all()
assert cost.grad.norm() > 1e-6
assert layer.diagnostics[0]["forward"]["status"] == "converged"
assert layer.diagnostics[0]["backward"]["status"] == "converged"
layer.close()
