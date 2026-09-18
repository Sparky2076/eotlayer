"""Check the cost-gradient identity for the regularized EOT value."""

# docs-start
import torch
from eotlayer import EOTLayer

cost = torch.tensor(
    [[0.10, 0.80, 0.55], [0.65, 0.20, 0.45], [0.70, 0.50, 0.15]],
    dtype=torch.float64,
    requires_grad=True,
)
source = cost.new_tensor([0.20, 0.30, 0.50]).requires_grad_(True)
target = cost.new_tensor([0.40, 0.35, 0.25]).requires_grad_(True)

value_layer = EOTLayer(epsilon=0.3, mode="envelope", forward_tol=1e-12)
value = value_layer(cost, source, target)
value.backward()

plan_layer = EOTLayer(epsilon=0.3, mode="implicit", forward_tol=1e-12)
reference_plan = plan_layer(cost.detach(), source.detach(), target.detach())
print("Regularized EOT value:", float(value.detach()))
print("Cost gradient matches plan:", torch.allclose(cost.grad, reference_plan))
print("Source potential:", source.grad)
print("Target potential:", target.grad)
# docs-end

assert value.shape == torch.Size([])
assert torch.allclose(cost.grad, reference_plan, atol=1e-11, rtol=1e-10)
assert torch.isfinite(source.grad).all() and torch.isfinite(target.grad).all()
assert target.grad[-1] == 0
assert value_layer.diagnostics[0]["backward"]["status"] == "completed"
value_layer.close()
plan_layer.close()
