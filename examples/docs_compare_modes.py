"""Compare finite unrolling and two implicit backends on identical data."""

# docs-start
import torch
from eotlayer import EOTLayer

base_cost = torch.tensor(
    [[0.10, 0.80, 0.55], [0.65, 0.20, 0.45], [0.70, 0.50, 0.15]],
    dtype=torch.float64,
)
source = base_cost.new_tensor([0.20, 0.30, 0.50])
target = base_cost.new_tensor([0.40, 0.35, 0.25])
weights = base_cost.new_tensor(
    [[0.2, -0.4, 0.7], [0.5, 0.1, -0.3], [-0.6, 0.8, 0.4]]
)
settings = {
    "unrolled / 8": dict(mode="unrolled", iter_num=8),
    "implicit / dense": dict(mode="implicit", backward_solver="dense"),
    "implicit / jacobi_w": dict(mode="implicit", backward_solver="jacobi_w"),
}
gradients = {}
for name, options in settings.items():
    cost = base_cost.clone().requires_grad_(True)
    layer = EOTLayer(epsilon=0.3, **options)
    plan = layer(cost, source, target)
    (plan * weights).sum().backward()
    gradients[name] = cost.grad.clone()
    error = layer.diagnostics[0]["forward"]["normalized_marginal_error"]
    print(name, "marginal error:", error)
    layer.close()

reference = gradients["implicit / dense"]
for name, gradient in gradients.items():
    relative = (gradient - reference).norm() / reference.norm()
    print(name, "gradient difference:", float(relative))
# docs-end

assert all(torch.isfinite(g).all() for g in gradients.values())
assert torch.allclose(gradients["implicit / jacobi_w"], reference, atol=1e-9, rtol=1e-7)
