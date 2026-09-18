import torch

from eotlayer import EOTLayer


cost = torch.tensor(
    [[0.10, 0.80, 0.55], [0.65, 0.20, 0.45], [0.70, 0.50, 0.15]],
    dtype=torch.float64,
    requires_grad=True,
)
source = torch.tensor([0.20, 0.30, 0.50], dtype=cost.dtype)
target = torch.tensor([0.40, 0.35, 0.25], dtype=cost.dtype)

# A converged forward solve with an implicit, PCG-based backward pass.
layer = EOTLayer(
    epsilon=0.3,
    mode="implicit",
    forward_tol=1e-10,
    forward_iter_max=10_000,
    backward_solver="jacobi_w",
    backward_tol=1e-8,
    backward_iter_max=1_000,
)
plan = layer(cost, source, target)
loss = (plan * torch.arange(9, dtype=cost.dtype).reshape(3, 3)).sum()
loss.backward()

# Diagnostics are snapshots; read them again after backward.
record = layer.diagnostics[0]
assert record["forward"]["status"] == "converged"
assert record["backward"]["status"] == "converged"
print("forward iterations:", record["forward"]["iterations"])
print("backward iterations:", record["backward"]["iterations"])
print("backward residual:", record["backward"]["relative_residual"])

# The same call signature also supports the other differentiation contracts.
finite_plan = EOTLayer(0.3, "unrolled", iter_num=8)(
    cost.detach(), source, target
)
transport_value = EOTLayer(
    0.3, "envelope", forward_tol=1e-10, forward_iter_max=10_000
)(cost.detach(), source, target)
assert finite_plan.shape == cost.shape
assert transport_value.shape == torch.Size([])
