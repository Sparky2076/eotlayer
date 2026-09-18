# Optimization workflow

## 1. Construct costs and marginals

Create a dense cost matrix and strictly positive source and target weights with
equal total mass. All inputs must share their dtype and device. To learn the
weights, construct positive normalized marginals, for example with softmax.

## 2. Choose the differentiation contract

Decide whether the outer loss depends on a finite algorithm, a converged plan, or
the regularized value. Select `unrolled`, `implicit`, or `envelope` accordingly.

## 3. Set tolerances and a solver

For a first float64 check, the examples use `forward_tol=1e-10` and
`backward_tol=1e-8`. These are example settings, not accuracy guarantees.
Float32 usually needs looser tolerances. For `unrolled`, set `iter_num` instead.

## 4. Evaluate a meaningful loss

Call `plan = layer(cost, source, target)`, then construct the outer loss from the
plan. A useful demonstration uses a nonseparable weight matrix $G$ in
$L=\langle G,P\rangle$. `plan.sum()` is fixed by total mass and is a poor
demonstration of a nonzero cost gradient.

## 5. Backpropagate and inspect diagnostics

Call `loss.backward()`, then read `layer.diagnostics` again. The property returns
independent snapshots: a dictionary saved before backward does not update itself.
Inspect each batch item's forward status, backward status, and effective tolerance.

## 6. Validate in the regime you will use

For small problems, compare against a stricter float64 reference and feasible
finite-difference perturbations. Keep the plan and upstream derivative identical
when comparing linear backends. Separately check the actual outer optimization
trajectory if the layer is part of training.

The [Basic plan and gradient](../demos/basic_plan.md) example implements this
workflow end to end.
