# Troubleshooting

## `ModuleNotFoundError: eotlayer`

Install the source package into the interpreter used to run the example:

```bash
python -m pip install -e .
python -c "import eotlayer; print(eotlayer.__file__)"
```

## Missing PyTorch, NumPy, or SciPy

Use `python -m pip` for the selected environment. The package declares all three
runtime dependencies. The docs can build without them, but numerical examples
cannot run without them.

## Forward did not converge

Inspect `normalized_marginal_error` and `forward_tol`. Check that the tolerance
is attainable for the dtype, increase the iteration budget when appropriate, and
consider epsilon continuation for difficult small-epsilon problems. Compare a
small float64 instance to distinguish precision limits from insufficient iterations.

## Operator is not positive definite

PCG checks positive curvature and preconditioner positivity. A very concentrated
plan, floating-point underflow, or an insufficiently accurate forward state may
cause trouble. Inspect the state and tolerances and use a small Dense reference
to diagnose the case. A different solver is not by itself an accuracy certificate.

## IC(0) rejects CUDA / Sparse-W rejects CPU

This is the current device contract. Use CPU for public `ic0` and a compatible
CUDA/cuDSS environment for public `sparse_w`. The layer does not silently move the
problem to a different device.

## My plan loss has a zero gradient

The total plan mass is fixed. A loss such as `plan.sum()` is therefore constant
with respect to costs. Likewise, a weight matrix of the form $G_{ij}=u_i+v_j$
produces a fixed value when marginals are prescribed. Try the nonseparable weights
in the [basic example](../demos/basic_plan.md).

## Diagnostics still say `pending`

Read `layer.diagnostics` again after `.backward()`. Earlier snapshots do not
update. Also ensure that the loss actually depends on the returned plan or value.

## Double backward fails

The implicit and envelope custom functions support first derivatives only. A
higher derivative of those paths is outside the current contract. Finite
unrolling has a different derivative target and should not be silently substituted.

## Documentation search looks incomplete

Serve the HTML directory through `python -m http.server` and allow the search index
to load. When opening `index.html` as a local file, browser restrictions can affect
some scripts. Mathematical rendering uses MathJax from a CDN and needs network
access on first load.
