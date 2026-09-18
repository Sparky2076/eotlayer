# Demos and examples

These examples use explicit float64 tensors and the public `EOTLayer` interface.
Every displayed script is included from its runnable source file, so the page and
the example stay in sync.

```{toctree}
:maxdepth: 1

basic_plan
compare_modes
envelope
batching
```

After installation, run an example from the project root:

```bash
python examples/docs_quickstart.py
```

Run all four documented examples and the public-signature check with:

```bash
python scripts/check_docs.py --run-examples
```
