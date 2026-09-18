# API reference

The supported package entry point is `from eotlayer import EOTLayer`. The
constructor signature on the reference page is extracted from source at build
time and checked against `project-contract.json`.

```{toctree}
:maxdepth: 1

layer
diagnostics
backends
```

Internal modules such as `eotlayer.forward`, `eotlayer.backward`, and
`eotlayer._core` support research and backend development. They are not additional
stable top-level exports in this pre-alpha release.
