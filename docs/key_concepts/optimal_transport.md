# Entropic optimal transport

## Costs, marginals, and the plan

Let $C\in\mathbb{R}^{m\times n}$ contain the cost of moving mass from each source
to each target. The vectors $a\in\mathbb{R}_{>0}^m$ and
$b\in\mathbb{R}_{>0}^n$ specify the prescribed masses. EOTLayer requires
$\sum_i a_i=\sum_j b_j$; this common mass does not have to equal one.

The transport plan solves

```{math}
\min_{P\geq 0}\quad
\langle C,P\rangle + \varepsilon\sum_{i,j}P_{ij}(\log P_{ij}-1)
\quad\text{subject to}\quad P\mathbf{1}=a,\quad P^\top\mathbf{1}=b.
```

The scalar $\varepsilon>0$ controls entropic regularization and has the same units
as the cost. It is configuration, not a differentiable tensor input.
The entropy convention above determines the value returned by `envelope` mode;
it is not a debiased Sinkhorn divergence.

## Log-domain Sinkhorn

Writing $\log K=-C/\varepsilon$, the solver alternates

```{math}
\begin{aligned}
\log u_i &\leftarrow \log a_i-\operatorname{logsumexp}_j(\log K_{ij}+\log v_j),\\
\log v_j &\leftarrow \log b_j-\operatorname{logsumexp}_i(\log K_{ij}+\log u_i).
\end{aligned}
```

The plan is recovered from
$P_{ij}=\exp(\log u_i+\log K_{ij}+\log v_j)$.
Log-domain updates improve numerical stability; they do not eliminate precision
limits or underflow in the materialized plan.

## Epsilon continuation

In `implicit` and `envelope` modes, `epsilon_scaling_start` enables a geometric
schedule down to `epsilon`. `epsilon_scaling_decay` is strictly between zero and
one, and defaults to `0.5` when a start value is supplied. A plan is accepted only
at the target epsilon.

```python
layer = EOTLayer(
    epsilon=0.1,
    mode="implicit",
    epsilon_scaling_start=0.8,
    epsilon_scaling_decay=0.5,
    forward_tol=1e-10,
    backward_solver="jacobi_w",
)
```

This snippet assumes `EOTLayer` has been imported. The start value must be at
least the target epsilon. Omitting both continuation arguments keeps fixed-epsilon
Sinkhorn. Continuation is not available in `unrolled` mode.
