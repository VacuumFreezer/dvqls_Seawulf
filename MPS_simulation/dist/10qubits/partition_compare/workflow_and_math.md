# Partition Compare Workflow and Formulation

## Goal

This folder prepares a controlled partition comparison for one shared `13`-qubit Ising-inspired linear system. Every case uses the same global matrix and the same right-hand side, and only the partition changes:

- `8 x 8` agents, each agent carries a `10`-qubit local block
- `4 x 4` agents, each agent carries an `11`-qubit local block
- `2 x 2` agents, each agent carries a `12`-qubit local block
- `1 x 1` agent, a single `13`-qubit local block

We always enforce

$$
m = n, \qquad m \cdot 2^{q_{\mathrm{local}}} = 2^{13},
$$

where:

- $m = n_{\mathrm{rows}}$
- $n = n_{\mathrm{cols}}$
- $q_{\mathrm{local}}$ is the local qubit count per agent
- $d = 2^{q_{\mathrm{local}}}$ is the local block dimension

So every case solves the same global problem in $\mathbb{R}^{2^{13}}$, but with different distributed layouts.

## Shared Linear System

The common `13`-qubit Ising-inspired Hamiltonian is

$$
H_0 = \sum_{k=1}^{13} X_k \;+\; J \sum_{k=1}^{12} Z_k Z_{k+1},
\qquad J = 0.1.
$$

As in the earlier workflow, we shift and scale this operator to obtain the linear-system matrix

$$
A = \frac{H_0 + \eta I}{\zeta},
$$

with

$$
\eta = \frac{\lambda_{\max}(H_0) - \kappa \lambda_{\min}(H_0)}{\kappa - 1},
\qquad
\zeta = \lambda_{\max}(H_0) + \eta,
$$

so that the spectrum of $A$ is controlled by the common parameter $\kappa$.

The right-hand side is the normalized all-ones vector

$$
b = \frac{1}{\sqrt{2^{13}}}\mathbf{1}.
$$

Because every partition uses the same $(A,b)$, any difference in convergence comes from the partition and communication pattern rather than from a different problem instance.

## Block Partition

Let the global solution be partitioned into `n` column blocks:

$$
x =
\begin{bmatrix}
x_1 \\
\vdots \\
x_n
\end{bmatrix},
\qquad
x_j \in \mathbb{R}^d,
\qquad
x \in \mathbb{R}^{nd} = \mathbb{R}^{2^{13}}.
$$

For each row $i \in \{1,\dots,m\}$ and column $j \in \{1,\dots,n\}$, agent $[[ij]]$ stores a local copy $x_{ij} \in \mathbb{R}^d$ and, when $n>1$, a messenger variable $z_{ij} \in \mathbb{R}^d$.

The global matrix is sliced into local dense blocks

$$
A_{ij} \in \mathbb{R}^{d \times d},
\qquad
i = 1,\dots,m,\quad j = 1,\dots,n.
$$

Within each row, we stack the local variables as

$$
x_i =
\begin{bmatrix}
x_{i1} \\
\vdots \\
x_{in}
\end{bmatrix}
\in \mathbb{R}^{nd},
\qquad
z_i =
\begin{bmatrix}
z_{i1} \\
\vdots \\
z_{in}
\end{bmatrix}
\in \mathbb{R}^{nd}.
$$

We also define the block-diagonal row operator and stacked row right-hand side:

$$
\bar A_i = \mathrm{blkdiag}(A_{i1}, \dots, A_{in}) \in \mathbb{R}^{nd \times nd},
$$

$$
\bar b_i =
\begin{bmatrix}
b_{i1} \\
\vdots \\
b_{in}
\end{bmatrix}
\in \mathbb{R}^{nd},
\qquad
b_{ij} = \frac{b_i}{n},
$$

where $b_i \in \mathbb{R}^d$ is the $i$-th row block of the global vector $b$.

## Communication Graphs

Both communication layers are line graphs.

- Column graph: for each fixed column $j$, the agents $[[1j]], [[2j]], \dots, [[mj]]$ communicate along a line graph. This is the consensus/gradient-tracking layer for replicated copies of $x_j$.
- Row graph: for each fixed row $i$, the agents $[[i1]], [[i2]], \dots, [[in]]$ communicate along a line graph. This is the messenger layer for the variables $z_{ij}$.

### Metropolis Mixing Matrix

For the column-consensus layer, the code uses the symmetric Metropolis matrix on the line graph with `m` vertices. If $r \sim s$ means that vertices $r$ and $s$ are neighbors in the line graph and $d_r$ is the graph degree of vertex $r$, then

$$
(W_{\mathrm{col}})_{rs}
=
\begin{cases}
\dfrac{1}{1 + \max(d_r, d_s)}, & r \sim s,\\[1.0ex]
1 - \sum_{t \in \mathcal N_r} (W_{\mathrm{col}})_{rt}, & r = s,\\[1.0ex]
0, & \text{otherwise.}
\end{cases}
$$

For a line graph with more than two vertices, every edge weight is `1/3`, the endpoint diagonal entries are `2/3`, and the interior diagonal entries are `1/3`.

Example: for a `4`-node line graph,

$$
W_{\mathrm{col}}^{(4)} =
\begin{bmatrix}
\tfrac{2}{3} & \tfrac{1}{3} & 0 & 0 \\
\tfrac{1}{3} & \tfrac{1}{3} & \tfrac{1}{3} & 0 \\
0 & \tfrac{1}{3} & \tfrac{1}{3} & \tfrac{1}{3} \\
0 & 0 & \tfrac{1}{3} & \tfrac{2}{3}
\end{bmatrix}.
$$

For the original `2 x 2` case, this reduces to

$$
W_{\mathrm{col}}^{(2)} =
\begin{bmatrix}
\tfrac{1}{2} & \tfrac{1}{2} \\
\tfrac{1}{2} & \tfrac{1}{2}
\end{bmatrix},
$$

which matches the earlier code exactly.

For the `8 x 8` case, the same pattern gives an `8 x 8` tridiagonal Metropolis matrix with `2/3` on the two endpoint diagonals and `1/3` on every interior diagonal and edge entry.

### Row Laplacian

For the messenger layer, the code uses the standard line-graph Laplacian

$$
L_{\mathrm{row}} = D - A_{\mathrm{graph}},
$$

where $D$ is the degree matrix and $A_{\mathrm{graph}}$ is the adjacency matrix of the row line graph.

Example: for a `4`-node line graph,

$$
L_{\mathrm{row}}^{(4)} =
\begin{bmatrix}
1 & -1 & 0 & 0 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
0 & 0 & -1 & 1
\end{bmatrix}.
$$

For the original `2 x 2` case,

$$
L_{\mathrm{row}}^{(2)} =
\begin{bmatrix}
1 & -1 \\
-1 & 1
\end{bmatrix}.
$$

## Lifted Objective for the General `m x n` Network

The row-wise lifted objective is

$$
f_i(x_i, z_i)
=
\left\|
\bar A_i x_i - \bar b_i - (L_{\mathrm{row}} \otimes I_d) z_i
\right\|_2^2.
$$

The full distributed objective is therefore

$$
f(x,z)
=
\sum_{i=1}^{m}
\left\|
\bar A_i x_i - \bar b_i - (L_{\mathrm{row}} \otimes I_d) z_i
\right\|_2^2.
$$

Writing this component-wise gives

$$
f(x,z)
=
\sum_{i=1}^{m}
\sum_{j=1}^{n}
\left\|
A_{ij} x_{ij}
- b_{ij}
- \sum_{k=1}^{n} (L_{\mathrm{row}})_{jk} z_{ik}
\right\|_2^2.
$$

This is the exact form implemented in the optimizer.

## Global Solution Estimate and Metrics

The optimizer forms one global estimate by averaging the replicated column blocks across rows:

$$
\bar x_j = \frac{1}{m}\sum_{i=1}^{m} x_{ij},
\qquad
x_{\mathrm{est}} =
\begin{bmatrix}
\bar x_1 \\
\vdots \\
\bar x_n
\end{bmatrix}.
$$

The four reported quantities are:

1. Global lifted cost

$$
f(x,z).
$$

2. Global residual

$$
\|A x_{\mathrm{est}} - b\|_2.
$$

3. Consensus error between rows

$$
e_{\mathrm{cons}} =
\left(
\sum_{j=1}^{n}
\sum_{i=1}^{m-1}
\|x_{ij} - x_{i+1,j}\|_2^2
\right)^{1/2}.
$$

4. Solution `L2` error

$$
e_{\mathrm{sol}} = \|x_{\mathrm{est}} - x^\star\|_2,
$$

where $x^\star$ is the exact sparse-solve solution of the shared global linear system.

## `8 x 8` Case

For the `8 x 8` network:

- `m = 8`, `n = 8`
- `d = 2^{10}`
- there are `64` agents
- each column has `8` replicated copies of the same block
- each row solves an `8`-block lifted problem coupled by `L_row^(8)`

The communication objects are:

$$
W_{\mathrm{col}}^{(8)} \in \mathbb{R}^{8 \times 8},
\qquad
L_{\mathrm{row}}^{(8)} \in \mathbb{R}^{8 \times 8}.
$$

The `x` variables are mixed by $W_{\mathrm{col}}^{(8)}$ along each column, while the messenger correction inside each row is generated by $(L_{\mathrm{row}}^{(8)} \otimes I_d) z_i$.

## Single-Agent `1 x 1` Case

For the single-agent case:

- `m = 1`, `n = 1`
- `d = 2^{13}`
- there is no replication of `x`
- there is no row communication
- the messenger variable disappears

In this case,

$$
W_{\mathrm{col}}^{(1)} = [1],
\qquad
L_{\mathrm{row}}^{(1)} = [0].
$$

So the lifted objective reduces to the ordinary least-squares objective

$$
f(x) = \|A x - b\|_2^2.
$$

Accordingly, the code disables the messenger contribution and the consensus metric is identically zero.

## Variational Ansatz

Each agent uses the requested local ansatz on `q_local` wires:

$$
|\psi(\theta)\rangle
=
\left(
\prod_{\ell=1}^{2}
U_{\mathrm{CZ,odd}}^{(\ell)}
U_{\mathrm{RY},2}^{(\ell)}
U_{\mathrm{CZ,even}}^{(\ell)}
U_{\mathrm{RY},1}^{(\ell)}
\right)
H^{\otimes q_{\mathrm{local}}}
|0\rangle^{\otimes q_{\mathrm{local}}}.
$$

The first scalar optimization parameter multiplies the final local state, so each agent optimizes one scale plus the ansatz angles.

## Optimization Workflow

The optimizer follows the same high-level structure as the earlier distributed workflow:

- the `x` variables use column mixing, gradient tracking, and Adam-style scaling
- the `z` variables use Adam-style updates for the messenger layer
- the `1 x 1` branch removes the messenger update entirely

At every iteration the run records:

- global lifted cost
- global residual
- consensus error between rows
- solution `L2` error

Reports truncate long vectors and matrix previews to the configured number of leading elements.

## Files in This Folder

- `partition_compare_common.py`: shared math, state construction, partitioning, metrics, and report helpers
- `partition_compare_optimize.py`: main optimizer entry point
- `run_partition_case.py`: wrapper that sends one prepared case into its own output directory
- `gen_tasks.py`: task generator for Seawulf array jobs
- `submit_array.slurm`: prepared Slurm launcher
- `param.yaml`: shared hyperparameters and case definitions
