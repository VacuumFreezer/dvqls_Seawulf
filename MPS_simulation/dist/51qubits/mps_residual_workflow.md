# 51-Qubit MPS Residual Workflow

This note summarizes the residual computation used in
`quimb_dist_eq26_j0p1_two_layer_ry_cz_true_scales_51q.py`.

## State Representation

We do **not** build a dense vector $x \in \mathbb{C}^{2^{51}}$.
Instead, each local variational state is stored as an MPS:

$$
|x_{ij}\rangle, \qquad |z_{ij}\rangle .
$$

For the $2 \times 2$ partition, the reconstructed column states are

$$
|x_1\rangle
=
\frac{1}{2}
\left(
\sigma_{00} |x_{00}\rangle
+
\sigma_{10} |x_{10}\rangle
\right),
$$

$$
|x_2\rangle
=
\frac{1}{2}
\left(
\sigma_{01} |x_{01}\rangle
+
\sigma_{11} |x_{11}\rangle
\right).
$$

## Optimization Objective

The local objective is the quadratic residual written through inner products:

$$
f(\alpha, \beta)
=
\sum_{i=1}^{2} \sum_{j=1}^{2}
\left\|
A_{ij} \bigl( \sigma_{ij} |x_{ij}\rangle \bigr)
-
\sum_{k=1}^{2} L_{jk} \lambda_{ik} |z_{ik}\rangle
-
b_{ij}
\right\|_2^2 .
$$

This is evaluated by expanding the square into overlaps such as
$\langle A x, A x \rangle$, $\langle b, A x \rangle$, and
$\langle z, z \rangle$, so no dense $2^{51}$ vector is formed.

## Reported Global Residual

The reported residual is computed from the averaged column states:

$$
r_i
=
A_{i1} |x_1\rangle + A_{i2} |x_2\rangle - |b_i\rangle,
\qquad i = 1, 2.
$$

Then

$$
\| A x - b \|
=
\sqrt{
\| r_1 \|_2^2 + \| r_2 \|_2^2
},
$$

with each row norm obtained from an MPS overlap:

$$
\| r_i \|_2^2 = \langle r_i, r_i \rangle .
$$

## Key Point

The workflow reconstructs $x$ **implicitly as MPS objects**, not as a dense
classical vector. The residual norm is therefore obtained by
MPO--MPS application plus MPS inner products.
