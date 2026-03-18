# Distributed Eq. (26) 2x2 MPS Benchmark

## Goal

This benchmark uses the 11-qubit Eq. (26) linear system from the VQLS paper as the global problem and maps it to a 2x2 distributed network. The purpose is not to reproduce the hardware-oriented Hadamard-test expansion from the draft PDF. Instead, the goal is to evaluate the same distributed objective as efficiently as possible on a classical MPS/MPO simulator.

The implementation lives in [quimb_dist_eq26_2x2_benchmark.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py).

## 1. Global Linear System

The global 11-qubit operator is the Eq. (26) Ising-inspired matrix

`A = (sum_j X_j + J sum_j Z_j Z_{j+1} + eta I) / zeta`

with `J = 0.1` and `eta, zeta` chosen exactly as in the centralized benchmark so that the condition number target is `kappa = 20`.

The right-hand side is the normalized plus state

`|b> = |+>^{11}`.

## 2. 2x2 Block Partition

The Hilbert space is split as

`C^(2^11) = C^2 \otimes C^(2^10)`.

This induces a 2x2 block matrix

`A = [[A11, A12], [A21, A22]]`

where each block is a `1024 x 1024` operator acting on 10 qubits.

Using the first qubit as the partition qubit, the four local operators are

- `A11 = (H_10 + J Z_0 + eta I) / zeta`
- `A12 = I / zeta`
- `A21 = I / zeta`
- `A22 = (H_10 - J Z_0 + eta I) / zeta`

with

`H_10 = sum_{r=0}^9 X_r + J sum_{r=0}^8 Z_r Z_{r+1}`.

This means each agent only needs a 10-qubit MPO.

## 3. Local Right-Hand Side Vectors

The global `|b>` is partitioned by row first:

- `b1 = (1 / sqrt(2)) |+>^{10}`
- `b2 = (1 / sqrt(2)) |+>^{10}`

For the column split inside each row, the benchmark uses the symmetric choice

- `b_{i1} = b_i / 2`
- `b_{i2} = b_i / 2`

so that `b_{i1} + b_{i2} = b_i`.

## 4. Network Structure

There are four agents:

- row 1: `[1,1]`, `[1,2]`
- row 2: `[2,1]`, `[2,2]`

In the current code, each agent is assumed to have a unit self-arc in addition to the row edge. To keep the row operator consistent with a Laplacian-type coupling, I use the row-stochastic self-loop normalization

`L_row = I - D^(-1) W = [[0.5, -0.5], [-0.5, 0.5]]`

for the messenger variable `z`.

Each column uses the two-node consensus matrix

`W_col = [[0.5, 0.5], [0.5, 0.5]]`

for the replicated solution variable `x` and the gradient tracker `y`.

## 5. Local Variational Variables

Each agent `[i,j]` maintains two parameterized 10-qubit states:

- `|x_ij> = sigma_ij |X_ij(alpha_ij)>`
- `|z_ij> = lambda_ij |Z_ij(beta_ij)>`

The normalized states `|X_ij>` and `|Z_ij>` are produced by the same 4-layer `Ry/CZ` MPS ansatz used in the centralized scripts. The scalar amplitudes `sigma_ij` and `lambda_ij` are stored as the first element of the local parameter vector.

## 6. Local Objective Used in the Benchmark

For each agent `[i,j]`, define

`s_ij = A_ij |x_ij> - sum_k L_row[j,k] |z_ik>`.

The local cost is

`f_ij = || s_ij - |b_ij> ||^2`.

The global distributed objective is

`f = sum_i sum_j f_ij`.

This matches the structure in Sections 4 and 5 of the PDF, but it is evaluated directly as a tensor-network contraction instead of as a sum of hardware measurement subroutines.

## 7. Efficient MPS/MPO Evaluation

The benchmark never forms the residual state explicitly. Instead it expands the squared norm into overlaps:

- `||A_ij x_ij||^2`
- `||b_ij||^2`
- `sum_{k,p} L_jk L_jp <z_ik | z_ip>`
- `Re <A_ij x_ij | z_ik>`
- `Re <b_ij | A_ij x_ij>`
- `Re <b_ij | z_ik>`

This is the key efficiency choice.

Why this is better for an MPS simulator:

- each `A_ij` is applied once as an MPO to a 10-qubit MPS
- all other terms are simple MPS-MPS overlaps
- there is no need to expand the cost into Hadamard-test terms
- there is no need to simulate ancillas or controlled versions of the local operators

## 8. Reverse-Mode Gradient Strategy

The fastest classical route is to differentiate the full distributed objective directly:

- `grad_alpha = d f / d alpha`
- `grad_beta = d f / d beta`

This is important because:

- `grad_alpha[i,j]` is exactly the local `x`-gradient for agent `[i,j]`
- `grad_beta[i,j]` is already the row-aggregated messenger gradient, since `z_ij` appears in both row costs of row `i`

So one reverse-mode pass of the total objective gives all gradients needed by the distributed algorithm.

## 9. One Distributed Iteration in the Benchmark

The timed iteration follows the Section 5 update logic in a classical-exact form:

1. Evaluate the current global objective and its reverse-mode gradients.
2. Use the `beta`-gradient to update the messenger moments and the messenger parameters with Adam.
3. Use the tracked variable `y` to update the solution moments and the solution parameters with column consensus plus Adam.
4. Re-evaluate the `alpha`-gradient at the updated parameters.
5. Update the gradient tracker with
   `y(t+1) = W_col y(t) + g_alpha(t+1) - g_alpha(t)`.

This means one exact iteration requires:

- one full reverse-mode pass for `(grad_alpha(t), grad_beta(t))`
- one additional reverse-mode pass for `grad_alpha(t+1)`

The benchmark includes both passes in the reported one-iteration timing.

## 10. Why This Matches the PDF

The implementation is faithful to the algorithmic structure in Sections 4 and 5:

- row coupling is enforced through the Laplacian-weighted messenger terms
- column consistency is enforced through consensus mixing on the replicated solution parameters
- gradient tracking is performed on the solution parameters
- Adam is used as the local descent rule

The only deliberate simplification is the evaluation method:

- the PDF expands terms for future hardware measurement
- the benchmark contracts the same objective directly on an MPS/MPO simulator

That simplification is valid for classical simulation and is exactly the efficient choice you asked for.
