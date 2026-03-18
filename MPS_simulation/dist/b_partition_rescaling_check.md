# b Partition and Rescaling Check

## Section 5.1 target structure

For the 2x2 network, the distributed formulation uses:

- a global right-hand side `b`
- two row blocks `b_1`, `b_2`
- four agent-local vectors `b_ij` such that, for each row `i`,
  `b_i = b_i1 + b_i2`
- a normalized local state `|B_ij>` together with a scalar norm `mu_ij = ||b_ij||_2`
  so that `|b_ij> = mu_ij |B_ij>`

## What the code now does

The distributed builder in [quimb_dist_eq26_2x2_benchmark.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py) now constructs this explicitly:

1. Build the global Eq. (26) right-hand side
   `b = 2^{-n/2} * 1`
2. Cut `b` into the two row blocks
   `b_1 = b[:2^d]`, `b_2 = b[2^d:]`
3. Split each row block equally across the two column agents
   `b_i1 = 0.5 * b_i`, `b_i2 = 0.5 * b_i`
4. For each local block, compute
   `mu_ij = ||b_ij||_2`, `|B_ij> = |b_ij> / mu_ij`

The local cost code uses these per-agent `|B_ij>` and `mu_ij` values directly. It no longer relies on a shared implicit `|+...+>` / scalar pair.

## Result for the Eq. (26) benchmark used here

For `global_qubits = 6` and `local_qubits = 5`:

- `||b_1||_2 = ||b_2||_2 = 0.7071067811865476`
- `||b_11||_2 = ||b_12||_2 = ||b_21||_2 = ||b_22||_2 = 0.3535533905932738`

Because the Eq. (26) right-hand side is uniform, the normalized local states are all the same `|+>^{\\otimes 5}` state, but the important point is that the code now derives that from the explicit block vectors instead of assuming it.

Therefore, for this benchmark, the `b -> b_i -> b_ij` construction is consistent with the Section 5.1 requirement.

## Rescaled-x diagnostic

The saved 200-iteration distributed run ended with a reconstructed `x` that had the correct direction but the wrong norm.

Using the best scalar rescaling

`c* = <x_reconstructed, x_true> / <x_reconstructed, x_reconstructed>`

gives:

- `c* = 0.2811653953744823`
- cosine similarity with `x_true`: `0.9996982581544565`
- raw relative error: `2.555972044673687`
- raw residual: `2.554798552800677`
- rescaled relative error: `0.024564051842999448`
- rescaled residual: `0.02224696445838185`

So the dominant issue in the current distributed result is a global scale mismatch, not a directional mismatch.
