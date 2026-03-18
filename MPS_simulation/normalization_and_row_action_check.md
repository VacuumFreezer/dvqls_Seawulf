# Normalization and Row-Action Check

## 1. How `|X_ij>` is read out

The local normalized quantum state is read from the variational MPS circuit as:

`|X_ij> = build_circuit_numpy(...).psi`

and converted to a dense vector by:

`np.asarray(state.to_dense()).reshape(-1)`

This is a unit vector, because:

- the circuit starts from `|0...0>`
- the circuit uses only unitary `RY` and `CZ` gates

For the structured 200-iteration run, the recorded norms are:

`|| |X_ij> ||_2 = [[0.999999999939, 0.999999999940], [0.999999999939, 0.999999999940]]`

So numerically, each `|X_ij>` is unit norm.

## 2. Meaning of `sigma_ij`

The unnormalized local vector is:

`x_ij = sigma_ij |X_ij>`

Therefore:

`||x_ij||_2 = sigma_ij`

For the structured 200-iteration run:

- `sigma = [[0.708204968218, 0.708685232046], [0.708209995206, 0.708691512292]]`
- `||x_ij||_2 = [[0.708204968175, 0.708685232003], [0.708209995163, 0.708691512250]]`

These match to numerical precision.

## 3. Why a row-copy norm can be near 1

The row copy is the concatenated vector:

`[x_i1 ; x_i2]`

Its norm is:

`||[x_i1 ; x_i2]||_2 = sqrt(||x_i1||_2^2 + ||x_i2||_2^2) = sqrt(sigma_i1^2 + sigma_i2^2)`

So if both `sigma_i1` and `sigma_i2` are around `0.708`, the row-copy norm is around `1.00`.

For the structured 200-iteration run:

- row-copy norms: `[1.001892726297, 1.001900722019]`

This is consistent with the `sigma` values above.

By contrast, the final reconstructed global column blocks are:

- `x_1 = (x_11 + x_21) / 2`
- `x_2 = (x_12 + x_22) / 2`

and their norms are:

- `||x_1||_2 = 0.708207481667`
- `||x_2||_2 = 0.708688372121`

## 4. Explicit `b` decomposition

For the current explicit implementation:

- `b_1 = b_11 + b_12`
- `b_2 = b_21 + b_22`
- `b = [b_1 ; b_2]`

For the `global_qubits = 6`, `local_qubits = 5` Eq. (26) benchmark:

- each entry of `b_1` and `b_2` is `0.125`
- each entry of `b_11`, `b_12`, `b_21`, `b_22` is `0.0625`

So the block decomposition is exact.

## 5. Row-action check

The quantity

`sum_j A_ij x_ij`

is now recorded in the optimization report.

For the structured 200-iteration run, the row residuals are:

- `||sum_j A_1j x_1j - b_1||_2 = 1.805695962517`
- `||sum_j A_2j x_2j - b_2||_2 = 1.807336513367`

Therefore, the current distributed optimization is **not** satisfying the raw row equations, even though the surrogate distributed cost is very small. This explains why post-hoc rescaling looked good directionally but is not a valid fix.
