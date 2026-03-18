# Cost And Solution Reconstruction Check

## Purpose

This note records the detailed consistency check requested for the distributed 2x2 MPS implementation:

1. check that the coded distributed global cost is equivalent to the paper definition,
2. check that the recovered global solution vector `x` is reconstructed correctly from the local block variables.

The relevant code is in:

- [quimb_dist_eq26_2x2_benchmark.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py)
- [quimb_dist_eq26_2x2_optimize.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_optimize.py)

## 1. Paper Definition Of The Distributed Cost

From Section 4.3 and Section 5.4 of the PDF, the local objective is

`f_ij = || A_ij x_ij - b_ij - sum_k L_jk z_ik ||^2`

and the global distributed objective is

`f = sum_i sum_j f_ij`.

With the variational parameterization

- `|x_ij> = sigma_ij |X_ij>`
- `|z_ij> = lambda_ij |Z_ij>`

the paper expands the local cost as

`f_ij = <s_ij|s_ij> + <b_ij|b_ij> - 2 Re <b_ij|s_ij>`

with

`s_ij = sigma_ij A_ij |X_ij> - sum_k L_jk lambda_ik |Z_ik>`.

Expanding this gives the exact terms used in code:

- `sigma_ij^2 <X_ij| A_ij^\dagger A_ij |X_ij>`
- `sum_{k,p} L_jk L_jp lambda_ik lambda_ip <Z_ik|Z_ip>`
- `- 2 sigma_ij sum_k L_jk lambda_ik Re <X_ij| A_ij^\dagger |Z_ik>`
- `+ ||b_ij||^2`
- `- 2 sigma_ij Re <b_ij| A_ij |X_ij>`
- `+ 2 sum_k L_jk lambda_ik Re <b_ij|Z_ik>`

This is exactly what `global_cost_jax` and `global_cost_numpy` evaluate.

## 2. How The Cost Is Evaluated In Code

The code does **not** build the residual MPS explicitly. Instead, for each agent `[i,j]`, it computes:

1. the local normalized ansatz state `|X_ij>`,
2. the local normalized messenger states `|Z_i1>`, `|Z_i2>`,
3. the MPO-applied state `A_ij |X_ij>`,
4. all overlaps required by the expanded formula above.

This is mathematically equivalent to the paper definition, but much cheaper on a classical MPS simulator because it avoids:

- controlled-unitary Hadamard-test constructions,
- explicit ancillas,
- repeated decomposition of the same local residual into hardware-measurement subterms.

## 3. The Main Bug I Found

The original implementation tried to write the four 10-qubit local operators analytically. That was partly right:

- the off-diagonal blocks `A12` and `A21` were correct,
- the diagonal blocks `A11` and `A22` were **not** the exact blocks of the simulator matrix.

This was the real reason the previous run produced a tiny distributed cost together with a very large global residual.

### What was wrong

I originally assumed a simple closed-form block formula for the 11-qubit Eq. (26) matrix. In quimb's simulator basis, that assumption did not match the exact block structure of the matrix produced by the global MPO/sparse builder.

Numerically, before the fix:

- `A12` and `A21` matched the true simulator blocks,
- `A11` and `A22` had large block errors.

So the distributed objective was minimizing the wrong local operators.

### How I fixed it

The code now constructs the local operators by **exact block extraction** from the global 11-qubit simulator matrix:

1. build the full scaled 11-qubit Eq. (26) operator `A`,
2. convert it to a dense `2048 x 2048` matrix once during setup,
3. slice its exact `1024 x 1024` block submatrices,
4. convert those exact blocks back to 10-qubit MPOs.

This removes the basis-ordering ambiguity completely.

## 4. Consistency Check For The Cost Function

After the block fix, I checked the coded cost against the direct definition of

`sum_i sum_j || A_ij x_ij - b_ij - sum_k L_jk z_ik ||^2`

using explicit dense vectors at the same parameter point.

At the initialization point, the two values are:

- coded cost: `0.8650392441779186`
- direct dense residual cost: `0.8650399111512868`
- absolute difference: `6.67e-07`

This level of difference is only numerical contraction noise. So the coded distributed cost is equivalent to the paper definition.

## 5. How The Global Solution Vector x Is Reconstructed

Each agent `[i,j]` stores a local block variable

`x_ij = sigma_ij |X_ij>`.

For a 2x2 partition:

- column `j = 1` corresponds to the first `1024` amplitudes of the global solution block structure,
- column `j = 2` corresponds to the second `1024` amplitudes.

Within each row, the recovered row solution is therefore the stacked vector

- `x_row1 = [x_11 ; x_12]`
- `x_row2 = [x_21 ; x_22]`

Because the distributed algorithm uses row-wise replicated copies of the same global block variables, the final recovered global solution is the column-wise average of the two rows:

- `x_1 = (x_11 + x_21) / 2`
- `x_2 = (x_12 + x_22) / 2`
- `x_recovered = [x_1 ; x_2]`

Equivalently, this is the average of the two stacked row vectors:

`x_recovered = (x_row1 + x_row2) / 2`.

This is exactly what the optimization script now uses for:

- the global residual `||A x - b||_2`,
- the relative solution error,
- the row-consensus error.

## 6. Consistency Check For The Recovered x

To verify the reconstruction, I compared two ways of computing the global residual:

1. apply the full 11-qubit sparse matrix to the recovered stacked vector `x_recovered`,
2. apply the four local block operators to the recovered block vectors `x_1`, `x_2`, then stack the two row residuals.

After the block fix, the two residual vectors agree to numerical precision:

- `|| r_full - r_blocks ||_2 = 1.22e-14`

This means the recovered stacked vector `x_recovered` is now consistent with the global simulator matrix.

## 7. Why The Old Relative Error Was Nonsense

The previous relative error around `5` while the distributed cost was around `1e-4` was not a property of the distributed algorithm itself. It was caused by the wrong diagonal local blocks.

So the old low-cost/high-error result should be treated as invalid.

## 8. Current Status After The Fix

With the exact block extraction:

- the distributed cost is now the correct paper-defined lifted objective,
- the recovered global vector `x` is reconstructed correctly as the stacked averaged local blocks,
- the full residual computed from `A x - b` is consistent with the local block formulation.

I also refreshed the one-step benchmark under this corrected implementation, so the files in `MPS_simulation/dist` now use the fixed block construction.
