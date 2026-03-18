# 30-Qubit `quimb` + `jax` NaN Gradient Diagnosis

## Summary

The 30-qubit NaN is not caused by the forward cost evaluation. It comes from reverse-mode differentiation through the compressed MPO application path used in

- [quimb_dist_eq26_2x2_benchmark.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py)

for the diagonal local blocks `A11` and `A22`.

The practical fix for this benchmark is:

- do **not** compress `A_ij |x_ij>` inside the differentiated path
- use exact MPO-on-MPS application with `contract=True, compress=False`

For this 30/29 setup, that exact path is still low-memory.

## Why The NaNs Happen

`quimb`'s generic operator application path is:

- `MPO.apply(...)`
- `tensor_network_apply_op_vec(...)`
- if `compress=True`, call `x.compress(**compress_opts)`

The relevant local source files are:

- [tensor_arbgeom.py](/home/patchouli/miniconda3/envs/MPS/lib/python3.11/site-packages/quimb/tensor/tensor_arbgeom.py)
- [tensor_1d.py](/home/patchouli/miniconda3/envs/MPS/lib/python3.11/site-packages/quimb/tensor/tensor_1d.py)
- [tensor_1d_compress.py](/home/patchouli/miniconda3/envs/MPS/lib/python3.11/site-packages/quimb/tensor/tensor_1d_compress.py)
- [decomp.py](/home/patchouli/miniconda3/envs/MPS/lib/python3.11/site-packages/quimb/tensor/decomp.py)

`quimb` compression uses SVD-based routines. JAX's SVD derivative contains factors of the form

- `1 / ((s_i + s_j) * (s_i - s_j))`

from the rule in:

- [linalg.py](/home/patchouli/miniconda3/envs/MPS/lib/python3.11/site-packages/jax/_src/lax/linalg.py)

So when singular values are equal or nearly equal, the reverse-mode sensitivity can blow up even when the forward value is finite. On top of that, truncated compression introduces discrete rank-selection logic, which is not a smooth map.

This matches the observed pattern:

- forward cost is finite
- reverse-mode `beta` gradient is finite
- reverse-mode `alpha` gradient is non-finite only on the nontrivial diagonal MPO blocks
- small angle perturbations still produce finite forward costs

So the failure is in the backward pass through compression, not in the distributed cost itself.

## Variant Check

I tested the same 30-qubit initialization under several MPO-apply variants.

- baseline compressed path:
  - `contract=True, compress=True, max_bond=64, cutoff=1e-10`
  - cost: `1.2150798201360553`
  - `alpha` finite: `False`
  - `alpha` NaN count: `462`
  - elapsed: `121.82 s`
- no-compression exact path:
  - `contract=True, compress=False`
  - cost: `1.2150801443782933`
  - `alpha` finite: `True`
  - `alpha` NaN count: `0`
  - elapsed: `65.53 s`
- compression with zero cutoff:
  - `contract=True, compress=True, max_bond=64, cutoff=0.0`
  - cost: `1.2150801443782968`
  - `alpha` finite: `False`
  - `alpha` NaN count: `464`
  - elapsed: `69.47 s`

The `cutoff=0` test is important. It shows the problem is not only the dynamic threshold. The compressed/canonicalizing SVD path itself is enough to make reverse-mode unstable here.

## Why `compress=False` Is Feasible Here

For the structured 30-qubit benchmark:

- ansatz MPS bond dimensions are small
- diagonal MPO bond dimension is `7`
- exact `A_ij |x_ij>` bonds stay small enough

Measured exact post-apply bond dimensions:

- agent `(0, 0)`: `x` max bond `4`, `A` max bond `7`, `A x` max bond `28`
- agent `(0, 1)`: `x` max bond `4`, `A` max bond `1`, `A x` max bond `4`
- agent `(1, 1)`: `x` max bond `5`, `A` max bond `7`, `A x` max bond `35`

So for this benchmark, exact MPO-on-MPS application is still modest.

## Code Change

I added an explicit `--apply-no-compress` option to the distributed benchmark and optimizer:

- [quimb_dist_eq26_2x2_benchmark.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py)
- [quimb_dist_eq26_2x2_optimize.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_optimize.py)
- [quimb_dist_eq26_2x2_optimize_random_init.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_optimize_random_init.py)

This now uses an `apply_block_mpo(...)` helper:

- compressed path when `--apply-no-compress` is not set
- exact path when `--apply-no-compress` is set

## Recommended Workflow For 30 Qubits

For the current Eq. (26) distributed benchmark:

1. Use `--apply-no-compress`.
2. Keep `jax_enable_x64=True`.
3. If a future case really requires truncating compression, do not rely on reverse-mode through that path.
4. For those future larger cases, use parameter-shift or SPSA for the angle parameters instead.

## External References

- Quimb JAX example: https://quimb.readthedocs.io/en/main/examples/ex_quimb_within_jax_flax_optax.html
- Quimb tensor-network API docs: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tensor_1d/index.html
- JAX FAQ: https://docs.jax.dev/en/latest/faq.html
