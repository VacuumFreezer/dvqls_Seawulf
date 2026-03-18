# 30-Qubit Direct-MPO Smoke Test

## Conclusion
Under the revised workflow, a 30-qubit forward pass looks feasible in memory, but the current reverse-mode optimization path is not yet reliable.
This smoke test never materialized dense global A or dense b, built the blocks directly as MPOs, completed the forward cost evaluation, but the reverse-mode alpha gradient was not finite.

## Mathematical Block Construction
For the current 2x2 split, the global basis is partitioned by the first qubit.
With 29 local qubits,
- `A11 = (H_rest + J Z_1 + eta I) / zeta`
- `A12 = I / zeta`
- `A21 = I / zeta`
- `A22 = (H_rest - J Z_1 + eta I) / zeta`
- `b_i = (1 / sqrt(2)) |+^{29}>`
- `b_ij = (1 / (2 sqrt(2))) |+^{29}>`

## Timings
- Spectrum scaling (`eta`, `zeta`) via MPO DMRG: `0.938420 s`.
- Direct block MPO construction: `0.938420 s`.
- One forward distributed cost evaluation: `0.677296 s`.
- One reverse-mode gradient evaluation: `113.420843 s`.

## Diagnostics
- Forward cost value: `1.21507982014`.
- Alpha gradient finite: `False`.
- Beta gradient finite: `True`.
- Alpha gradient L2 norm: `nan`.
- Beta gradient L2 norm: `0.982391188818`.
- `eta = 33.2380730524`.
- `zeta = 63.3106153378`.

## MPO Sizes
- `A11` summary: `{'num_tensors': 29, 'max_bond': 7, 'bond_sizes': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]}`.
- `A12` summary: `{'num_tensors': 29, 'max_bond': 1, 'bond_sizes': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}`.
- `A22` summary: `{'num_tensors': 29, 'max_bond': 7, 'bond_sizes': [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]}`.

## Practical Reading
Your understanding is mostly right.
The correction is that the residual does not need a dense global matrix either, but it does still need to be defined carefully from inner products of the reconstructed averaged column blocks, not only from the per-agent row copies.
The real requirement for 30 qubits is: build `A_ij` and `b_ij` directly as MPO/MPS objects, and do all monitoring through contraction formulas.
Even with that fix, the present JAX reverse-mode path can still fail numerically at 30 qubits, so memory feasibility and optimization feasibility are not the same claim.

## Artifacts
- JSON: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/quimb_dist_eq26_2x2_direct_mpo_smoke_30q.json`
- Report: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/quimb_dist_eq26_2x2_direct_mpo_smoke_30q_report.md`
