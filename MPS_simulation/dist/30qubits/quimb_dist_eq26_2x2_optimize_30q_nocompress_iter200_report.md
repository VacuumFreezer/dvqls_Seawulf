# 30-Qubit Distributed Optimization Report

## Setup
- Global system: `30` qubits.
- Local block size: `29` qubits.
- Iterations: `30`.
- Learning rate: `0.01`.
- Initialization mode: `structured_linspace`.
- No-compression MPO apply: `True`.
- Row Laplacian: `[[0.5, -0.5], [-0.5, 0.5]]`.
- `eta = 33.2380730524`.
- `zeta = 63.3106153378`.

## Outcome
- Final global cost: `0.585048418644`.
- Best global cost in run: `0.585048418644`.
- Final alpha gradient L2: `0.178169968364`.
- Final beta gradient L2: `0.0849408493218`.
- Final alpha step L2: `0.0979482799268`.
- Final beta step L2: `0.0676946922629`.
- Total elapsed time: `2890.251869 s`.
- Mean time per iteration: `96.341729 s`.

## Notes
- This low-memory 30-qubit run does not build the dense global matrix.
- The optimization uses direct MPO block construction and `compress=False` in `A_ij |x_ij>`.
- No dense residual or exact `L2` error is reported in this workflow.

## Artifacts
- JSON: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/quimb_dist_eq26_2x2_optimize_30q_nocompress_iter200.json`
- Figure: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/quimb_dist_eq26_2x2_optimize_30q_nocompress_iter200_cost.png`
- Report: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/quimb_dist_eq26_2x2_optimize_30q_nocompress_iter200_report.md`
