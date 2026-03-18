# Distributed Optimization Report

## Setup
- Global system: `6` qubits.
- Local block size: `5` qubits.
- Iterations: `20`.
- Learning rate: `0.01`.
- Initialization mode: `structured_linspace`.
- No-compression MPO apply: `True`.
- Row Laplacian: `[[0.5, -0.5], [-0.5, 0.5]]`.
- `eta = 6.64539992206`.
- `zeta = 12.6579046134`.

## Outcome
- Final global cost: `0.136901823442`.
- Best global cost in run: `0.136901823442`.
- Final alpha gradient L2: `0.117698720937`.
- Final beta gradient L2: `0.173095968266`.
- Final alpha step L2: `0.0705709642418`.
- Final beta step L2: `0.0702944213904`.
- Total elapsed time: `443.055158 s`.
- Mean time per iteration: `22.152758 s`.

## Notes
- This low-memory 30-qubit run does not build the dense global matrix.
- The optimization uses direct MPO block construction and `compress=False` in `A_ij |x_ij>`.
- No dense residual or exact `L2` error is reported in this workflow.

## Artifacts
- JSON: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/5qubits/quimb_dist_eq26_2x2_optimize_lowmem_n6_local5_k20_nocompress_iter200.json`
- Figure: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/5qubits/quimb_dist_eq26_2x2_optimize_lowmem_n6_local5_k20_nocompress_iter200_cost.png`
- Report: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/5qubits/quimb_dist_eq26_2x2_optimize_lowmem_n6_local5_k20_nocompress_iter200_report.md`
