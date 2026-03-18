# 30-Qubit Distributed Optimization Report

## Setup
- Global system: `30` qubits.
- Local block size: `29` qubits.
- Iterations: `1`.
- Learning rate: `0.01`.
- Initialization mode: `structured_linspace`.
- No-compression MPO apply: `True`.
- Row Laplacian: `[[0.5, -0.5], [-0.5, 0.5]]`.
- `eta = 33.2380730524`.
- `zeta = 63.3106153378`.

## Outcome
- Final global cost: `1.21508014438`.
- Best global cost in run: `1.21508014438`.
- Final alpha gradient L2: `1.39246685094`.
- Final beta gradient L2: `0.982391179249`.
- Final alpha step L2: `0.3010064287`.
- Final beta step L2: `0.305168446502`.
- Total elapsed time: `159.970110 s`.
- Mean time per iteration: `159.970110 s`.

## Notes
- This low-memory 30-qubit run does not build the dense global matrix.
- The optimization uses direct MPO block construction and `compress=False` in `A_ij |x_ij>`.
- No dense residual or exact `L2` error is reported in this workflow.

## Artifacts
- JSON: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/_smoke_iter1.json`
- Figure: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/_smoke_iter1_cost.png`
- Report: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/_smoke_iter1_report.md`
