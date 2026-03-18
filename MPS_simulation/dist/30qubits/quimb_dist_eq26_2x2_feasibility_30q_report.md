# 30-Qubit Feasibility Check

## Conclusion
The current distributed workflow is not feasible at 30 global qubits as written.
The MPS circuit ansatz itself is fine at 29 local qubits, but the current operator and RHS construction path is not scalable.

## Quick Check
- Pure 29-qubit MPS circuit build time: `0.602340 s`.
- Circuit state norm: `0.999999999413`.
- Max MPS bond during the smoke test: `4`.
- Number of MPS tensors: `29`.

## Current Workflow Bottlenecks
- The distributed code builds a full sparse global matrix with `build_sparse(global_qubits)`.
  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:168` and `MPS_simulation/dist/quimb_dist_eq26_2x2_optimize.py:165`.
- It then converts the full global matrix to dense with `a_sparse.toarray()`.
  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:180` and `MPS_simulation/dist/quimb_dist_eq26_2x2_optimize.py:591`.
- Local MPOs are constructed from dense blocks using `MatrixProductOperator.from_dense(...)`.
  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:148`.
- The RHS `b` is also materialized as a dense global vector and then split into dense local vectors.
  File: `MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark.py:191-216`.

## Size Estimates For 30 Global Qubits
- Global dimension: `1073741824`.
- Local block dimension: `536870912`.
- Dense global matrix size: `16.000 EiB`.
- Dense local block size: `4.000 EiB`.
- Dense global `b` vector size: `16.000 GiB`.
- Dense local `b_ij` vector size: `8.000 GiB`.
- Sparse nnz lower bound for the full matrix: `33285996544`.
- Float64 CSR storage lower bound for the full matrix: `380.000 GiB`.
- Detected system RAM: `7.605 GiB`.
- Dense global matrix / RAM ratio: `2.259e+09`.
- Dense local block / RAM ratio: `5.647e+08`.
- Dense global `b` vector / RAM ratio: `2.104e+00`.

## Practical Reading
A 30-qubit run is not blocked by the MPS ansatz circuit. It is blocked by the dense and sparse matrix construction used to get `A_ij` and `b_ij` in the current workflow.
To make 30 global qubits feasible, we would need to replace the current matrix extraction path with direct MPO/MPS constructions for the local operators and RHS blocks.

## Artifacts
- JSON: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/quimb_dist_eq26_2x2_feasibility_30q.json`
- Report: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/30qubits/quimb_dist_eq26_2x2_feasibility_30q_report.md`
