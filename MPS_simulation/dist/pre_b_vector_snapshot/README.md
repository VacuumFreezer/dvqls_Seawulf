# Pre-b-vector Snapshot

This folder contains a reconstructed snapshot of the distributed MPS code before the explicit
`b -> b_i -> b_ij` implementation was introduced.

Because these files are not tracked in git in this workspace, this snapshot is reconstructed from
the current code and the recorded behavior of the earlier run. The intended behavior of this
snapshot is:

- implicit local `b_ij` handling through a shared `|+...+>` state and scalar `b_scale`
- structured deterministic initialization
  - `sigma_ij` initialized near `0.75`
  - `lambda_ij` initialized near `0.10`
  - circuit angles initialized by shifted linspace patterns

Files:

- [quimb_dist_eq26_2x2_benchmark.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/pre_b_vector_snapshot/quimb_dist_eq26_2x2_benchmark.py)
- [quimb_dist_eq26_2x2_optimize.py](/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/pre_b_vector_snapshot/quimb_dist_eq26_2x2_optimize.py)
