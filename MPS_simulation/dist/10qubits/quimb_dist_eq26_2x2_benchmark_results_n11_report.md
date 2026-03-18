# Distributed Eq. (26) 2x2 MPS Benchmark

## Setup
- Global system: 11-qubit Eq. (26) Ising-inspired linear system.
- Partition: 2x2 block decomposition with 10-qubit local operators.
- Row graph Laplacian: `[[0.5, -0.5], [-0.5, 0.5]]`.
- Column consensus weights: `[[0.5, 0.5], [0.5, 0.5]]`.

## Timings
- Forward global cost mean: `0.173947 s`.
- Reverse-mode global gradient mean: `16.518731 s`.
- One distributed iteration mean: `26.813515 s`.

## One Iteration Diagnostics
- Current global cost: `39.3949934257`.
- Alpha gradient L2 norm: `75.59842303`.
- Beta gradient L2 norm: `8.31227397115`.
- Alpha update L2 norm: `0.899487302241`.
- Beta update L2 norm: `0.899957681538`.

## Artifacts
- JSON: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/quimb_dist_eq26_2x2_benchmark_results_n11.json`
