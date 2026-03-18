# Distributed Eq. (26) 2x2 MPS Benchmark

## Setup
- Global system: 11-qubit Eq. (26) Ising-inspired linear system.
- Partition: 2x2 block decomposition with 10-qubit local operators.
- Row graph Laplacian: `[[1, -1], [-1, 1]]`.
- Column consensus weights: `[[0.5, 0.5], [0.5, 0.5]]`.

## Timings
- Forward global cost mean: `1.008581 s`.
- Reverse-mode global gradient mean: `15.539311 s`.
- One distributed iteration mean: `34.839053 s`.

## One Iteration Diagnostics
- Current global cost: `0.870158621731`.
- Alpha gradient L2 norm: `1.08469196931`.
- Beta gradient L2 norm: `2.12426391513`.
- Alpha update L2 norm: `0.899579577476`.
- Beta update L2 norm: `0.899856685462`.

## Artifacts
- JSON: `/home/patchouli/projects/Distributed_vqls/MPS_simulation/dist/pilots/quimb_dist_eq26_2x2_benchmark_pilot.json`
