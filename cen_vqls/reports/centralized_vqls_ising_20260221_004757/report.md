# Centralized VQLS Report

- Timestamp: 2026-02-21T00:47:57
- Static ops path: `/home/patchouli/projects/Distributed_vqls/4agents_Ising/static_ops_16agents_Ising.py`
- Total qubits: 9
- Agents: 4 (index qubits=2, local qubits=7)

## Matrix Consistency
- Formula vs block max abs diff: 0.000000e+00
- Formula vs block Frobenius diff: 0.000000e+00
- allclose(atol=1e-12, rtol=0): True

## b-Vector and Unitary
- ||b_unnorm||: 8.000000000000
- b unitary mode: hadamard_tensor
- ||b_normed - |+>^n||: 3.140185e-16
- ||U_b|0> - b_normed|| (phase-aligned): 3.140185e-16

## Optimization Summary
- Optimize metric: global_CG
- Steps: 2000
- Learning rate: 0.03
- Adam betas: (0.9, 0.999)
- Adam eps: 1e-08
- Best iteration: 2000
- Best global_CG: 4.017464818928e-06

## Final Metrics
- global_CG: 4.008808633160e-06
- global_CL: 9.375778189935e-07
- global_CG_hat: 4.378442459654e-04
- global_CL_hat: 1.024027562124e-04
- beta: 1.092205405755e+02
- overlap_sq: 1.092201027312e+02
- L2 abs error (unnormalized x): 1.590919164154e-02
- L2 rel error (unnormalized x): 2.078720660541e-02

## Files
- History: `/home/patchouli/projects/Distributed_vqls/cen_vqls/reports/centralized_vqls_ising_20260221_004757/history.json`
- Checkpoints: `/home/patchouli/projects/Distributed_vqls/cen_vqls/reports/centralized_vqls_ising_20260221_004757/checkpoints.json`
- Config used: `/home/patchouli/projects/Distributed_vqls/cen_vqls/reports/centralized_vqls_ising_20260221_004757/config_used.yaml`

## Hyperparameters
```yaml
problem:
  static_ops_path: 4agents_Ising/static_ops_16agents_Ising.py
  consistency_atol: 1e-12
  b_state_tolerance: 1e-10
ansatz:
  layers: 6
  init_low: -3.141592653589793
  init_high: 3.141592653589793
optimization:
  steps: 2000
  learning_rate: 0.03
  beta1: 0.9
  beta2: 0.999
  eps: 1e-08
  print_every: 20
  optimize_metric: global_CG
runtime:
  seed: 0
  device: default.qubit
  interface: autograd
  diff_method: backprop
report:
  out_dir: cen_vqls/reports
  tag: centralized_vqls_ising
```
