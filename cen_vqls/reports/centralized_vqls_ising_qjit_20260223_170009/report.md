# Centralized VQLS Report

- Timestamp: 2026-02-23T17:00:09
- Static ops path: `/home/patchouli/projects/Distributed_vqls/4agents_Ising/static_ops_16agents_Ising.py`
- Total qubits: 7
- Agents: 4 (index qubits=2, local qubits=5)

## Matrix Consistency
- Formula vs block max abs diff: 0.000000e+00
- Formula vs block Frobenius diff: 0.000000e+00
- allclose(atol=1e-12, rtol=0): True

## b-Vector and Unitary
- ||b_unnorm||: 8.000000000000
- b unitary mode: hadamard_tensor
- ||b_normed - |+>^n||: 1.570092e-16
- ||U_b|0> - b_normed|| (phase-aligned): 3.140185e-16

## Optimization Summary
- Optimize metric: global_CG
- Steps: 200
- Learning rate: 0.03
- Adam betas: (0.9, 0.999)
- Adam eps: 1e-08
- Best iteration: 200
- Best global_CG: 9.747080763589e-04

## Final Metrics
- global_CG: 9.694464822020e-04
- global_CL: 2.261457622674e-04
- global_CG_hat: 6.253325107140e-02
- global_CL_hat: 1.458732378757e-02
- beta: 6.450407755298e+01
- overlap_sq: 6.444154430190e+01
- L2 abs error (unnormalized x): 1.912942767344e+00
- L2 rel error (unnormalized x): 2.027220159256e+00

## Files
- History: `/home/patchouli/projects/Distributed_vqls/cen_vqls/reports/centralized_vqls_ising_qjit_20260223_170009/history.json`
- Checkpoints: `/home/patchouli/projects/Distributed_vqls/cen_vqls/reports/centralized_vqls_ising_qjit_20260223_170009/checkpoints.json`
- Config used: `/home/patchouli/projects/Distributed_vqls/cen_vqls/reports/centralized_vqls_ising_qjit_20260223_170009/config_used.yaml`

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
  steps: 200
  learning_rate: 0.03
  beta1: 0.9
  beta2: 0.999
  eps: 1e-08
  print_every: 20
  optimize_metric: global_CG
runtime:
  seed: 0
  device: lightning.qubit
  interface: jax
  diff_method: adjoint
  use_qjit: true
  fallback_to_jax_jit: false
  qjit_autograph: false
  jit_training_step: true
  warmup_compile: true
report:
  out_dir: cen_vqls/reports
  tag: centralized_vqls_ising_qjit
```

## QJIT Backend
- Requested qjit: True
- Active backend: qjit
- Catalyst available: True
- qjit_autograph: False
- jit_training_step: True
